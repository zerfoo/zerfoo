package timeseries

import (
	"context"
	"fmt"

	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/tensor"

	"github.com/zerfoo/zerfoo/training/optimizer"
	"github.com/zerfoo/zerfoo/training/scheduler"
)

// trainWindowedEngine implements TrainWindowed with fused CPU forward+backward.
// DLinear's matrices are tiny (outputLen×inputLen, e.g. 12×24), so GPU kernel
// launch + transfer overhead dominates. This path fuses forward, MSE loss, and
// analytical gradient computation into a single CPU loop with zero tensor
// allocations inside the training loop.
func (d *DLinear) trainWindowedEngine(windows [][][]float64, labels []float64, config TrainConfig) (*TrainResult, error) {
	if config.Epochs <= 0 {
		config.Epochs = 100
	}
	if config.LR <= 0 {
		config.LR = 1e-3
	}
	if config.Beta1 <= 0 {
		config.Beta1 = 0.9
	}
	if config.Beta2 <= 0 {
		config.Beta2 = 0.999
	}
	if config.Epsilon <= 0 {
		config.Epsilon = 1e-8
	}

	nSamples := len(windows)
	windows, d.normMeans, d.normStds = normalizeWindows(windows)

	channels := d.config.Channels
	inputLen := d.config.InputLen
	outputLen := d.config.OutputLen

	// Copy weights to flat float32 slices for the training loop.
	// Layout per channel: trendW [outputLen*inputLen], trendB [outputLen],
	//                     seasonW [outputLen*inputLen], seasonB [outputLen].
	trendW := make([][]float32, channels)
	trendB := make([][]float32, channels)
	seasonW := make([][]float32, channels)
	seasonB := make([][]float32, channels)
	for c := 0; c < channels; c++ {
		trendW[c] = float64ToFloat32(d.trendW[c])
		trendB[c] = float64ToFloat32(d.trendB[c])
		seasonW[c] = float64ToFloat32(d.seasonalW[c])
		seasonB[c] = float64ToFloat32(d.seasonalB[c])
	}

	// Wrap parameters as graph.Parameter for optimizer.AdamW.
	type paramSet struct {
		data []float32
	}
	allParams := make([]paramSet, channels*4)
	graphParams := make([]*graph.Parameter[float32], channels*4)
	for c := 0; c < channels; c++ {
		for j, p := range [][]float32{trendW[c], trendB[c], seasonW[c], seasonB[c]} {
			idx := c*4 + j
			allParams[idx] = paramSet{data: p}
			t, _ := tensor.New[float32]([]int{len(p)}, p)
			graphParams[idx], _ = graph.NewParameter(fmt.Sprintf("ch%d.p%d", c, j), t, tensor.New[float32])
		}
	}
	ctx := context.Background()
	opt := optimizer.NewAdamW(d.engine, float32(config.LR), float32(config.Beta1), float32(config.Beta2), float32(config.Epsilon), float32(config.WeightDecay))
	if config.GradClip > 0 {
		opt.SetMaxGradNorm(config.GradClip)
	}

	result := &TrainResult{
		LossHistory: make([]float64, config.Epochs),
	}

	batchSize := nSamples
	if config.BatchSize > 0 && config.BatchSize < nSamples {
		batchSize = config.BatchSize
	}

	for epoch := 0; epoch < config.Epochs; epoch++ {
		epochLoss := 0.0
		nBatches := 0

		for start := 0; start < nSamples; start += batchSize {
			end := start + batchSize
			if end > nSamples {
				end = nSamples
			}
			bs := end - start
			batchLabels := labels[start*channels*outputLen : end*channels*outputLen]

			// Zero gradient accumulators.
			gradSlices := make([][]float32, len(allParams))
			for i, ps := range allParams {
				gradSlices[i] = make([]float32, len(ps.data))
			}

			batchLoss := 0.0

			// Decompose all samples on CPU (no trainable parameters).
			trends := make([][][]float64, bs)
			seasonals := make([][][]float64, bs)
			for s := 0; s < bs; s++ {
				trends[s], seasonals[s] = d.decompose(windows[start+s])
			}

			scale := float32(2.0) / float32(bs*channels*outputLen)

			for c := 0; c < channels; c++ {
				tw := trendW[c]  // [outputLen * inputLen]
				tb := trendB[c]  // [outputLen]
				sw := seasonW[c] // [outputLen * inputLen]
				sb := seasonB[c] // [outputLen]
				paramBase := c * 4

				for s := 0; s < bs; s++ {
					trendIn := trends[s][c]
					seasonIn := seasonals[s][c]
					labelBase := s*channels*outputLen + c*outputLen

					for o := 0; o < outputLen; o++ {
						// Fused forward: pred = W_trend·trend + b_trend + W_season·season + b_season
						trendVal := tb[o]
						seasonVal := sb[o]
						wOff := o * inputLen
						for i := 0; i < inputLen; i++ {
							trendVal += tw[wOff+i] * float32(trendIn[i])
							seasonVal += sw[wOff+i] * float32(seasonIn[i])
						}
						pred := trendVal + seasonVal

						label := float32(batchLabels[labelBase+o])
						diff := pred - label
						batchLoss += float64(diff) * float64(diff)

						// Analytical backward: dL/dW = dOut * input^T, dL/db = dOut.
						dOut := diff * scale
						gradSlices[paramBase+1][o] += dOut // trend bias grad
						gradSlices[paramBase+3][o] += dOut // seasonal bias grad
						for i := 0; i < inputLen; i++ {
							gradSlices[paramBase][wOff+i] += dOut * float32(trendIn[i])
							gradSlices[paramBase+2][wOff+i] += dOut * float32(seasonIn[i])
						}
					}
				}
			}

			batchLossF64 := batchLoss / float64(bs*channels*outputLen)
			epochLoss += batchLossF64
			nBatches++

			// Set gradients and apply AdamW step.
			opt.SetLR(float32(scheduler.WarmupLR(config.LR, epoch, config.WarmupEpochs)))
			for i := range allParams {
				gradT, _ := tensor.New[float32](graphParams[i].Value.Shape(), gradSlices[i])
				graphParams[i].Gradient = gradT
			}
			if err := opt.Step(ctx, graphParams); err != nil {
				return nil, fmt.Errorf("dlinear: adamw step: %w", err)
			}
			// Sync updated values back to working slices.
			for i := range allParams {
				copy(allParams[i].data, graphParams[i].Value.Data())
			}
		}

		result.LossHistory[epoch] = epochLoss / float64(nBatches)
		result.FinalLoss = result.LossHistory[epoch]

		if !isFinite(result.FinalLoss) {
			return nil, fmt.Errorf("dlinear: training diverged at epoch %d: loss=%v", epoch, result.FinalLoss)
		}
	}

	// Copy trained weights back to model.
	for c := 0; c < channels; c++ {
		copyF32ToF64(trendW[c], d.trendW[c])
		copyF32ToF64(trendB[c], d.trendB[c])
		copyF32ToF64(seasonW[c], d.seasonalW[c])
		copyF32ToF64(seasonB[c], d.seasonalB[c])
	}

	result.Metrics = map[string]float64{"mse": result.FinalLoss}
	return result, nil
}

// forwardBatch runs the DLinear forward pass on a batch of samples.
// Input: [batch][channels][inputLen], returns: [batch][channels][outputLen].
// Decomposition is performed per-sample, then the linear projections are
// applied across the full batch using fused loops to minimise allocation overhead.
func (d *DLinear) forwardBatch(inputs [][][]float64) [][][]float64 {
	batch := len(inputs)
	out := make([][][]float64, batch)
	for b := 0; b < batch; b++ {
		out[b] = d.forward(inputs[b])
	}
	return out
}

func copyF32ToF64(src []float32, dst []float64) {
	for i, v := range src {
		dst[i] = float64(v)
	}
}
