package timeseries

import (
	"context"
	"fmt"
	"math"

	"github.com/zerfoo/ztensor/tensor"
)

// trainWindowedEngine implements TrainWindowed using the compute engine for
// GPU-accelerated tensor operations. The moving-average decomposition is
// performed on the CPU (it has no trainable parameters), then the linear
// projections, MSE loss, and gradient computation are executed through the
// engine. Weight updates use AdamW identical to the CPU path.
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

	ctx := context.Background()
	eng := d.engine
	channels := d.config.Channels
	inputLen := d.config.InputLen
	outputLen := d.config.OutputLen

	trendWT := make([]*tensor.TensorNumeric[float32], channels)
	trendBT := make([]*tensor.TensorNumeric[float32], channels)
	seasonWT := make([]*tensor.TensorNumeric[float32], channels)
	seasonBT := make([]*tensor.TensorNumeric[float32], channels)
	for c := 0; c < channels; c++ {
		var err error
		if trendWT[c], err = tensor.New[float32]([]int{outputLen, inputLen}, float64ToFloat32(d.trendW[c])); err != nil {
			return nil, fmt.Errorf("dlinear engine: trend weight tensor: %w", err)
		}
		if trendBT[c], err = tensor.New[float32]([]int{1, outputLen}, float64ToFloat32(d.trendB[c])); err != nil {
			return nil, fmt.Errorf("dlinear engine: trend bias tensor: %w", err)
		}
		if seasonWT[c], err = tensor.New[float32]([]int{outputLen, inputLen}, float64ToFloat32(d.seasonalW[c])); err != nil {
			return nil, fmt.Errorf("dlinear engine: seasonal weight tensor: %w", err)
		}
		if seasonBT[c], err = tensor.New[float32]([]int{1, outputLen}, float64ToFloat32(d.seasonalB[c])); err != nil {
			return nil, fmt.Errorf("dlinear engine: seasonal bias tensor: %w", err)
		}
	}

	type adamState struct {
		m *tensor.TensorNumeric[float32]
		v *tensor.TensorNumeric[float32]
	}
	newAdamState := func(shape []int) (*adamState, error) {
		sz := 1
		for _, s := range shape {
			sz *= s
		}
		mT, err := tensor.New[float32](shape, make([]float32, sz))
		if err != nil {
			return nil, err
		}
		vT, err := tensor.New[float32](shape, make([]float32, sz))
		if err != nil {
			return nil, err
		}
		return &adamState{m: mT, v: vT}, nil
	}

	type paramSet struct {
		param *tensor.TensorNumeric[float32]
		adam  *adamState
	}
	var allParams []paramSet
	for c := 0; c < channels; c++ {
		for _, pt := range []*tensor.TensorNumeric[float32]{trendWT[c], trendBT[c], seasonWT[c], seasonBT[c]} {
			as, err := newAdamState(pt.Shape())
			if err != nil {
				return nil, fmt.Errorf("dlinear engine: adam state: %w", err)
			}
			allParams = append(allParams, paramSet{param: pt, adam: as})
		}
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

			// CPU gradient accumulators — one flat slice per parameter.
			gradSlices := make([][]float32, len(allParams))
			for i, ps := range allParams {
				gradSlices[i] = make([]float32, len(ps.param.Data()))
			}

			batchLoss := 0.0

			// Decompose all samples on CPU (no trainable parameters).
			trends := make([][][]float64, bs)
			seasonals := make([][][]float64, bs)
			for s := 0; s < bs; s++ {
				trends[s], seasonals[s] = d.decompose(windows[start+s])
			}

			for c := 0; c < channels; c++ {
				// Gather batched inputs: [bs, inputLen].
				trendData := make([]float32, bs*inputLen)
				seasonData := make([]float32, bs*inputLen)
				for s := 0; s < bs; s++ {
					for i := 0; i < inputLen; i++ {
						trendData[s*inputLen+i] = float32(trends[s][c][i])
						seasonData[s*inputLen+i] = float32(seasonals[s][c][i])
					}
				}

				// GPU: one batched MatMul for trend.
				// W [outputLen, inputLen] × X^T [inputLen, bs] = [outputLen, bs]
				trendBatch, err := tensor.New[float32]([]int{bs, inputLen}, trendData)
				if err != nil {
					return nil, err
				}
				trendBatchT, err := eng.Transpose(ctx, trendBatch, []int{1, 0})
				if err != nil {
					return nil, fmt.Errorf("dlinear engine: transpose trend batch: %w", err)
				}
				trendOutT, err := eng.MatMul(ctx, trendWT[c], trendBatchT)
				if err != nil {
					return nil, fmt.Errorf("dlinear engine: matmul trend batch: %w", err)
				}

				// GPU: one batched MatMul for seasonal.
				seasonBatch, err := tensor.New[float32]([]int{bs, inputLen}, seasonData)
				if err != nil {
					return nil, err
				}
				seasonBatchT, err := eng.Transpose(ctx, seasonBatch, []int{1, 0})
				if err != nil {
					return nil, fmt.Errorf("dlinear engine: transpose seasonal batch: %w", err)
				}
				seasonOutT, err := eng.MatMul(ctx, seasonWT[c], seasonBatchT)
				if err != nil {
					return nil, fmt.Errorf("dlinear engine: matmul seasonal batch: %w", err)
				}

				// CPU: read back results and compute loss + gradients analytically.
				// trendOutT, seasonOutT are [outputLen, bs] row-major.
				trendOut := trendOutT.Data()
				seasonOut := seasonOutT.Data()
				trendBias := trendBT[c].Data()
				seasonBias := seasonBT[c].Data()

				paramBase := c * 4
				scale := float32(2.0) / float32(bs*channels*outputLen)

				for s := 0; s < bs; s++ {
					labelBase := s*channels*outputLen + c*outputLen
					for o := 0; o < outputLen; o++ {
						pred := trendOut[o*bs+s] + trendBias[o] + seasonOut[o*bs+s] + seasonBias[o]
						label := float32(batchLabels[labelBase+o])
						diff := pred - label
						batchLoss += float64(diff) * float64(diff)

						dOut := diff * scale
						gradSlices[paramBase+1][o] += dOut // trend bias
						gradSlices[paramBase+3][o] += dOut // seasonal bias
						for i := 0; i < inputLen; i++ {
							gradSlices[paramBase][o*inputLen+i] += dOut * trendData[s*inputLen+i]
							gradSlices[paramBase+2][o*inputLen+i] += dOut * seasonData[s*inputLen+i]
						}
					}
				}
			}

			batchLossF64 := batchLoss / float64(bs*channels*outputLen)
			epochLoss += batchLossF64
			nBatches++

			if config.GradClip > 0 {
				norm := 0.0
				for _, g := range gradSlices {
					for _, v := range g {
						norm += float64(v) * float64(v)
					}
				}
				norm = math.Sqrt(norm)
				if norm > config.GradClip {
					clipScale := float32(config.GradClip / norm)
					for _, g := range gradSlices {
						for j := range g {
							g[j] *= clipScale
						}
					}
				}
			}

			lr := float32(warmupLR(config.LR, epoch, config.WarmupEpochs))
			t := float32(epoch*((nSamples+batchSize-1)/batchSize) + nBatches)
			beta1 := float32(config.Beta1)
			beta2 := float32(config.Beta2)
			eps := float32(config.Epsilon)
			wd := float32(config.WeightDecay)

			bc1 := float32(1.0 - math.Pow(float64(beta1), float64(t)))
			bc2 := float32(1.0 - math.Pow(float64(beta2), float64(t)))

			for i, ps := range allParams {
				mData := ps.adam.m.Data()
				vData := ps.adam.v.Data()
				pData := ps.param.Data()

				for j := range pData {
					g := gradSlices[i][j]
					mData[j] = beta1*mData[j] + (1-beta1)*g
					vData[j] = beta2*vData[j] + (1-beta2)*g*g
					mHat := mData[j] / bc1
					vHat := vData[j] / bc2
					pData[j] -= lr * (mHat/(float32(math.Sqrt(float64(vHat)))+eps) + wd*pData[j])
				}
			}
		}

		result.LossHistory[epoch] = epochLoss / float64(nBatches)
		result.FinalLoss = result.LossHistory[epoch]

		if !isFinite(result.FinalLoss) {
			return nil, fmt.Errorf("dlinear: training diverged at epoch %d: loss=%v", epoch, result.FinalLoss)
		}
	}

	for c := 0; c < channels; c++ {
		copyF32ToF64(trendWT[c].Data(), d.trendW[c])
		copyF32ToF64(trendBT[c].Data(), d.trendB[c])
		copyF32ToF64(seasonWT[c].Data(), d.seasonalW[c])
		copyF32ToF64(seasonBT[c].Data(), d.seasonalB[c])
	}

	result.Metrics = map[string]float64{"mse": result.FinalLoss}
	return result, nil
}

func copyF32ToF64(src []float32, dst []float64) {
	for i, v := range src {
		dst[i] = float64(v)
	}
}
