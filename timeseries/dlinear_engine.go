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
	windows, _, _ = normalizeWindows(windows)

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

			grads := make([]*tensor.TensorNumeric[float32], len(allParams))
			for i, ps := range allParams {
				sz := 1
				for _, s := range ps.param.Shape() {
					sz *= s
				}
				var err error
				grads[i], err = tensor.New[float32](ps.param.Shape(), make([]float32, sz))
				if err != nil {
					return nil, err
				}
			}

			batchLoss := float32(0)

			for s := 0; s < bs; s++ {
				sample := windows[start+s]
				trend, seasonal := d.decompose(sample)

				for c := 0; c < channels; c++ {
					trendIn, err := tensor.New[float32]([]int{1, inputLen}, float64ToFloat32(trend[c]))
					if err != nil {
						return nil, err
					}
					seasonIn, err := tensor.New[float32]([]int{1, inputLen}, float64ToFloat32(seasonal[c]))
					if err != nil {
						return nil, err
					}

					trendInT, err := eng.Transpose(ctx, trendIn, []int{1, 0})
					if err != nil {
						return nil, fmt.Errorf("dlinear engine: transpose trend: %w", err)
					}
					trendOut, err := eng.MatMul(ctx, trendWT[c], trendInT)
					if err != nil {
						return nil, fmt.Errorf("dlinear engine: matmul trend: %w", err)
					}
					trendOut, err = eng.Reshape(ctx, trendOut, []int{1, outputLen})
					if err != nil {
						return nil, fmt.Errorf("dlinear engine: reshape trend out: %w", err)
					}
					trendOut, err = eng.Add(ctx, trendOut, trendBT[c])
					if err != nil {
						return nil, fmt.Errorf("dlinear engine: add trend bias: %w", err)
					}

					seasonInT, err := eng.Transpose(ctx, seasonIn, []int{1, 0})
					if err != nil {
						return nil, fmt.Errorf("dlinear engine: transpose seasonal: %w", err)
					}
					seasonOut, err := eng.MatMul(ctx, seasonWT[c], seasonInT)
					if err != nil {
						return nil, fmt.Errorf("dlinear engine: matmul seasonal: %w", err)
					}
					seasonOut, err = eng.Reshape(ctx, seasonOut, []int{1, outputLen})
					if err != nil {
						return nil, fmt.Errorf("dlinear engine: reshape seasonal out: %w", err)
					}
					seasonOut, err = eng.Add(ctx, seasonOut, seasonBT[c])
					if err != nil {
						return nil, fmt.Errorf("dlinear engine: add seasonal bias: %w", err)
					}

					pred, err := eng.Add(ctx, trendOut, seasonOut)
					if err != nil {
						return nil, fmt.Errorf("dlinear engine: add outputs: %w", err)
					}

					labelStart := s*channels*outputLen + c*outputLen
					labelT, err := tensor.New[float32]([]int{1, outputLen}, float64ToFloat32(batchLabels[labelStart:labelStart+outputLen]))
					if err != nil {
						return nil, err
					}

					diff, err := eng.Sub(ctx, pred, labelT)
					if err != nil {
						return nil, fmt.Errorf("dlinear engine: sub: %w", err)
					}

					diffSq, err := eng.Mul(ctx, diff, diff)
					if err != nil {
						return nil, fmt.Errorf("dlinear engine: mul diff: %w", err)
					}
					lossTensor, err := eng.Sum(ctx, diffSq, -1, false)
					if err != nil {
						return nil, fmt.Errorf("dlinear engine: sum loss: %w", err)
					}
					batchLoss += lossTensor.Data()[0]

					scale := float32(2.0) / float32(bs*channels*outputLen)
					dOut, err := eng.MulScalar(ctx, diff, scale)
					if err != nil {
						return nil, fmt.Errorf("dlinear engine: scale grad: %w", err)
					}

					paramBase := c * 4

					dOutT, err := eng.Transpose(ctx, dOut, []int{1, 0})
					if err != nil {
						return nil, err
					}
					dTrendW, err := eng.MatMul(ctx, dOutT, trendIn)
					if err != nil {
						return nil, err
					}
					grads[paramBase], err = eng.Add(ctx, grads[paramBase], dTrendW)
					if err != nil {
						return nil, err
					}

					grads[paramBase+1], err = eng.Add(ctx, grads[paramBase+1], dOut)
					if err != nil {
						return nil, err
					}

					dSeasonW, err := eng.MatMul(ctx, dOutT, seasonIn)
					if err != nil {
						return nil, err
					}
					grads[paramBase+2], err = eng.Add(ctx, grads[paramBase+2], dSeasonW)
					if err != nil {
						return nil, err
					}

					grads[paramBase+3], err = eng.Add(ctx, grads[paramBase+3], dOut)
					if err != nil {
						return nil, err
					}
				}
			}

			batchLossF64 := float64(batchLoss) / float64(bs*channels*outputLen)
			epochLoss += batchLossF64
			nBatches++

			if config.GradClip > 0 {
				norm := float64(0)
				for _, g := range grads {
					for _, v := range g.Data() {
						norm += float64(v) * float64(v)
					}
				}
				norm = math.Sqrt(norm)
				if norm > config.GradClip {
					clipScale := float32(config.GradClip / norm)
					for i, g := range grads {
						clipped, err := eng.MulScalar(ctx, g, clipScale)
						if err != nil {
							return nil, err
						}
						grads[i] = clipped
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
				gData := grads[i].Data()
				mData := ps.adam.m.Data()
				vData := ps.adam.v.Data()
				pData := ps.param.Data()

				for j := range pData {
					mData[j] = beta1*mData[j] + (1-beta1)*gData[j]
					vData[j] = beta2*vData[j] + (1-beta2)*gData[j]*gData[j]
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
