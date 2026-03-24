package timeseries

import (
	"context"
	"fmt"
	"math"

	"github.com/zerfoo/ztensor/tensor"
)

// flatParamSlices returns references to all trainable float32 parameter data slices.
func (m *PatchTST) flatParamSlices() [][]float32 {
	var slices [][]float32
	slices = append(slices, m.patchEmb.weights.Data(), m.patchEmb.biases.Data())
	slices = append(slices, m.posEmb.Data())
	for i := range m.layers {
		l := &m.layers[i]
		slices = append(slices,
			l.qProj.weights.Data(), l.qProj.biases.Data(),
			l.kProj.weights.Data(), l.kProj.biases.Data(),
			l.vProj.weights.Data(), l.vProj.biases.Data(),
			l.oProj.weights.Data(), l.oProj.biases.Data(),
			l.ffn1.weights.Data(), l.ffn1.biases.Data(),
			l.ffn2.weights.Data(), l.ffn2.biases.Data(),
			l.norm1.Data(), l.bias1.Data(),
			l.norm2.Data(), l.bias2.Data(),
		)
	}
	slices = append(slices, m.head.weights.Data(), m.head.biases.Data())
	return slices
}

// trainWindowedEngine runs the GPU/engine-accelerated training path.
// It uses the engine-based Forward for all forward passes and computes
// numerical gradients by perturbing float32 parameters directly.
func (m *PatchTST) trainWindowedEngine(windows [][][]float64, labels []float64, config TrainConfig) (*TrainResult, error) {
	nSamples := len(windows)
	outDim := m.config.OutputDim
	channels := len(windows[0])
	ctx := context.Background()

	paramSlices := m.flatParamSlices()
	nParams := 0
	for _, s := range paramSlices {
		nParams += len(s)
	}

	adamM := make([]float64, nParams)
	adamV := make([]float64, nParams)

	result := &TrainResult{
		LossHistory: make([]float64, config.Epochs),
	}

	batchSize := nSamples
	if config.BatchSize > 0 && config.BatchSize < nSamples {
		batchSize = config.BatchSize
	}

	buildInput := func(ws [][][]float64) (*tensor.TensorNumeric[float32], error) {
		bs := len(ws)
		data := make([]float32, bs*channels*m.config.InputLength)
		for s := 0; s < bs; s++ {
			for c := 0; c < channels; c++ {
				off := (s*channels + c) * m.config.InputLength
				for i, v := range ws[s][c] {
					data[off+i] = float32(v)
				}
			}
		}
		if channels == 1 {
			return tensor.New[float32]([]int{bs, m.config.InputLength}, data)
		}
		return tensor.New[float32]([]int{bs, channels, m.config.InputLength}, data)
	}

	avgChannels := func(pred *tensor.TensorNumeric[float32], bs int) []float64 {
		predData := pred.Data()
		shape := pred.Shape()
		avg := make([]float64, bs*outDim)
		if len(shape) == 3 && shape[1] > 1 {
			ch := shape[1]
			for s := 0; s < bs; s++ {
				for o := 0; o < outDim; o++ {
					sum := 0.0
					for c := 0; c < ch; c++ {
						sum += float64(predData[(s*ch+c)*outDim+o])
					}
					avg[s*outDim+o] = sum / float64(ch)
				}
			}
		} else {
			for i, v := range predData {
				avg[i] = float64(v)
			}
		}
		return avg
	}

	computeMSE := func(avgPred []float64, lbls []float64) float64 {
		loss := 0.0
		for i, v := range avgPred {
			diff := v - lbls[i]
			loss += diff * diff
		}
		return loss / float64(len(avgPred))
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
			batchLabels := labels[start*outDim : end*outDim]

			inputT, err := buildInput(windows[start:end])
			if err != nil {
				return nil, fmt.Errorf("patchtst: build input: %w", err)
			}
			pred, err := m.Forward(ctx, inputT)
			if err != nil {
				return nil, fmt.Errorf("patchtst: forward: %w", err)
			}

			avgPred := avgChannels(pred, bs)
			batchLoss := computeMSE(avgPred, batchLabels)

			grads := make([]float64, nParams)
			eps := float32(1e-4)
			gi := 0
			for _, slice := range paramSlices {
				for pi := range slice {
					orig := slice[pi]

					slice[pi] = orig + eps
					predPlus, fErr := m.Forward(ctx, inputT)
					if fErr != nil {
						return nil, fmt.Errorf("patchtst: forward +eps: %w", fErr)
					}
					lossPlus := computeMSE(avgChannels(predPlus, bs), batchLabels)

					slice[pi] = orig - eps
					predMinus, fErr := m.Forward(ctx, inputT)
					if fErr != nil {
						return nil, fmt.Errorf("patchtst: forward -eps: %w", fErr)
					}
					lossMinus := computeMSE(avgChannels(predMinus, bs), batchLabels)

					slice[pi] = orig
					grads[gi] = (lossPlus - lossMinus) / (2 * float64(eps))
					gi++
				}
			}

			epochLoss += batchLoss
			nBatches++

			if config.GradClip > 0 {
				norm := 0.0
				for _, g := range grads {
					norm += g * g
				}
				norm = math.Sqrt(norm)
				if norm > config.GradClip {
					scale := config.GradClip / norm
					for i := range grads {
						grads[i] *= scale
					}
				}
			}

			lr := warmupLR(config.LR, epoch, config.WarmupEpochs)
			t := float64(epoch*((nSamples+batchSize-1)/batchSize) + nBatches)
			gi = 0
			for _, slice := range paramSlices {
				for pi := range slice {
					adamM[gi] = config.Beta1*adamM[gi] + (1-config.Beta1)*grads[gi]
					adamV[gi] = config.Beta2*adamV[gi] + (1-config.Beta2)*grads[gi]*grads[gi]
					mHat := adamM[gi] / (1 - math.Pow(config.Beta1, t))
					vHat := adamV[gi] / (1 - math.Pow(config.Beta2, t))
					slice[pi] -= float32(lr * (mHat/(math.Sqrt(vHat)+config.Epsilon) + config.WeightDecay*float64(slice[pi])))
					gi++
				}
			}
		}

		result.LossHistory[epoch] = epochLoss / float64(nBatches)
		result.FinalLoss = result.LossHistory[epoch]

		if !isFinite(result.FinalLoss) {
			return nil, fmt.Errorf("patchtst: training diverged at epoch %d: loss=%v", epoch, result.FinalLoss)
		}
	}

	result.Metrics = map[string]float64{"mse": result.FinalLoss}
	return result, nil
}

// trainWindowedCPU runs the original CPU-only training path using float64
// parameters and forwardF64. Used as fallback when no engine is available.
func (m *PatchTST) trainWindowedCPU(windows [][][]float64, labels []float64, config TrainConfig) (*TrainResult, error) {
	nSamples := len(windows)

	params := m.extractParamsF64()
	nParams := params.paramCount()
	adamM := make([]float64, nParams)
	adamV := make([]float64, nParams)

	result := &TrainResult{
		LossHistory: make([]float64, config.Epochs),
	}

	batchSize := nSamples
	if config.BatchSize > 0 && config.BatchSize < nSamples {
		batchSize = config.BatchSize
	}

	outDim := m.config.OutputDim

	for epoch := 0; epoch < config.Epochs; epoch++ {
		epochLoss := 0.0
		nBatches := 0

		for start := 0; start < nSamples; start += batchSize {
			end := start + batchSize
			if end > nSamples {
				end = nSamples
			}
			bs := end - start

			grads := make([]float64, nParams)
			batchLoss := 0.0

			for s := 0; s < bs; s++ {
				pred := m.forwardF64(windows[start+s], params)
				sampleLabels := labels[(start+s)*outDim : (start+s+1)*outDim]

				for j := 0; j < outDim; j++ {
					diff := pred[j] - sampleLabels[j]
					batchLoss += diff * diff
				}

				eps := 1e-5
				flatP := params.flatParams()
				for pi := range flatP {
					orig := *flatP[pi]
					*flatP[pi] = orig + eps
					predPlus := m.forwardF64(windows[start+s], params)
					*flatP[pi] = orig - eps
					predMinus := m.forwardF64(windows[start+s], params)
					*flatP[pi] = orig

					grad := 0.0
					for j := 0; j < outDim; j++ {
						diff := pred[j] - sampleLabels[j]
						dLossDpred := 2.0 * diff / float64(bs*outDim)
						grad += dLossDpred * (predPlus[j] - predMinus[j]) / (2 * eps)
					}
					grads[pi] += grad
				}
			}

			batchLoss /= float64(bs * outDim)
			epochLoss += batchLoss
			nBatches++

			if config.GradClip > 0 {
				norm := 0.0
				for _, g := range grads {
					norm += g * g
				}
				norm = math.Sqrt(norm)
				if norm > config.GradClip {
					scale := config.GradClip / norm
					for i := range grads {
						grads[i] *= scale
					}
				}
			}

			lr := warmupLR(config.LR, epoch, config.WarmupEpochs)
			t := float64(epoch*((nSamples+batchSize-1)/batchSize) + nBatches)
			flatP := params.flatParams()
			for i := range flatP {
				adamM[i] = config.Beta1*adamM[i] + (1-config.Beta1)*grads[i]
				adamV[i] = config.Beta2*adamV[i] + (1-config.Beta2)*grads[i]*grads[i]
				mHat := adamM[i] / (1 - math.Pow(config.Beta1, t))
				vHat := adamV[i] / (1 - math.Pow(config.Beta2, t))
				*flatP[i] = *flatP[i] - lr*(mHat/(math.Sqrt(vHat)+config.Epsilon)+config.WeightDecay*(*flatP[i]))
			}
		}

		result.LossHistory[epoch] = epochLoss / float64(nBatches)
		result.FinalLoss = result.LossHistory[epoch]

		if !isFinite(result.FinalLoss) {
			return nil, fmt.Errorf("patchtst: training diverged at epoch %d: loss=%v", epoch, result.FinalLoss)
		}
	}

	m.writeBackF32(params)

	result.Metrics = map[string]float64{"mse": result.FinalLoss}
	return result, nil
}
