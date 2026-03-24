package timeseries

import (
	"fmt"
	"math"
)

// trainWindowedEngine runs the GPU/engine-accelerated training path.
// It uses analytical backpropagation via float64 parameters, then writes
// updated weights back to the float32 engine tensors each epoch.
func (m *PatchTST) trainWindowedEngine(windows [][][]float64, labels []float64, config TrainConfig) (*TrainResult, error) {
	nSamples := len(windows)
	outDim := m.config.OutputDim

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
				pred, cache := m.forwardF64WithCache(windows[start+s], params)
				sampleLabels := labels[(start+s)*outDim : (start+s+1)*outDim]

				// Compute MSE loss and dL/dPred.
				dOutput := make([]float64, outDim)
				for j := 0; j < outDim; j++ {
					diff := pred[j] - sampleLabels[j]
					batchLoss += diff * diff
					dOutput[j] = 2.0 * diff / float64(bs*outDim)
				}

				// Analytical backward pass.
				sampleGrads := m.backwardF64(dOutput, params, cache)
				for pi := range grads {
					grads[pi] += sampleGrads[pi]
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

	// Write optimized float64 params back to float32 tensors.
	m.writeBackF32(params)

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
				pred, cache := m.forwardF64WithCache(windows[start+s], params)
				sampleLabels := labels[(start+s)*outDim : (start+s+1)*outDim]

				// Compute MSE loss and dL/dPred.
				dOutput := make([]float64, outDim)
				for j := 0; j < outDim; j++ {
					diff := pred[j] - sampleLabels[j]
					batchLoss += diff * diff
					dOutput[j] = 2.0 * diff / float64(bs*outDim)
				}

				// Analytical backward pass.
				sampleGrads := m.backwardF64(dOutput, params, cache)
				for pi := range grads {
					grads[pi] += sampleGrads[pi]
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
