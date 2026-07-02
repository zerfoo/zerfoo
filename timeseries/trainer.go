package timeseries

import (
	"fmt"

	"github.com/zerfoo/zerfoo/training/optimizer"
	"github.com/zerfoo/zerfoo/training/scheduler"
)

// TrainableBackend is the interface that windowed time-series models implement
// to participate in the shared TrainLoop. Each model provides a per-sample
// forward/backward pass and exposes its parameters as a flat slice.
type TrainableBackend interface {
	// ForwardSample runs the model on a single input window [channels][inputLen]
	// and returns (flatOutput, cache, error). flatOutput is [channels*outputLen].
	// cache holds activations needed by BackwardSample.
	ForwardSample(input [][]float64) ([]float64, interface{}, error)

	// BackwardSample computes parameter gradients for a single sample.
	// dOutput is the loss gradient w.r.t. the flat output [channels*outputLen].
	// cache is the value returned by the corresponding ForwardSample call.
	// It accumulates gradients into the internal gradient buffer and returns
	// the gradient norm contribution (unused, reserved for future use).
	BackwardSample(dOutput []float64, cache interface{}) error

	// FlatGrads returns a pointer to the internal gradient accumulator.
	// The slice has length ParamCount(). The caller is responsible for
	// zeroing it before each batch via ZeroGrads.
	FlatGrads() []float64

	// ZeroGrads resets all accumulated gradients to zero.
	ZeroGrads()

	// FlatParams returns pointers to all trainable parameters.
	FlatParams() []*float64

	// ParamCount returns the total number of trainable parameters.
	ParamCount() int
}

// TrainLoop implements the shared AdamW training loop for any TrainableBackend.
// It handles batching, gradient clipping, LR warmup, and loss tracking.
func TrainLoop(backend TrainableBackend, windows [][][]float64, labels []float64, config TrainConfig) (*TrainResult, error) {
	nSamples := len(windows)
	if nSamples == 0 {
		return nil, fmt.Errorf("trainloop: empty training set")
	}

	nParams := backend.ParamCount()
	outDim := len(labels) / nSamples
	if outDim*nSamples != len(labels) {
		return nil, fmt.Errorf("trainloop: labels length %d not divisible by nSamples %d", len(labels), nSamples)
	}

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

	adamState := optimizer.NewAdamWStateF64(nParams)

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

			backend.ZeroGrads()
			batchLoss := 0.0

			for s := 0; s < bs; s++ {
				sampleIdx := start + s
				output, cache, err := backend.ForwardSample(windows[sampleIdx])
				if err != nil {
					return nil, fmt.Errorf("trainloop: forward sample %d: %w", sampleIdx, err)
				}

				// Compute MSE loss and output gradient.
				dOutput := make([]float64, outDim)
				sampleLoss := 0.0
				labelBase := sampleIdx * outDim
				for i := 0; i < outDim; i++ {
					diff := output[i] - labels[labelBase+i]
					sampleLoss += diff * diff
					dOutput[i] = 2.0 * diff / float64(bs*outDim)
				}
				batchLoss += sampleLoss

				if err := backend.BackwardSample(dOutput, cache); err != nil {
					return nil, fmt.Errorf("trainloop: backward sample %d: %w", sampleIdx, err)
				}
			}

			batchLoss /= float64(bs * outDim)
			epochLoss += batchLoss
			nBatches++

			// Gradient clipping.
			grads := backend.FlatGrads()
			optimizer.ClipGradientsF64(grads, config.GradClip)

			// AdamW update with LR warmup.
			lr := scheduler.WarmupLR(config.LR, epoch, config.WarmupEpochs)
			t := float64(epoch*((nSamples+batchSize-1)/batchSize) + nBatches)
			optimizer.AdamWUpdateF64(backend.FlatParams(), grads, adamState, lr, config.Beta1, config.Beta2, config.Epsilon, config.WeightDecay, t)
		}

		result.LossHistory[epoch] = epochLoss / float64(nBatches)
		result.FinalLoss = result.LossHistory[epoch]

		if !isFinite(result.FinalLoss) {
			return nil, fmt.Errorf("trainloop: training diverged at epoch %d: loss=%v", epoch, result.FinalLoss)
		}
	}

	result.Metrics = map[string]float64{"mse": result.FinalLoss}
	return result, nil
}
