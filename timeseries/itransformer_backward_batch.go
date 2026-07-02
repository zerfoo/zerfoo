package timeseries

import (
	"context"
	"fmt"

	"github.com/zerfoo/ztensor/tensor"
)

// backwardBatchEngine computes analytical gradients for the iTransformer over a
// full batch. Input and target are [batch, channels, inputLen] and
// [batch, channels, outputLen] tensors respectively. Loss is MSE mean-reduced
// across batch, channels, and outputLen. Returns accumulated gradients in
// flatParams order.
func (m *ITransformer) backwardBatchEngine(
	ctx context.Context,
	input *tensor.TensorNumeric[float32],
	target *tensor.TensorNumeric[float32],
) ([]float64, float64, error) {
	inShape := input.Shape()
	if len(inShape) != 3 {
		return nil, 0, fmt.Errorf("itransformer: backwardBatchEngine expects 3D input [batch, channels, inputLen], got shape %v", inShape)
	}
	batch := inShape[0]
	channels := inShape[1]
	inputLen := inShape[2]
	if channels != m.config.Channels {
		return nil, 0, fmt.Errorf("itransformer: expected %d channels, got %d", m.config.Channels, channels)
	}
	if inputLen != m.config.InputLen {
		return nil, 0, fmt.Errorf("itransformer: expected inputLen %d, got %d", m.config.InputLen, inputLen)
	}

	tgtShape := target.Shape()
	if len(tgtShape) != 3 {
		return nil, 0, fmt.Errorf("itransformer: backwardBatchEngine expects 3D target [batch, channels, outputLen], got shape %v", tgtShape)
	}
	outputLen := m.config.OutputLen
	if tgtShape[0] != batch || tgtShape[1] != channels || tgtShape[2] != outputLen {
		return nil, 0, fmt.Errorf("itransformer: target shape %v doesn't match [%d, %d, %d]", tgtShape, batch, channels, outputLen)
	}

	inData := input.Data()
	tgtData := target.Data()
	scale := 1.0 / float64(batch*channels*outputLen)

	accGrads := newITransformerGrads(m.config)
	totalLoss := 0.0

	for b := 0; b < batch; b++ {
		// Extract per-sample input: [channels][inputLen].
		sampleInput := make([][]float64, channels)
		for c := 0; c < channels; c++ {
			sampleInput[c] = make([]float64, inputLen)
			off := b*channels*inputLen + c*inputLen
			for i := 0; i < inputLen; i++ {
				sampleInput[c][i] = float64(inData[off+i])
			}
		}

		// Forward with cache using engine-accelerated linear projections.
		pred, cache := m.forwardWithCacheEngine(ctx, sampleInput)

		// Compute MSE loss gradient for this sample.
		dOutput := make([][]float64, channels)
		for c := 0; c < channels; c++ {
			dOutput[c] = make([]float64, outputLen)
			for o := 0; o < outputLen; o++ {
				tgtIdx := b*channels*outputLen + c*outputLen + o
				diff := pred[c][o] - float64(tgtData[tgtIdx])
				totalLoss += diff * diff
				dOutput[c][o] = 2.0 * diff * scale
			}
		}

		// Accumulate gradients via analytical backward pass.
		m.backward(dOutput, cache, &accGrads)
	}

	grads := accGrads.collectGrads(m.config)
	loss := totalLoss * scale
	return grads, loss, nil
}
