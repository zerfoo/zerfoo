package timeseries

import (
	"context"
	"fmt"

	"github.com/zerfoo/ztensor/tensor"
)

// forwardBatchEngine runs the N-HiTS forward pass on a batched 3D input using
// the compute engine for all linear operations.
// Input shape: [batch, channels, inputLen]. Output shape: [batch, outputLen].
func (m *NHiTS) forwardBatchEngine(ctx context.Context, input *tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
	shape := input.Shape()
	if len(shape) != 3 {
		return nil, fmt.Errorf("nhits: forwardBatchEngine expects 3D input [batch, channels, inputLen], got shape %v", shape)
	}
	batch := shape[0]
	channels := shape[1]
	inputLen := shape[2]
	if channels != m.config.Channels {
		return nil, fmt.Errorf("nhits: expected %d channels, got %d", m.config.Channels, channels)
	}
	if inputLen != m.config.InputLength {
		return nil, fmt.Errorf("nhits: expected inputLen %d, got %d", m.config.InputLength, inputLen)
	}

	data := input.Data()
	outputLen := m.config.OutputLength

	// Initialize forecast accumulator: [batch, outputLen].
	forecastData := make([]float32, batch*outputLen)

	for _, stack := range m.stacks {
		stackOut, err := m.stackForwardEngine(ctx, data, batch, channels, inputLen, stack)
		if err != nil {
			return nil, err
		}
		for i := range forecastData {
			forecastData[i] += stackOut[i]
		}
	}

	return tensor.New[float32]([]int{batch, outputLen}, forecastData)
}

// stackForwardEngine processes one N-HiTS stack using engine ops:
// per-channel max-pool -> flatten -> MLP (engine MatMul + ReLU) -> output projection.
// Input data layout: [batch, channels, inputLen] flattened.
// Returns [batch * outputLen] slice.
func (m *NHiTS) stackForwardEngine(ctx context.Context, data []float32, batch, channels, inputLen int, stack nhitsStack) ([]float32, error) {
	pLen := pooledLen(inputLen, stack.poolKernel)
	flatDim := pLen * channels

	// Per-channel max-pooling then interleave into flat buffer.
	flatData := make([]float32, batch*flatDim)
	for c := 0; c < channels; c++ {
		// Extract channel c across all batches into contiguous buffer.
		chanData := make([]float32, batch*inputLen)
		for b := 0; b < batch; b++ {
			srcOff := b*channels*inputLen + c*inputLen
			dstOff := b * inputLen
			copy(chanData[dstOff:dstOff+inputLen], data[srcOff:srcOff+inputLen])
		}

		pooled := maxPool1D(chanData, inputLen, stack.poolKernel)

		// Place pooled channel into flat buffer: layout [batch, channels*pLen]
		// with channel c occupying columns [c*pLen : (c+1)*pLen].
		for b := 0; b < batch; b++ {
			srcOff := b * pLen
			dstOff := b*flatDim + c*pLen
			copy(flatData[dstOff:dstOff+pLen], pooled[srcOff:srcOff+pLen])
		}
	}

	// Create tensor [batch, flatDim] and run MLP via engine.
	h, err := tensor.New[float32]([]int{batch, flatDim}, flatData)
	if err != nil {
		return nil, err
	}

	for _, l := range stack.mlpLayers {
		h, err = m.linearForward(ctx, h, l)
		if err != nil {
			return nil, err
		}
		h, err = m.engine.UnaryOp(ctx, h, m.ops.ReLU)
		if err != nil {
			return nil, err
		}
	}

	// Output projection -> [batch, outputLen].
	h, err = m.linearForward(ctx, h, stack.outputProj)
	if err != nil {
		return nil, err
	}

	return h.Data(), nil
}
