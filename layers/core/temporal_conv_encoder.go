package core

import (
	"context"
	"fmt"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/graph"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
	"github.com/zerfoo/zerfoo/types"
)

// TemporalConvEncoder replaces the non-differentiable SpectralFingerprint.
// Two stacked Conv1D layers with ReLU + global average pool + linear projection.
//
// Input:  [batch, numStats, windowSize] (per-era group stats over time)
// Output: [batch, outputDim]
//
// Pipeline:
//
//	conv1 = Conv1D(numStats, hiddenChannels, kernel=3, pad=1) + ReLU
//	conv2 = Conv1D(hiddenChannels, hiddenChannels, kernel=3, pad=1) + ReLU
//	pooled = ReduceMean(conv2_out, axis=2)  -- global average pool
//	output = Linear(hiddenChannels, outputDim)
type TemporalConvEncoder[T tensor.Numeric] struct {
	name           string
	engine         compute.Engine[T]
	ops            numeric.Arithmetic[T]
	conv1          *Conv1D[T]
	conv2          *Conv1D[T]
	linear         *Linear[T]
	hiddenChannels int
	outputDim      int
	// Cached for backward
	lastConv1Out *tensor.TensorNumeric[T]
	lastConv2Out *tensor.TensorNumeric[T]
	lastPooled   *tensor.TensorNumeric[T]
}

// NewTemporalConvEncoder creates a new TemporalConvEncoder.
func NewTemporalConvEncoder[T tensor.Numeric](
	name string,
	engine compute.Engine[T],
	ops numeric.Arithmetic[T],
	numStats, hiddenChannels, outputDim int,
) (*TemporalConvEncoder[T], error) {
	if name == "" {
		return nil, fmt.Errorf("layer name cannot be empty")
	}

	conv1, err := NewConv1D[T](name+"_conv1", engine, ops, numStats, hiddenChannels, 3, Conv1DPadding(1))
	if err != nil {
		return nil, fmt.Errorf("conv1: %w", err)
	}

	conv2, err := NewConv1D[T](name+"_conv2", engine, ops, hiddenChannels, hiddenChannels, 3, Conv1DPadding(1))
	if err != nil {
		return nil, fmt.Errorf("conv2: %w", err)
	}

	linear, err := NewLinear[T](name+"_proj", engine, ops, hiddenChannels, outputDim)
	if err != nil {
		return nil, fmt.Errorf("proj: %w", err)
	}

	return &TemporalConvEncoder[T]{
		name:           name,
		engine:         engine,
		ops:            ops,
		conv1:          conv1,
		conv2:          conv2,
		linear:         linear,
		hiddenChannels: hiddenChannels,
		outputDim:      outputDim,
	}, nil
}

func (t *TemporalConvEncoder[T]) OpType() string { return "TemporalConvEncoder" }

func (t *TemporalConvEncoder[T]) Attributes() map[string]interface{} {
	return map[string]interface{}{
		"hidden_channels": t.hiddenChannels,
		"output_dim":      t.outputDim,
	}
}

func (t *TemporalConvEncoder[T]) OutputShape() []int {
	return []int{-1, t.outputDim}
}

// Forward processes temporal group statistics through conv layers.
func (te *TemporalConvEncoder[T]) Forward(ctx context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if len(inputs) != 1 {
		return nil, fmt.Errorf("TemporalConvEncoder requires 1 input, got %d", len(inputs))
	}
	input := inputs[0]
	if len(input.Shape()) != 3 {
		return nil, fmt.Errorf("TemporalConvEncoder input must be 3D [batch, numStats, window], got %v", input.Shape())
	}

	// Conv1 + ReLU
	conv1Out, err := te.conv1.Forward(ctx, input)
	if err != nil {
		return nil, fmt.Errorf("conv1: %w", err)
	}
	relu1Data := conv1Out.Data()
	for i, v := range relu1Data {
		if te.ops.GreaterThan(te.ops.FromFloat64(0), v) {
			relu1Data[i] = te.ops.FromFloat64(0)
		}
	}
	conv1Out.SetData(relu1Data)
	te.lastConv1Out = conv1Out

	// Conv2 + ReLU
	conv2Out, err := te.conv2.Forward(ctx, conv1Out)
	if err != nil {
		return nil, fmt.Errorf("conv2: %w", err)
	}
	relu2Data := conv2Out.Data()
	for i, v := range relu2Data {
		if te.ops.GreaterThan(te.ops.FromFloat64(0), v) {
			relu2Data[i] = te.ops.FromFloat64(0)
		}
	}
	conv2Out.SetData(relu2Data)
	te.lastConv2Out = conv2Out

	// Global average pool over temporal dimension (axis=2)
	pooled, err := te.engine.ReduceMean(ctx, conv2Out, 2, false)
	if err != nil {
		return nil, fmt.Errorf("pool: %w", err)
	}
	te.lastPooled = pooled

	// Linear projection
	output, err := te.linear.Forward(ctx, pooled)
	if err != nil {
		return nil, fmt.Errorf("proj: %w", err)
	}

	return output, nil
}

// Backward computes gradients through the encoder.
func (te *TemporalConvEncoder[T]) Backward(ctx context.Context, mode types.BackwardMode, outputGradient *tensor.TensorNumeric[T], inputs ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	if len(inputs) != 1 {
		return nil, fmt.Errorf("TemporalConvEncoder requires 1 input for backward, got %d", len(inputs))
	}

	// Backward through linear
	dPooled, err := te.linear.Backward(ctx, mode, outputGradient, te.lastPooled)
	if err != nil {
		return nil, fmt.Errorf("proj backward: %w", err)
	}

	// Backward through global average pool: distribute gradient evenly
	conv2Shape := te.lastConv2Out.Shape()
	batch := conv2Shape[0]
	channels := conv2Shape[1]
	timeLen := conv2Shape[2]
	invLen := te.ops.FromFloat64(1.0 / float64(timeLen))

	dPooledData := dPooled[0].Data()
	dConv2Data := make([]T, batch*channels*timeLen)
	for b := range batch {
		for c := range channels {
			grad := te.ops.Mul(dPooledData[b*channels+c], invLen)
			for t := range timeLen {
				dConv2Data[b*channels*timeLen+c*timeLen+t] = grad
			}
		}
	}
	dConv2, err := tensor.New[T](conv2Shape, dConv2Data)
	if err != nil {
		return nil, err
	}

	// ReLU backward on conv2
	conv2Data := te.lastConv2Out.Data()
	dConv2RData := dConv2.Data()
	for i, v := range conv2Data {
		if !te.ops.GreaterThan(v, te.ops.FromFloat64(0)) {
			dConv2RData[i] = te.ops.FromFloat64(0)
		}
	}
	dConv2.SetData(dConv2RData)

	// Backward through conv2
	dConv1Out, err := te.conv2.Backward(ctx, mode, dConv2, te.lastConv1Out)
	if err != nil {
		return nil, fmt.Errorf("conv2 backward: %w", err)
	}

	// ReLU backward on conv1
	conv1Data := te.lastConv1Out.Data()
	dConv1Data := dConv1Out[0].Data()
	for i, v := range conv1Data {
		if !te.ops.GreaterThan(v, te.ops.FromFloat64(0)) {
			dConv1Data[i] = te.ops.FromFloat64(0)
		}
	}
	dConv1Out[0].SetData(dConv1Data)

	// Backward through conv1
	dInput, err := te.conv1.Backward(ctx, mode, dConv1Out[0], inputs[0])
	if err != nil {
		return nil, fmt.Errorf("conv1 backward: %w", err)
	}

	return dInput, nil
}

// Parameters returns all trainable parameters.
func (te *TemporalConvEncoder[T]) Parameters() []*graph.Parameter[T] {
	var params []*graph.Parameter[T]
	params = append(params, te.conv1.Parameters()...)
	params = append(params, te.conv2.Parameters()...)
	params = append(params, te.linear.Parameters()...)
	return params
}
