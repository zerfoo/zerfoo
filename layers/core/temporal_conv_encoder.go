package core

import (
	"context"
	"fmt"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/ztensor/types"
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
	// Cached for backward. Conv outputs and the pooled tensor are
	// expensive to recompute, so they are registered with the
	// save-for-backward contract (ztensor ADR 006) every Forward.
	lastConv1Out *tensor.TensorNumeric[T]
	lastConv2Out *tensor.TensorNumeric[T]
	lastPooled   *tensor.TensorNumeric[T]
	saver        graph.Saver[T] // wired by graph Builder (graph.SaverAware); nil outside a Graph
}

// SetSaver implements graph.SaverAware (ztensor ADR 006).
func (t *TemporalConvEncoder[T]) SetSaver(sv graph.Saver[T]) {
	t.saver = sv
}

// Statically assert that the type participates in the save-for-backward contract.
var _ graph.SaverAware[float32] = (*TemporalConvEncoder[float32])(nil)

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
	conv1Out, err = te.engine.UnaryOp(ctx, conv1Out, te.ops.ReLU)
	if err != nil {
		return nil, fmt.Errorf("relu1: %w", err)
	}
	te.lastConv1Out = conv1Out

	// Conv2 + ReLU
	conv2Out, err := te.conv2.Forward(ctx, conv1Out)
	if err != nil {
		return nil, fmt.Errorf("conv2: %w", err)
	}
	conv2Out, err = te.engine.UnaryOp(ctx, conv2Out, te.ops.ReLU)
	if err != nil {
		return nil, fmt.Errorf("relu2: %w", err)
	}
	te.lastConv2Out = conv2Out

	// Global average pool over temporal dimension (axis=2)
	pooled, err := te.engine.ReduceMean(ctx, conv2Out, 2, false)
	if err != nil {
		return nil, fmt.Errorf("pool: %w", err)
	}
	te.lastPooled = pooled
	if te.saver != nil {
		te.saver.SaveForBackward(te.lastConv1Out, te.lastConv2Out, pooled)
	}

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
	// Reshape dPooled from [batch, channels] to [batch, channels, 1] then
	// broadcast to [batch, channels, timeLen] via engine ops.
	conv2Shape := te.lastConv2Out.Shape()
	timeLen := conv2Shape[2]
	invLen := te.ops.FromFloat64(1.0 / float64(timeLen))

	// Scale by 1/timeLen
	dPooledScaled, err := te.engine.MulScalar(ctx, dPooled[0], invLen)
	if err != nil {
		return nil, fmt.Errorf("pool backward scale: %w", err)
	}
	// Reshape to [batch, channels, 1] for broadcasting
	dPooledShape := dPooledScaled.Shape()
	dPooled3D, err := te.engine.Reshape(ctx, dPooledScaled, []int{dPooledShape[0], dPooledShape[1], 1})
	if err != nil {
		return nil, fmt.Errorf("pool backward reshape: %w", err)
	}
	// Broadcast to [batch, channels, timeLen] by repeating along time axis
	dConv2, err := te.engine.Repeat(ctx, dPooled3D, 2, timeLen)
	if err != nil {
		return nil, fmt.Errorf("pool backward repeat: %w", err)
	}

	// ReLU backward on conv2: mask gradient where activation was <= 0
	dConv2Mask, err := te.engine.UnaryOp(ctx, te.lastConv2Out, te.ops.ReLUGrad)
	if err != nil {
		return nil, fmt.Errorf("relu2 backward: %w", err)
	}
	dConv2, err = te.engine.Mul(ctx, dConv2, dConv2Mask)
	if err != nil {
		return nil, fmt.Errorf("relu2 backward mul: %w", err)
	}

	// Backward through conv2
	dConv1Out, err := te.conv2.Backward(ctx, mode, dConv2, te.lastConv1Out)
	if err != nil {
		return nil, fmt.Errorf("conv2 backward: %w", err)
	}

	// ReLU backward on conv1: mask gradient where activation was <= 0
	dConv1Mask, err := te.engine.UnaryOp(ctx, te.lastConv1Out, te.ops.ReLUGrad)
	if err != nil {
		return nil, fmt.Errorf("relu1 backward: %w", err)
	}
	dConv1Masked, err := te.engine.Mul(ctx, dConv1Out[0], dConv1Mask)
	if err != nil {
		return nil, fmt.Errorf("relu1 backward mul: %w", err)
	}

	// Backward through conv1
	dInput, err := te.conv1.Backward(ctx, mode, dConv1Masked, inputs[0])
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
