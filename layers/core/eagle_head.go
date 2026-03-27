package core

import (
	"context"
	"fmt"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/zerfoo/layers/activations"
	"github.com/zerfoo/zerfoo/layers/normalization"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
)

// EAGLEHead is a lightweight FFN that predicts the next hidden state from
// the penultimate transformer layer's hidden state. It is used in EAGLE-style
// self-speculative decoding to generate draft tokens without a separate model.
//
// Architecture: LayerNorm -> Linear(hidden, hidden) -> SiLU -> Linear(hidden, hidden)
type EAGLEHead[T tensor.Numeric] struct {
	engine  compute.Engine[T]
	ops     numeric.Arithmetic[T]
	norm    *normalization.LayerNormalization[T]
	fc1     *Linear[T]
	sigmoid *activations.Sigmoid[T]
	fc2     *Linear[T]
}

// NewEAGLEHead creates a new EAGLEHead layer with the given hidden dimension.
// The head maintains the same dimensionality throughout: hidden -> hidden.
func NewEAGLEHead[T tensor.Numeric](
	engine compute.Engine[T],
	ops numeric.Arithmetic[T],
	hiddenDim int,
) (*EAGLEHead[T], error) {
	norm, err := normalization.NewLayerNormalization[T](engine, hiddenDim)
	if err != nil {
		return nil, fmt.Errorf("EAGLEHead: create norm: %w", err)
	}

	fc1, err := NewLinear[T]("eagle_head_fc1", engine, ops, hiddenDim, hiddenDim)
	if err != nil {
		return nil, fmt.Errorf("EAGLEHead: create fc1: %w", err)
	}

	fc2, err := NewLinear[T]("eagle_head_fc2", engine, ops, hiddenDim, hiddenDim)
	if err != nil {
		return nil, fmt.Errorf("EAGLEHead: create fc2: %w", err)
	}

	return &EAGLEHead[T]{
		engine:  engine,
		ops:     ops,
		norm:    norm,
		fc1:     fc1,
		sigmoid: activations.NewSigmoid[T](engine, ops),
		fc2:     fc2,
	}, nil
}

// OpType returns the operation type.
func (h *EAGLEHead[T]) OpType() string { return "EAGLEHead" }

// Attributes returns the attributes of the layer.
func (h *EAGLEHead[T]) Attributes() map[string]interface{} {
	return map[string]interface{}{}
}

// OutputShape returns the output shape.
func (h *EAGLEHead[T]) OutputShape() []int {
	return h.fc2.OutputShape()
}

// Forward computes the EAGLE head prediction.
// Input shape: [batch, seq, hidden] -> Output shape: [batch, seq, hidden]
//
// The computation is: LayerNorm -> Linear -> SiLU -> Linear
// where SiLU(x) = x * sigmoid(x).
func (h *EAGLEHead[T]) Forward(ctx context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if len(inputs) != 1 {
		return nil, fmt.Errorf("EAGLEHead requires exactly one input, got %d", len(inputs))
	}

	input := inputs[0]
	inputShape := input.Shape()
	if len(inputShape) != 3 {
		return nil, fmt.Errorf("EAGLEHead expects 3D input [batch, seq, hidden], got %dD", len(inputShape))
	}

	batchSize := inputShape[0]
	seqLen := inputShape[1]
	hiddenDim := inputShape[2]

	// Reshape to 2D for layer operations: [batch*seq, hidden]
	flat, err := h.engine.Reshape(ctx, input, []int{batchSize * seqLen, hiddenDim})
	if err != nil {
		return nil, fmt.Errorf("EAGLEHead: reshape to 2D: %w", err)
	}

	// LayerNorm
	normed, err := h.norm.Forward(ctx, flat)
	if err != nil {
		return nil, fmt.Errorf("EAGLEHead: norm: %w", err)
	}

	// fc1: Linear(hidden, hidden)
	fc1Out, err := h.fc1.Forward(ctx, normed)
	if err != nil {
		return nil, fmt.Errorf("EAGLEHead: fc1: %w", err)
	}

	// SiLU(x) = x * sigmoid(x)
	gate, err := h.sigmoid.Forward(ctx, fc1Out)
	if err != nil {
		return nil, fmt.Errorf("EAGLEHead: sigmoid: %w", err)
	}
	siluOut, err := h.engine.Mul(ctx, fc1Out, gate)
	if err != nil {
		return nil, fmt.Errorf("EAGLEHead: silu mul: %w", err)
	}

	// fc2: Linear(hidden, hidden)
	fc2Out, err := h.fc2.Forward(ctx, siluOut)
	if err != nil {
		return nil, fmt.Errorf("EAGLEHead: fc2: %w", err)
	}

	// Reshape back to 3D: [batch, seq, hidden]
	output, err := h.engine.Reshape(ctx, fc2Out, []int{batchSize, seqLen, hiddenDim})
	if err != nil {
		return nil, fmt.Errorf("EAGLEHead: reshape to 3D: %w", err)
	}

	return output, nil
}

// Parameters returns all trainable parameters of the EAGLE head.
func (h *EAGLEHead[T]) Parameters() []*graph.Parameter[T] {
	params := h.norm.Parameters()
	params = append(params, h.fc1.Parameters()...)
	params = append(params, h.fc2.Parameters()...)
	return params
}
