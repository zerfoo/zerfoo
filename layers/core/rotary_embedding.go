package core

import (
	"context"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/graph"
	"github.com/zerfoo/zerfoo/tensor"
)

// RotaryEmbedding applies rotary position embedding to input tensors.
type RotaryEmbedding[T tensor.Numeric] struct {
	engine      compute.Engine[T]
	name        string
	outputShape []int
}

// NewRotaryEmbedding creates a new RotaryEmbedding layer.
func NewRotaryEmbedding[T tensor.Numeric](engine compute.Engine[T]) *RotaryEmbedding[T] {
	return &RotaryEmbedding[T]{
		engine: engine,
		name:   "RotaryEmbedding",
	}
}

// Name returns the name of the layer.
func (r *RotaryEmbedding[T]) Name() string {
	return r.name
}

// SetName sets the name of the layer.
func (r *RotaryEmbedding[T]) SetName(name string) {
	r.name = name
}

// Parameters returns the parameters of the layer (none for RotaryEmbedding).
func (r *RotaryEmbedding[T]) Parameters() []*graph.Parameter[T] {
	return nil
}

// OutputShape returns the output shape of the layer.
func (r *RotaryEmbedding[T]) OutputShape() []int {
	return r.outputShape
}

// Forward applies rotary embedding to the input.
// For now, this is a simplified implementation that returns the input unchanged.
func (r *RotaryEmbedding[T]) Forward(_ context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if len(inputs) < 1 {
		panic("RotaryEmbedding layer requires at least 1 input")
	}

	input := inputs[0]
	r.outputShape = input.Shape()

	// Simplified implementation: return the first input unchanged
	// In a full implementation, this would apply rotary position embeddings
	return input, nil
}

// Backward computes the gradients for the RotaryEmbedding layer.
func (r *RotaryEmbedding[T]) Backward(_ context.Context, outputGradient *tensor.TensorNumeric[T], inputs ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	if len(inputs) < 1 {
		panic("RotaryEmbedding layer requires at least 1 input")
	}

	// For simplified implementation, gradient passes through unchanged
	gradients := make([]*tensor.TensorNumeric[T], len(inputs))
	gradients[0] = outputGradient

	// Set remaining gradients to nil (no gradient flow)
	for i := 1; i < len(inputs); i++ {
		gradients[i] = nil
	}

	return gradients, nil
}

// OpType returns the operation type of the RotaryEmbedding layer.
func (r *RotaryEmbedding[T]) OpType() string {
	return r.name
}

// Attributes returns nil for the RotaryEmbedding layer.
func (r *RotaryEmbedding[T]) Attributes() map[string]interface{} {
	return nil
}
