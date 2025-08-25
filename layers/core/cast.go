// Package core provides core layer implementations for the Zerfoo ML framework.
package core

import (
	"context"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/graph"
	"github.com/zerfoo/zerfoo/tensor"
	"github.com/zerfoo/zerfoo/types"
)

// Cast is a layer that converts tensor elements to a different data type.
type Cast[T tensor.Numeric] struct {
	engine      compute.Engine[T]
	outputShape []int
}

// NewCast creates a new Cast layer.
func NewCast[T tensor.Numeric](engine compute.Engine[T]) *Cast[T] {
	return &Cast[T]{
		engine: engine,
	}
}

// OutputShape returns the output shape of the Cast layer.
func (c *Cast[T]) OutputShape() []int {
	return c.outputShape
}

// Parameters returns no trainable parameters for the Cast layer.
func (c *Cast[T]) Parameters() []*graph.Parameter[T] {
	return nil
}

// Forward computes the cast operation (for same type, this is essentially a copy).
func (c *Cast[T]) Forward(_ context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if len(inputs) != 1 {
		panic("Cast layer requires exactly 1 input")
	}

	input := inputs[0]
	c.outputShape = input.Shape()

	// For same-type casting, we just return a copy of the input
	// In a more complete implementation, this would handle type conversions
	return input, nil
}

// Backward computes the gradients for the Cast layer.
func (c *Cast[T]) Backward(_ context.Context, mode types.BackwardMode, outputGradient *tensor.TensorNumeric[T], inputs ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	if len(inputs) != 1 {
		panic("Cast layer requires exactly 1 input")
	}

	// For same-type casting, gradient passes through unchanged
	return []*tensor.TensorNumeric[T]{outputGradient}, nil
}

// OpType returns the operation type of the Cast layer.
func (c *Cast[T]) OpType() string {
	return "Cast"
}

// Attributes returns nil for the Cast layer.
func (c *Cast[T]) Attributes() map[string]interface{} {
	return nil
}

// Statically assert that the type implements the graph.Node interface.
var _ graph.Node[float32] = (*Cast[float32])(nil)
