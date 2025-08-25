// Package core provides the Shape layer for the Zerfoo ML framework.
package core

import (
	"context"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/graph"
	"github.com/zerfoo/zerfoo/tensor"
	"github.com/zerfoo/zerfoo/types"
)

// Shape is a layer that outputs the shape of its input tensor.
type Shape[T tensor.Numeric] struct {
	engine      compute.Engine[T]
	outputShape []int
}

// New creates a new Shape layer.
func New[T tensor.Numeric](engine compute.Engine[T]) *Shape[T] {
	return &Shape[T]{
		engine: engine,
	}
}

// OutputShape returns the output shape of the Shape layer.
func (s *Shape[T]) OutputShape() []int {
	return s.outputShape
}

// Parameters returns no trainable parameters for the Shape layer.
func (s *Shape[T]) Parameters() []*graph.Parameter[T] {
	return nil
}

// Forward computes the shape of the input tensor.
func (s *Shape[T]) Forward(_ context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	input := inputs[0]
	shape := input.Shape()
	s.outputShape = []int{len(shape)}

	// The output of the Shape layer is a 1D tensor of integers.
	// However, the graph is typed with T. We will need to handle this
	// type mismatch. For now, we will cast the shape to T.
	shapeT := make([]T, len(shape))
	for i, v := range shape {
		shapeT[i] = T(v)
	}

	return tensor.New[T](s.outputShape, shapeT)
}

// Backward computes the gradients for the Shape layer.
func (s *Shape[T]) Backward(_ context.Context, mode types.BackwardMode, _ *tensor.TensorNumeric[T], _ ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	// The Shape layer has no trainable parameters and its output is not a function
	// of the input tensor's values, so the gradient is zero.
	return nil, nil
}

// OpType returns the operation type of the Shape layer.
func (s *Shape[T]) OpType() string {
	return "Shape"
}

// Attributes returns nil for the Shape layer.
func (s *Shape[T]) Attributes() map[string]interface{} {
	return nil
}
