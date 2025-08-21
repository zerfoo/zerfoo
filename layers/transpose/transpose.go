// Package transpose provides the Transpose layer for the Zerfoo ML framework.
package transpose

import (
	"context"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/graph"
	"github.com/zerfoo/zerfoo/tensor"
)

// Transpose is a layer that transposes a tensor.
type Transpose[T tensor.Numeric] struct {
	engine      compute.Engine[T]
	axes        []int
	outputShape []int
}

// New creates a new Transpose layer.
func New[T tensor.Numeric](engine compute.Engine[T], axes []int) *Transpose[T] {
	return &Transpose[T]{
		engine: engine,
		axes:   axes,
	}
}

// OutputShape returns the output shape of the Transpose layer.
func (t *Transpose[T]) OutputShape() []int {
	return t.outputShape
}

// Parameters returns no trainable parameters for the Transpose layer.
func (t *Transpose[T]) Parameters() []*graph.Parameter[T] {
	return nil
}

// Forward computes the transpose operation.
func (t *Transpose[T]) Forward(ctx context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	input := inputs[0]
	shape := input.Shape()
	
	axes := t.axes
	if axes == nil {
		// Default is to reverse the dimensions
		axes = make([]int, len(shape))
		for i := range shape {
			axes[i] = len(shape) - 1 - i
		}
	}

	outputShape := make([]int, len(shape))
	for i, axis := range axes {
		outputShape[i] = shape[axis]
	}
	t.outputShape = outputShape

	return t.engine.Transpose(ctx, input, axes)
}

// Backward computes the gradients for the Transpose layer.
func (t *Transpose[T]) Backward(ctx context.Context, outputGradient *tensor.TensorNumeric[T], inputs ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	// The gradient of the transpose is the transpose of the gradient.
	return []*tensor.TensorNumeric[T]{outputGradient}, nil
}
