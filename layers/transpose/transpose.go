// Package transpose provides the Transpose layer for the Zerfoo ML framework.
package transpose

import (
	"context"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/graph"
	"github.com/zerfoo/zerfoo/tensor"
)

// Transpose represents a transpose operation.
type Transpose[T tensor.Numeric] struct {
	engine      compute.Engine[T]
	perm        []int
	outputShape []int
}

// OpType returns the operation type.
func (t *Transpose[T]) OpType() string {
	return "Transpose"
}

// Attributes returns the attributes.
func (t *Transpose[T]) Attributes() map[string]interface{} {
	return map[string]interface{}{
		"perm": t.perm,
	}
}

// New creates a new Transpose layer.
func New[T tensor.Numeric](engine compute.Engine[T], axes []int) *Transpose[T] {
	return &Transpose[T]{
		engine: engine,
		perm:   axes,
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

	outputShape := make([]int, len(shape))
	for i, axis := range t.perm {
		outputShape[i] = shape[axis]
	}

	t.outputShape = outputShape

	// Transpose the input tensor
	transposed, err := t.engine.Transpose(ctx, inputs[0], t.perm)

	return transposed, err
}

// Backward computes the gradients for the Transpose layer.
func (t *Transpose[T]) Backward(_ context.Context, outputGradient *tensor.TensorNumeric[T], _ ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	// The gradient of the transpose is the transpose of the gradient.
	return []*tensor.TensorNumeric[T]{outputGradient}, nil
}
