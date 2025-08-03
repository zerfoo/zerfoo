package activations

import (
	"context"
	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
)

// Tanh implements the hyperbolic tangent activation function.
type Tanh[T tensor.Numeric] struct {
	*BaseActivation[T]
}

// NewTanh creates a new Tanh activation function.
func NewTanh[T tensor.Numeric](engine compute.Engine[T], ops numeric.Arithmetic[T]) *Tanh[T] {
	return &Tanh[T]{
		BaseActivation: NewBaseActivation(engine, ops, ops.Tanh, ops.TanhGrad),
	}
}

// OutputShape returns the output shape of the Tanh activation.
func (t *Tanh[T]) OutputShape() []int {
	return t.BaseActivation.OutputShape()
}

// Forward performs the forward pass of the Tanh activation.
func (t *Tanh[T]) Forward(ctx context.Context, inputs ...*tensor.Tensor[T]) (*tensor.Tensor[T], error) {
	return t.BaseActivation.Forward(ctx, inputs...)
}

// Backward computes the gradient of the Tanh activation.
func (t *Tanh[T]) Backward(ctx context.Context, outputGradient *tensor.Tensor[T]) ([]*tensor.Tensor[T], error) {
	return t.BaseActivation.Backward(ctx, outputGradient)
}
