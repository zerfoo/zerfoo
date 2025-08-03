package activations

import (
	"context"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
)

// ReLU implements the Rectified Linear Unit activation function.
type ReLU[T tensor.Numeric] struct {
	*BaseActivation[T]
}

// NewReLU creates a new ReLU activation function.
func NewReLU[T tensor.Numeric](engine compute.Engine[T], ops numeric.Arithmetic[T]) *ReLU[T] {
	return &ReLU[T]{
		BaseActivation: NewBaseActivation(engine, ops, ops.ReLU, ops.ReLUGrad),
	}
}

// OutputShape returns the output shape of the ReLU activation.
func (r *ReLU[T]) OutputShape() []int {
	return r.BaseActivation.OutputShape()
}

// Forward performs the forward pass of the ReLU activation.
func (r *ReLU[T]) Forward(ctx context.Context, inputs ...*tensor.Tensor[T]) (*tensor.Tensor[T], error) {
	return r.BaseActivation.Forward(ctx, inputs...)
}

// Backward performs the backward pass of the ReLU activation.
func (r *ReLU[T]) Backward(ctx context.Context, outputGradient *tensor.Tensor[T]) ([]*tensor.Tensor[T], error) {
	return r.BaseActivation.Backward(ctx, outputGradient)
}
