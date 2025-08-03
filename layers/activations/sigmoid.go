package activations

import (
	"context"
	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
)

// Sigmoid implements the sigmoid activation function.
type Sigmoid[T tensor.Numeric] struct {
	*BaseActivation[T]
}

// NewSigmoid creates a new Sigmoid activation function.
func NewSigmoid[T tensor.Numeric](engine compute.Engine[T], ops numeric.Arithmetic[T]) *Sigmoid[T] {
	return &Sigmoid[T]{
		BaseActivation: NewBaseActivation(engine, ops, ops.Sigmoid, ops.SigmoidGrad),
	}
}

// OutputShape returns the output shape of the Sigmoid activation.
func (s *Sigmoid[T]) OutputShape() []int {
	return s.BaseActivation.OutputShape()
}

// Forward performs the forward pass of the Sigmoid activation.
func (s *Sigmoid[T]) Forward(ctx context.Context, inputs ...*tensor.Tensor[T]) (*tensor.Tensor[T], error) {
	return s.BaseActivation.Forward(ctx, inputs...)
}

// Backward performs the backward pass of the Sigmoid activation.
func (s *Sigmoid[T]) Backward(ctx context.Context, outputGradient *tensor.Tensor[T]) ([]*tensor.Tensor[T], error) {
	return s.BaseActivation.Backward(ctx, outputGradient)
}
