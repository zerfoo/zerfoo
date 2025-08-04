package activations

import (
	"context"
	"fmt"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/graph"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
)

// BaseActivation provides common functionality for unary activation functions.
type BaseActivation[T tensor.Numeric] struct {
	graph.NoParameters[T]
	engine      compute.Engine[T]
	ops         numeric.Arithmetic[T]
	lastInput   *tensor.Tensor[T]
	outputShape []int
	forwardOp   func(T) T
	backwardOp  func(T) T
}

// NewBaseActivation creates a new base activation with the given forward and backward operations.
func NewBaseActivation[T tensor.Numeric](engine compute.Engine[T], ops numeric.Arithmetic[T], forwardOp, backwardOp func(T) T) *BaseActivation[T] {
	return &BaseActivation[T]{
		engine:     engine,
		ops:        ops,
		forwardOp:  forwardOp,
		backwardOp: backwardOp,
	}
}

// OutputShape returns the output shape of the activation.
func (b *BaseActivation[T]) OutputShape() []int {
	return b.outputShape
}

// Forward performs the forward pass of the activation function.
func (b *BaseActivation[T]) Forward(ctx context.Context, inputs ...*tensor.Tensor[T]) (*tensor.Tensor[T], error) {
	if len(inputs) != 1 {
		return nil, fmt.Errorf("BaseActivation: %w, expected %d, got %d", graph.ErrInvalidInputCount, 1, len(inputs))
	}
	b.lastInput = inputs[0]
	b.outputShape = b.lastInput.Shape()
	output, err := b.engine.UnaryOp(ctx, b.lastInput, b.forwardOp)
	if err != nil {
		return nil, err
	}

	return output, nil
}

// Backward performs the backward pass of the activation function.
func (b *BaseActivation[T]) Backward(ctx context.Context, outputGradient *tensor.Tensor[T]) ([]*tensor.Tensor[T], error) {
	derivative, err := b.engine.UnaryOp(ctx, b.lastInput, b.backwardOp)
	if err != nil {
		return nil, err
	}
	inputGrad, err := b.engine.Mul(ctx, outputGradient, derivative)
	if err != nil {
		return nil, err
	}

	return []*tensor.Tensor[T]{inputGrad}, nil
}
