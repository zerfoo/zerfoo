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
	lastInput   *tensor.TensorNumeric[T]
	outputShape []int
	opType      string
	forwardOp   func(T) T
	backwardOp  func(T) T
}

// BaseActivationOptions holds configuration options for BaseActivation.
// BaseActivationOptions holds configuration options for BaseActivation.
type BaseActivationOptions[T tensor.Numeric] struct {
	ForwardOp  func(T) T
	BackwardOp func(T) T
}

// BaseActivationOption is a function that applies an option to BaseActivationOptions.
type BaseActivationOption[T tensor.Numeric] func(*BaseActivationOptions[T])

// WithForwardOp sets the forward operation for the BaseActivation.
func WithForwardOp[T tensor.Numeric](op func(T) T) BaseActivationOption[T] {
	return func(o *BaseActivationOptions[T]) {
		o.ForwardOp = op
	}
}

// WithBackwardOp sets the backward operation for the BaseActivation.
func WithBackwardOp[T tensor.Numeric](op func(T) T) BaseActivationOption[T] {
	return func(o *BaseActivationOptions[T]) {
		o.BackwardOp = op
	}
}

// NewBaseActivation creates a new base activation with the given forward and backward operations.
func NewBaseActivation[T tensor.Numeric](engine compute.Engine[T], ops numeric.Arithmetic[T], opType string, opts ...BaseActivationOption[T]) *BaseActivation[T] {
	options := &BaseActivationOptions[T]{}
	for _, opt := range opts {
		opt(options)
	}

	return &BaseActivation[T]{
		engine:     engine,
		ops:        ops,
		opType:     opType,
		forwardOp:  options.ForwardOp,
		backwardOp: options.BackwardOp,
	}
}

// OpType returns the operation type of the activation.
func (b *BaseActivation[T]) OpType() string {
	return b.opType
}

// Attributes returns the attributes of the activation.
func (b *BaseActivation[T]) Attributes() map[string]interface{} {
	return make(map[string]interface{})
}

// OutputShape returns the output shape of the activation.
func (b *BaseActivation[T]) OutputShape() []int {
	return b.outputShape
}

// Forward performs the forward pass of the activation function.
func (b *BaseActivation[T]) Forward(ctx context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
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
func (b *BaseActivation[T]) Backward(ctx context.Context, outputGradient *tensor.TensorNumeric[T], _ ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	derivative, err := b.engine.UnaryOp(ctx, b.lastInput, b.backwardOp)
	if err != nil {
		return nil, err
	}
	inputGrad, err := b.engine.Mul(ctx, outputGradient, derivative)
	if err != nil {
		return nil, err
	}

	return []*tensor.TensorNumeric[T]{inputGrad}, nil
}
