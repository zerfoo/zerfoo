package activations

import (
	"context"
	"fmt"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/ztensor/types"
)

// BaseActivation provides common functionality for unary activation functions.
//
// Backward recomputes the derivative from the live `inputs ...` it receives
// (ztensor ADR 006 recompute pattern) instead of a struct-field input cache,
// so an arena buffer reuse between Forward and Backward cannot corrupt it.
type BaseActivation[T tensor.Numeric] struct {
	graph.NoParameters[T]
	engine      compute.Engine[T]
	ops         numeric.Arithmetic[T]
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

	b.outputShape = inputs[0].Shape()

	output, err := b.engine.UnaryOp(ctx, inputs[0], b.forwardOp)
	if err != nil {
		return nil, err
	}

	return output, nil
}

// Backward performs the backward pass of the activation function. The
// derivative is recomputed from the live input the graph passes in, not
// from a cached forward tensor (ztensor ADR 006).
func (b *BaseActivation[T]) Backward(ctx context.Context, mode types.BackwardMode, outputGradient *tensor.TensorNumeric[T], inputs ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	if len(inputs) != 1 {
		return nil, fmt.Errorf("BaseActivation: %w, expected %d, got %d", graph.ErrInvalidInputCount, 1, len(inputs))
	}

	derivative, err := b.engine.UnaryOp(ctx, inputs[0], b.backwardOp)
	if err != nil {
		return nil, err
	}

	inputGrad, err := b.engine.Mul(ctx, outputGradient, derivative)
	if err != nil {
		return nil, err
	}

	return []*tensor.TensorNumeric[T]{inputGrad}, nil
}

// Statically assert that the type implements the graph.Node interface.
var _ graph.Node[float32] = (*BaseActivation[float32])(nil)
