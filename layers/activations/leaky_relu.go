package activations

import (
	"context"
	"fmt"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/graph"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
)

// LeakyReLU implements the Leaky Rectified Linear Unit activation function.
type LeakyReLU[T tensor.Numeric] struct {
	graph.NoParameters[T]
	engine      compute.Engine[T]
	ops         numeric.Arithmetic[T]
	lastInput   *tensor.TensorNumeric[T]
	outputShape []int
	alpha       float64
}

// LeakyReLUOptions holds configuration options for LeakyReLU.
// LeakyReLUOptions holds configuration options for LeakyReLU.
type LeakyReLUOptions[T tensor.Numeric] struct {
	Alpha float64
}

// LeakyReLUOption is a function that applies an option to LeakyReLUOptions.
type LeakyReLUOption[T tensor.Numeric] func(*LeakyReLUOptions[T])

// WithAlpha sets the alpha parameter for LeakyReLU.
func WithAlpha[T tensor.Numeric](alpha float64) LeakyReLUOption[T] {
	return func(o *LeakyReLUOptions[T]) {
		o.Alpha = alpha
	}
}

// NewLeakyReLU creates a new LeakyReLU activation function.
func NewLeakyReLU[T tensor.Numeric](engine compute.Engine[T], ops numeric.Arithmetic[T], opts ...LeakyReLUOption[T]) *LeakyReLU[T] {
	options := &LeakyReLUOptions[T]{
		Alpha: 0.01, // Default alpha value
	}
	for _, opt := range opts {
		opt(options)
	}

	return &LeakyReLU[T]{engine: engine, ops: ops, alpha: options.Alpha}
}

// OutputShape returns the output shape of the LeakyReLU layer.
func (l *LeakyReLU[T]) OutputShape() []int {
	return l.outputShape
}

// Forward computes the LeakyReLU activation for the given input.
func (l *LeakyReLU[T]) Forward(ctx context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if len(inputs) != 1 {
		return nil, fmt.Errorf("LeakyReLU: %w, expected %d, got %d", graph.ErrInvalidInputCount, 1, len(inputs))
	}
	l.lastInput = inputs[0]
	l.outputShape = l.lastInput.Shape()
	output, err := l.engine.UnaryOp(ctx, l.lastInput, func(val T) T { return l.ops.LeakyReLU(val, l.alpha) })
	if err != nil {
		return nil, err
	}

	return output, nil
}

// Backward computes the gradients for the LeakyReLU activation.
func (l *LeakyReLU[T]) Backward(ctx context.Context, outputGradient *tensor.TensorNumeric[T], _ ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	dleakyrelu, err := l.engine.UnaryOp(ctx, l.lastInput, func(val T) T { return l.ops.LeakyReLUGrad(val, l.alpha) })
	if err != nil {
		return nil, err
	}
	inputGrad, err := l.engine.Mul(ctx, outputGradient, dleakyrelu)
	if err != nil {
		return nil, err
	}

	return []*tensor.TensorNumeric[T]{inputGrad}, nil
}

// OpType returns the operation type of the LeakyReLU layer.
func (l *LeakyReLU[T]) OpType() string {
	return "LeakyReLU"
}

// Attributes returns the attributes of the LeakyReLU layer.
func (l *LeakyReLU[T]) Attributes() map[string]interface{} {
	return map[string]interface{}{"alpha": l.alpha}
}
