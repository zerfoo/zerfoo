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

// LeakyReLU implements the Leaky Rectified Linear Unit activation function.
type LeakyReLU[T tensor.Numeric] struct {
	graph.NoParameters[T]
	engine      compute.Engine[T]
	ops         numeric.Arithmetic[T]
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

	l.outputShape = inputs[0].Shape()

	output, err := l.engine.UnaryOp(ctx, inputs[0], func(val T) T { return l.ops.LeakyReLU(val, l.alpha) })
	if err != nil {
		return nil, err
	}

	return output, nil
}

// Backward computes the gradients for the LeakyReLU activation. The
// derivative is recomputed from the live input the graph passes in, not
// from a cached forward tensor (ztensor ADR 006 recompute pattern).
func (l *LeakyReLU[T]) Backward(ctx context.Context, mode types.BackwardMode, outputGradient *tensor.TensorNumeric[T], inputs ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	if len(inputs) != 1 {
		return nil, fmt.Errorf("LeakyReLU: %w, expected %d, got %d", graph.ErrInvalidInputCount, 1, len(inputs))
	}

	dleakyrelu, err := l.engine.UnaryOp(ctx, inputs[0], func(val T) T { return l.ops.LeakyReLUGrad(val, l.alpha) })
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

// Statically assert that the type implements the graph.Node interface.
var _ graph.Node[float32] = (*LeakyReLU[float32])(nil)
