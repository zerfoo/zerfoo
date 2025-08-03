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
	lastInput   *tensor.Tensor[T]
	outputShape []int
	alpha       float64
}

func NewLeakyReLU[T tensor.Numeric](engine compute.Engine[T], ops numeric.Arithmetic[T], alpha float64) *LeakyReLU[T] {
	return &LeakyReLU[T]{engine: engine, ops: ops, alpha: alpha}
}

func (l *LeakyReLU[T]) OutputShape() []int {
	return l.outputShape
}

func (l *LeakyReLU[T]) Forward(ctx context.Context, inputs ...*tensor.Tensor[T]) (*tensor.Tensor[T], error) {
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

func (l *LeakyReLU[T]) Backward(ctx context.Context, outputGradient *tensor.Tensor[T]) ([]*tensor.Tensor[T], error) {
	dleakyrelu, err := l.engine.UnaryOp(ctx, l.lastInput, func(val T) T { return l.ops.LeakyReLUGrad(val, l.alpha) })
	if err != nil {
		return nil, err
	}
	inputGrad, err := l.engine.Mul(ctx, outputGradient, dleakyrelu)
	if err != nil {
		return nil, err
	}
	return []*tensor.Tensor[T]{inputGrad}, nil
}
