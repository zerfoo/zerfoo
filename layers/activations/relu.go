package activations

import (
	"context"
	"fmt"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/graph"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
)

// ReLU implements the Rectified Linear Unit activation function.
type ReLU[T tensor.Numeric] struct {
	graph.NoParameters[T]
	engine      compute.Engine[T]
	ops         numeric.Arithmetic[T]
	lastInput   *tensor.Tensor[T]
	outputShape []int
}

func NewReLU[T tensor.Numeric](engine compute.Engine[T], ops numeric.Arithmetic[T]) *ReLU[T] {
	return &ReLU[T]{engine: engine, ops: ops}
}

func (r *ReLU[T]) OutputShape() []int {
	return r.outputShape
}

func (r *ReLU[T]) Forward(ctx context.Context, inputs ...*tensor.Tensor[T]) (*tensor.Tensor[T], error) {
	if len(inputs) != 1 {
		return nil, fmt.Errorf("ReLU: %w, expected %d, got %d", graph.ErrInvalidInputCount, 1, len(inputs))
	}
	r.lastInput = inputs[0]
	r.outputShape = r.lastInput.Shape()
	output, err := r.engine.UnaryOp(ctx, r.lastInput, r.ops.ReLU)
	if err != nil {
		return nil, err
	}
	return output, nil
}

func (r *ReLU[T]) Backward(ctx context.Context, outputGradient *tensor.Tensor[T]) ([]*tensor.Tensor[T], error) {
	drelu, err := r.engine.UnaryOp(ctx, r.lastInput, r.ops.ReLUGrad)
	if err != nil {
		return nil, err
	}
	inputGrad, err := r.engine.Mul(ctx, outputGradient, drelu)
	if err != nil {
		return nil, err
	}
	return []*tensor.Tensor[T]{inputGrad}, nil
}
