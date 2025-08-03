package activations

import (
	"context"
	"fmt"
	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/graph"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
)

// Tanh implements the hyperbolic tangent activation function.
type Tanh[T tensor.Numeric] struct {
	graph.NoParameters[T]
	engine      compute.Engine[T]
	ops         numeric.Arithmetic[T]
	lastInput   *tensor.Tensor[T]
	outputShape []int
}

func NewTanh[T tensor.Numeric](engine compute.Engine[T], ops numeric.Arithmetic[T]) *Tanh[T] {
	return &Tanh[T]{engine: engine, ops: ops}
}

func (t *Tanh[T]) OutputShape() []int {
	return t.outputShape
}

func (t *Tanh[T]) Forward(ctx context.Context, inputs ...*tensor.Tensor[T]) (*tensor.Tensor[T], error) {
	if len(inputs) != 1 {
		return nil, fmt.Errorf("Tanh: %w, expected %d, got %d", graph.ErrInvalidInputCount, 1, len(inputs))
	}
	t.lastInput = inputs[0]
	t.outputShape = t.lastInput.Shape()
	output, err := t.engine.UnaryOp(ctx, t.lastInput, t.ops.Tanh)
	if err != nil {
		return nil, err
	}
	return output, nil
}

// Backward computes the gradient of the Tanh activation.
func (t *Tanh[T]) Backward(ctx context.Context, outputGradient *tensor.Tensor[T]) ([]*tensor.Tensor[T], error) {
	dtanh, err := t.engine.UnaryOp(ctx, t.lastInput, t.ops.TanhGrad)
	if err != nil {
		return nil, err
	}
	inputGrad, err := t.engine.Mul(ctx, outputGradient, dtanh)
	if err != nil {
		return nil, err
	}
	return []*tensor.Tensor[T]{inputGrad}, nil
}
