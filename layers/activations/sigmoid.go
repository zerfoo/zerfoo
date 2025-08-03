package activations

import (
	"context"
	"fmt"
	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/graph"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
)

// Sigmoid implements the sigmoid activation function.
type Sigmoid[T tensor.Numeric] struct {
	graph.NoParameters[T]
	engine      compute.Engine[T]
	ops         numeric.Arithmetic[T]
	lastInput   *tensor.Tensor[T]
	outputShape []int
}

func NewSigmoid[T tensor.Numeric](engine compute.Engine[T], ops numeric.Arithmetic[T]) *Sigmoid[T] {
	return &Sigmoid[T]{engine: engine, ops: ops}
}

func (s *Sigmoid[T]) OutputShape() []int {
	return s.outputShape
}

func (s *Sigmoid[T]) Forward(ctx context.Context, inputs ...*tensor.Tensor[T]) (*tensor.Tensor[T], error) {
	if len(inputs) != 1 {
		return nil, fmt.Errorf("Sigmoid: %w, expected %d, got %d", graph.ErrInvalidInputCount, 1, len(inputs))
	}
	s.lastInput = inputs[0]
	s.outputShape = s.lastInput.Shape()
	output, err := s.engine.UnaryOp(ctx, s.lastInput, s.ops.Sigmoid)
	if err != nil {
		return nil, err
	}
	return output, nil
}

func (s *Sigmoid[T]) Backward(ctx context.Context, outputGradient *tensor.Tensor[T]) ([]*tensor.Tensor[T], error) {
	dsigmoid, err := s.engine.UnaryOp(ctx, s.lastInput, s.ops.SigmoidGrad)
	if err != nil {
		return nil, err
	}
	inputGrad, err := s.engine.Mul(ctx, outputGradient, dsigmoid)
	if err != nil {
		return nil, err
	}
	return []*tensor.Tensor[T]{inputGrad}, nil
}
