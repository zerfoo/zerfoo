package core

import (
	"context"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/graph"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
)

// LMHead maps hidden states to logits.
type LMHead[T tensor.Numeric] struct {
	dense *Dense[T]
}

// NewLMHead creates a new LMHead.
func NewLMHead[T tensor.Numeric](name string, engine compute.Engine[T], ops numeric.Arithmetic[T], modelDim, vocabSize int) (*LMHead[T], error) {
	dense, err := NewDense[T](name, engine, ops, modelDim, vocabSize)
	if err != nil {
		return nil, err
	}
	return &LMHead[T]{dense: dense}, nil
}

// OutputShape returns the output shape of the LMHead.
func (h *LMHead[T]) OutputShape() []int {
	return h.dense.OutputShape()
}

// Forward performs the forward pass of the LMHead.
func (h *LMHead[T]) Forward(ctx context.Context, inputs ...*tensor.Tensor[T]) (*tensor.Tensor[T], error) {
	return h.dense.Forward(ctx, inputs...)
}

// Backward computes the gradients for the LMHead.
func (h *LMHead[T]) Backward(ctx context.Context, outputGradient *tensor.Tensor[T], inputs ...*tensor.Tensor[T]) ([]*tensor.Tensor[T], error) {
	return h.dense.Backward(ctx, outputGradient, inputs...)
}

// Parameters returns the parameters of the LMHead.
func (h *LMHead[T]) Parameters() []*graph.Parameter[T] {
	return h.dense.Parameters()
}
