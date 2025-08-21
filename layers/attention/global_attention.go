package attention

import (
	"context"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/graph"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
)

// GlobalAttention implements a standard multi-head self-attention mechanism.
type GlobalAttention[T tensor.Numeric] struct {
	gqa *GroupedQueryAttention[T]
}

// NewGlobalAttention creates a new GlobalAttention layer.
func NewGlobalAttention[T tensor.Numeric](
	engine compute.Engine[T],
	ops numeric.Arithmetic[T],
	modelDim, numQueryHeads, numKeyValueHeads int,
	base float64,
	maxSeqLen int,
) (*GlobalAttention[T], error) {
	gqa, err := NewGroupedQueryAttention[T](engine, ops, modelDim, numQueryHeads, numKeyValueHeads, base, maxSeqLen)
	if err != nil {
		return nil, err
	}
	return &GlobalAttention[T]{
		gqa: gqa,
	}, nil
}

// NewGlobalAttentionFromParams creates a new GlobalAttention layer from an existing GroupedQueryAttention layer.
func NewGlobalAttentionFromParams[T tensor.Numeric](gqa *GroupedQueryAttention[T]) *GlobalAttention[T] {
	return &GlobalAttention[T]{
		gqa: gqa,
	}
}

// Parameters returns the parameters of the GlobalAttention layer.
func (ga *GlobalAttention[T]) Parameters() []*graph.Parameter[T] {
	return ga.gqa.Parameters()
}

// Forward computes the forward pass of the GlobalAttention layer.
func (ga *GlobalAttention[T]) Forward(ctx context.Context, inputs ...*tensor.Tensor[T]) (*tensor.Tensor[T], error) {
	return ga.gqa.Forward(ctx, inputs...)
}

// Backward is not implemented
func (ga *GlobalAttention[T]) Backward(ctx context.Context, dOut *tensor.Tensor[T], inputs ...*tensor.Tensor[T]) ([]*tensor.Tensor[T], error) {
	return ga.gqa.Backward(ctx, dOut, inputs...)
}

func (ga *GlobalAttention[T]) OutputShape() []int {
	return ga.gqa.OutputShape()
}

// ScaleRope scales the rotary positional embeddings.
func (ga *GlobalAttention[T]) ScaleRope(ctx context.Context, factor float64) error {
	return ga.gqa.ScaleRope(ctx, factor)
}