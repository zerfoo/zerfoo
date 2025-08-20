package attention

import (
	"context"
	"fmt"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/graph"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
)

// GlobalSelfAttention implements a global self-attention mechanism.
type GlobalSelfAttention[T tensor.Numeric] struct {
	gqa         *GroupedQueryAttention[T]
	outputShape []int
}

// NewGlobalSelfAttention creates a new GlobalSelfAttention layer.
func NewGlobalSelfAttention[T tensor.Numeric](
	engine compute.Engine[T],
	ops numeric.Arithmetic[T],
	modelDim, numQueryHeads, numKeyValueHeads int,
	epsilon T,
	base float64,
	maxSeqLen int,
) (*GlobalSelfAttention[T], error) {
	gqa, err := NewGroupedQueryAttention[T](engine, ops, modelDim, numQueryHeads, numKeyValueHeads, base, maxSeqLen)
	if err != nil {
		return nil, fmt.Errorf("failed to create GroupedQueryAttention: %w", err)
	}

	return &GlobalSelfAttention[T]{
		gqa: gqa,
	}, nil
}

// OutputShape returns the output shape of the GlobalSelfAttention.
func (gsa *GlobalSelfAttention[T]) OutputShape() []int {
	return gsa.outputShape
}

// Parameters returns the parameters of the GlobalSelfAttention layer.
func (gsa *GlobalSelfAttention[T]) Parameters() []*graph.Parameter[T] {
	return gsa.gqa.Parameters()
}

// Forward computes the global self-attention.
func (gsa *GlobalSelfAttention[T]) Forward(ctx context.Context, inputs ...*tensor.Tensor[T]) (*tensor.Tensor[T], error) {
	output, err := gsa.gqa.Forward(ctx, inputs...)
	if err != nil {
		return nil, err
	}
	gsa.outputShape = output.Shape()
	return output, nil
}

// Backward computes the gradients for GlobalSelfAttention.
func (gsa *GlobalSelfAttention[T]) Backward(ctx context.Context, dOut *tensor.Tensor[T], inputs ...*tensor.Tensor[T]) ([]*tensor.Tensor[T], error) {
	return gsa.gqa.Backward(ctx, dOut, inputs...)
}

// ScaleRope scales the RoPE embeddings.
func (gsa *GlobalSelfAttention[T]) ScaleRope(ctx context.Context, factor float64) error {
	return gsa.gqa.rope.Scale(ctx, factor)
}
