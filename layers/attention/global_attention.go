package attention

import (
	"context"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/graph"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
)

// GlobalAttention wraps GroupedQueryAttention to provide a global attention interface.
type GlobalAttention[T tensor.Numeric] struct {
	gqa *GroupedQueryAttention[T]
}

// OpType returns the operation type.
func (ga *GlobalAttention[T]) OpType() string {
	return "GlobalAttention"
}

// Attributes returns the attributes.
func (ga *GlobalAttention[T]) Attributes() map[string]interface{} {
	return ga.gqa.Attributes()
}

// GlobalAttentionOptions holds configuration options for GlobalAttention layer.
type GlobalAttentionOptions struct {
	Base      float64
	MaxSeqLen int
}

// GlobalAttentionOption is a function that configures GlobalAttentionOptions.
type GlobalAttentionOption func(*GlobalAttentionOptions)

// WithGlobalAttentionBase sets the base (theta) parameter for rotary positional embeddings.
func WithGlobalAttentionBase(base float64) GlobalAttentionOption {
	return func(opts *GlobalAttentionOptions) {
		opts.Base = base
	}
}

// WithGlobalAttentionMaxSeqLen sets the maximum sequence length.
func WithGlobalAttentionMaxSeqLen(maxSeqLen int) GlobalAttentionOption {
	return func(opts *GlobalAttentionOptions) {
		opts.MaxSeqLen = maxSeqLen
	}
}

// NewGlobalAttention creates a new GlobalAttention layer.
//
// Parameters:
// - engine: compute engine for tensor operations
// - ops: arithmetic operations for the numeric type
// - modelDim: model dimension
// - numQueryHeads: number of query heads
// - numKeyValueHeads: number of key/value heads
// - options: functional options for configuration
//
// Default values:
// - base: 10000.0
// - maxSeqLen: 2048.
func NewGlobalAttention[T tensor.Numeric](
	engine compute.Engine[T],
	ops numeric.Arithmetic[T],
	modelDim, numQueryHeads, numKeyValueHeads int,
	options ...GlobalAttentionOption,
) (*GlobalAttention[T], error) {
	// Set default options
	opts := &GlobalAttentionOptions{
		Base:      10000.0,
		MaxSeqLen: 2048,
	}

	// Apply functional options
	for _, option := range options {
		option(opts)
	}

	base := opts.Base
	maxSeqLen := opts.MaxSeqLen
	gqa, err := NewGroupedQueryAttention[T](
		engine, ops, modelDim, numQueryHeads, numKeyValueHeads,
		WithRopeBase[T](base),
		WithMaxSeqLen[T](maxSeqLen),
	)
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
func (ga *GlobalAttention[T]) Forward(ctx context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	return ga.gqa.Forward(ctx, inputs...)
}

// Backward is not implemented.
func (ga *GlobalAttention[T]) Backward(ctx context.Context, dOut *tensor.TensorNumeric[T], inputs ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	return ga.gqa.Backward(ctx, dOut, inputs...)
}

func (ga *GlobalAttention[T]) OutputShape() []int {
	return ga.gqa.OutputShape()
}

// ScaleRope scales the rotary positional embeddings.
func (ga *GlobalAttention[T]) ScaleRope(ctx context.Context, factor float64) error {
	return ga.gqa.ScaleRope(ctx, factor)
}
