package attention

import (
	"context"
	"fmt"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/graph"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
	"github.com/zerfoo/zerfoo/types"
)

// GlobalAttention wraps GroupedQueryAttention to provide a global attention interface.
type GlobalAttention[T tensor.Numeric] struct {
	gqa            *GroupedQueryAttention[T]
	embedDim       int
	numHeads       int
	numKVHeads     int
}

// OpType returns the operation type.
func (ga *GlobalAttention[T]) OpType() string {
	return "GlobalAttention"
}

// Attributes returns the attributes.
func (ga *GlobalAttention[T]) Attributes() map[string]interface{} {
	return map[string]interface{}{
		"embed_dim":    ga.embedDim,
		"num_heads":    ga.numHeads,
		"num_kv_heads": ga.numKVHeads,
	}
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
		gqa:        gqa,
		embedDim:   modelDim,
		numHeads:   numQueryHeads,
		numKVHeads: numKeyValueHeads,
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

// Backward delegates the backward pass to the wrapped GroupedQueryAttention.
func (ga *GlobalAttention[T]) Backward(ctx context.Context, mode types.BackwardMode, dOut *tensor.TensorNumeric[T], inputs ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	// Delegate backward pass to the wrapped GroupedQueryAttention.
	return ga.gqa.Backward(ctx, mode, dOut, inputs...)
}

// OutputShape returns the output shape of the GlobalAttention layer.
func (ga *GlobalAttention[T]) OutputShape() []int {
	return ga.gqa.OutputShape()
}

// ScaleRope scales the rotary positional embeddings.
func (ga *GlobalAttention[T]) ScaleRope(ctx context.Context, factor float64) error {
	return ga.gqa.ScaleRope(ctx, factor)
}

// BuildGlobalAttention constructs a GlobalAttention node from attributes.
// Required attributes:
// - "embed_dim" (int): embedding dimension
// - "num_heads" (int): number of attention heads
// - "num_kv_heads" (int): number of key-value heads
func BuildGlobalAttention[T tensor.Numeric](
	engine compute.Engine[T],
	ops numeric.Arithmetic[T],
	_ string,
	_ map[string]*graph.Parameter[T],
	attributes map[string]interface{},
) (graph.Node[T], error) {
	embedDim, ok := attributes["embed_dim"].(int)
	if !ok {
		return nil, fmt.Errorf("missing or invalid attribute 'embed_dim' for GlobalAttention")
	}
	
	numHeads, ok := attributes["num_heads"].(int)
	if !ok {
		return nil, fmt.Errorf("missing or invalid attribute 'num_heads' for GlobalAttention")
	}
	
	numKVHeads, ok := attributes["num_kv_heads"].(int)
	if !ok {
		return nil, fmt.Errorf("missing or invalid attribute 'num_kv_heads' for GlobalAttention")
	}

	return NewGlobalAttention[T](engine, ops, embedDim, numHeads, numKVHeads)
}

// Statically assert that the type implements the graph.Node interface.
var _ graph.Node[float32] = (*GlobalAttention[float32])(nil)
