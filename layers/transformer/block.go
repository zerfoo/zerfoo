// Package transformer provides transformer building blocks such as the
// Transformer `Block` used in encoder/decoder stacks.
package transformer

import (
	"context"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/graph"
	"github.com/zerfoo/zerfoo/layers/core"
	"github.com/zerfoo/zerfoo/layers/normalization"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
	"github.com/zerfoo/zerfoo/types"
)

// Block represents a single Transformer block.
type Block[T tensor.Numeric] struct {
	attention         graph.Node[T]
	ffn               *core.FFN[T]
	norm1             *normalization.RMSNorm[T]
	norm2             *normalization.RMSNorm[T]
	normPostAttention *normalization.RMSNorm[T]
}

// BlockOptions holds configuration options for the Transformer block.
type BlockOptions[T tensor.Numeric] struct {
	Epsilon T
}

// BlockOption is a function that applies an option to BlockOptions.
type BlockOption[T tensor.Numeric] func(*BlockOptions[T])

// WithEpsilon sets the epsilon value for the RMS normalization layers.
func WithEpsilon[T tensor.Numeric](epsilon T) BlockOption[T] {
	return func(o *BlockOptions[T]) {
		o.Epsilon = epsilon
	}
}

// NewTransformerBlock creates a new Transformer block.
func NewTransformerBlock[T tensor.Numeric](
	engine compute.Engine[T],
	ops numeric.Arithmetic[T],
	modelDim, ffnDim int,
	attention graph.Node[T],
	opts ...BlockOption[T],
) (*Block[T], error) {
	// Default options
	options := &BlockOptions[T]{
		Epsilon: ops.FromFloat64(1e-6),
	}
	for _, opt := range opts {
		opt(options)
	}

	ffn, err := core.NewFFN[T]("ffn", engine, ops, modelDim, ffnDim, modelDim)
	if err != nil {
		return nil, err
	}

	attnNorm, err := normalization.NewRMSNorm[T]("attn_norm", engine, ops, modelDim, normalization.WithRMSNormEpsilon[T](options.Epsilon))
	if err != nil {
		return nil, err
	}

	norm2, err := normalization.NewRMSNorm[T]("norm2", engine, ops, modelDim, normalization.WithRMSNormEpsilon[T](options.Epsilon))
	if err != nil {
		return nil, err
	}

	normPostAttention, err := normalization.NewRMSNorm[T]("normPostAttention", engine, ops, modelDim, normalization.WithRMSNormEpsilon[T](options.Epsilon))
	if err != nil {
		return nil, err
	}

	return &Block[T]{
		attention:         attention,
		ffn:               ffn,
		norm1:             attnNorm,
		norm2:             norm2,
		normPostAttention: normPostAttention,
	}, nil
}

// Forward computes the forward pass of the Transformer block.
func (b *Block[T]) Forward(ctx context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	x := inputs[0]

	// Attention part
	norm1Output, err := b.norm1.Forward(ctx, x)
	if err != nil {
		return nil, err
	}

	attnOutput, err := b.attention.Forward(ctx, norm1Output)
	if err != nil {
		return nil, err
	}
	// Residual connection
	attnOutput, err = b.ffn.Engine().Add(ctx, x, attnOutput)
	if err != nil {
		return nil, err
	}
	// Post-attention normalization
	attnOutput, err = b.normPostAttention.Forward(ctx, attnOutput)
	if err != nil {
		return nil, err
	}

	// FFN part
	norm2Output, err := b.norm2.Forward(ctx, attnOutput)
	if err != nil {
		return nil, err
	}

	ffnOutput, err := b.ffn.Forward(ctx, norm2Output)
	if err != nil {
		return nil, err
	}
	// Residual connection
	ffnOutput, err = b.ffn.Engine().Add(ctx, attnOutput, ffnOutput)
	if err != nil {
		return nil, err
	}

	return ffnOutput, nil
}

// Backward computes the backward pass of the Transformer block.
func (b *Block[T]) Backward(_ context.Context, _ types.BackwardMode, dOut *tensor.TensorNumeric[T], _ ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	// This is a simplified backward pass and needs to be implemented correctly.
	return []*tensor.TensorNumeric[T]{dOut}, nil
}

// Parameters returns the parameters of the Transformer block.
func (b *Block[T]) Parameters() []*graph.Parameter[T] {
	var params []*graph.Parameter[T]

	params = append(params, b.attention.Parameters()...)
	params = append(params, b.ffn.Parameters()...)
	params = append(params, b.norm1.Parameters()...)
	params = append(params, b.norm2.Parameters()...)
	params = append(params, b.normPostAttention.Parameters()...)

	return params
}

// OutputShape returns the output shape of the Transformer block.
func (b *Block[T]) OutputShape() []int {
	return b.attention.OutputShape()
}

// Attention returns the attention layer of the Transformer block.
func (b *Block[T]) Attention() graph.Node[T] {
	return b.attention
}

// Engine returns the compute engine used by the Transformer block.
func (b *Block[T]) Engine() compute.Engine[T] {
	return b.ffn.Engine()
}

// Attributes returns the attributes of the Transformer block.
func (b *Block[T]) Attributes() map[string]any {
	return nil
}

// OpType returns the operator type of the Transformer block.
func (b *Block[T]) OpType() string {
	return "TransformerBlock"
}
