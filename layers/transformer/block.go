// Package transformer provides transformer building blocks such as the
// Transformer `Block` used in encoder/decoder stacks.
package transformer

import (
	"context"
	"fmt"

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
	engine            compute.Engine[T]

	// Cached forward intermediates for backward pass.
	fwdInput     *tensor.TensorNumeric[T] // x
	fwdNorm1Out  *tensor.TensorNumeric[T] // norm1(x)
	fwdResidual1 *tensor.TensorNumeric[T] // x + attention(norm1(x))
	fwdPostAttn  *tensor.TensorNumeric[T] // normPostAttention(residual1)
	fwdNorm2Out  *tensor.TensorNumeric[T] // norm2(postAttn)
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
		engine:            engine,
	}, nil
}

// Forward computes the forward pass of the Transformer block.
func (b *Block[T]) Forward(ctx context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	x := inputs[0]
	b.fwdInput = x

	// Attention part
	norm1Output, err := b.norm1.Forward(ctx, x)
	if err != nil {
		return nil, err
	}
	b.fwdNorm1Out = norm1Output

	attnOutput, err := b.attention.Forward(ctx, norm1Output)
	if err != nil {
		return nil, err
	}
	// Residual connection
	attnOutput, err = b.engine.Add(ctx, x, attnOutput)
	if err != nil {
		return nil, err
	}
	b.fwdResidual1 = attnOutput

	// Post-attention normalization
	attnOutput, err = b.normPostAttention.Forward(ctx, attnOutput)
	if err != nil {
		return nil, err
	}
	b.fwdPostAttn = attnOutput

	// FFN part
	norm2Output, err := b.norm2.Forward(ctx, attnOutput)
	if err != nil {
		return nil, err
	}
	b.fwdNorm2Out = norm2Output

	ffnOutput, err := b.ffn.Forward(ctx, norm2Output)
	if err != nil {
		return nil, err
	}
	// Residual connection
	ffnOutput, err = b.engine.Add(ctx, attnOutput, ffnOutput)
	if err != nil {
		return nil, err
	}

	return ffnOutput, nil
}

// Backward computes the backward pass of the Transformer block.
//
// The forward pass is:
//
//	n1  = norm1(x)
//	a   = attention(n1)
//	r1  = x + a              (residual 1)
//	npa = normPostAttention(r1)
//	n2  = norm2(npa)
//	f   = ffn(n2)
//	out = npa + f             (residual 2)
//
// Backward reverses this, splitting gradients at each residual addition.
func (b *Block[T]) Backward(ctx context.Context, mode types.BackwardMode, dOut *tensor.TensorNumeric[T], _ ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	// Residual 2: out = npa + f → d_npa = dOut, d_f = dOut
	dF := dOut
	dNPA := dOut

	// FFN backward: f = ffn(n2) → d_n2
	dN2, err := b.ffn.Backward(ctx, mode, dF, b.fwdNorm2Out)
	if err != nil {
		return nil, fmt.Errorf("ffn backward: %w", err)
	}

	// norm2 backward: n2 = norm2(npa) → d_npa_from_norm2
	dNPAFromNorm2, err := b.norm2.Backward(ctx, mode, dN2[0], b.fwdPostAttn)
	if err != nil {
		return nil, fmt.Errorf("norm2 backward: %w", err)
	}

	// Accumulate gradients at npa
	dNPATotal, err := b.engine.Add(ctx, dNPA, dNPAFromNorm2[0])
	if err != nil {
		return nil, fmt.Errorf("accumulate npa gradient: %w", err)
	}

	// normPostAttention backward: npa = normPostAttention(r1) → d_r1
	dR1, err := b.normPostAttention.Backward(ctx, mode, dNPATotal, b.fwdResidual1)
	if err != nil {
		return nil, fmt.Errorf("normPostAttention backward: %w", err)
	}

	// Residual 1: r1 = x + a → d_x = d_r1, d_a = d_r1
	dA := dR1[0]
	dX := dR1[0]

	// attention backward: a = attention(n1) → d_n1
	dN1, err := b.attention.Backward(ctx, mode, dA, b.fwdNorm1Out)
	if err != nil {
		return nil, fmt.Errorf("attention backward: %w", err)
	}

	// norm1 backward: n1 = norm1(x) → d_x_from_norm1
	dXFromNorm1, err := b.norm1.Backward(ctx, mode, dN1[0], b.fwdInput)
	if err != nil {
		return nil, fmt.Errorf("norm1 backward: %w", err)
	}

	// Accumulate gradients at x
	dXTotal, err := b.engine.Add(ctx, dX, dXFromNorm1[0])
	if err != nil {
		return nil, fmt.Errorf("accumulate input gradient: %w", err)
	}

	return []*tensor.TensorNumeric[T]{dXTotal}, nil
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
	return b.engine
}

// Attributes returns the attributes of the Transformer block.
func (b *Block[T]) Attributes() map[string]any {
	return nil
}

// OpType returns the operator type of the Transformer block.
func (b *Block[T]) OpType() string {
	return "TransformerBlock"
}

// Statically assert that the type implements the graph.Node interface.
var _ graph.Node[float32] = (*Block[float32])(nil)
