// Package transformer provides transformer-related neural network layers.
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
)

// Block implements a single Transformer encoder block.
type Block[T tensor.Numeric] struct {
	engine           compute.Engine[T]
	ops              numeric.Arithmetic[T]
	modelDim         int // d_model
	ffnDim           int // Dimension of the feed-forward network

	// Sub-layers
	rmsNorm1             *normalization.RMSNorm[T]
	attention            graph.Node[T]
	rmsNormPostAttention *normalization.RMSNorm[T]
	rmsNorm2             *normalization.RMSNorm[T]
	ffn                  *core.FFN[T]

	// Cached tensors for backward pass
	inputForLN1              *tensor.Tensor[T]
	outputFromAttention      *tensor.Tensor[T]
	outputFromPostAttentionNorm *tensor.Tensor[T]
	inputForLN2              *tensor.Tensor[T]
	outputFromFFN            *tensor.Tensor[T]
	outputShape              []int
}

// NewTransformerBlock creates a new Transformer encoder block.
func NewTransformerBlock[T tensor.Numeric](
	engine compute.Engine[T],
	ops numeric.Arithmetic[T],
	modelDim, ffnDim int,
	epsilon T,
	attentionLayer graph.Node[T],
) (*Block[T], error) {
	// RMS Normalization 1
	rmsNorm1, err := normalization.NewRMSNorm[T]("rmsNorm1", engine, ops, modelDim, epsilon)
	if err != nil {
		return nil, fmt.Errorf("failed to create RMSNorm1: %w", err)
	}

	// Post-Attention RMS Normalization
	rmsNormPostAttention, err := normalization.NewRMSNorm[T]("rmsNormPostAttention", engine, ops, modelDim, epsilon)
	if err != nil {
		return nil, fmt.Errorf("failed to create rmsNormPostAttention: %w", err)
	}

	// RMS Normalization 2
	rmsNorm2, err := normalization.NewRMSNorm[T]("rmsNorm2", engine, ops, modelDim, epsilon)
	if err != nil {
		return nil, fmt.Errorf("failed to create RMSNorm2: %w", err)
	}

	// Feed-Forward Network
	ffn, err := core.NewFFN[T]("ffn", engine, ops, modelDim, ffnDim, modelDim)
	if err != nil {
		return nil, fmt.Errorf("failed to create FFN: %w", err)
	}

	return &Block[T]{
		engine:               engine,
		ops:                  ops,
		modelDim:             modelDim,
		ffnDim:               ffnDim,
		rmsNorm1:             rmsNorm1,
		attention:            attentionLayer,
		rmsNormPostAttention: rmsNormPostAttention,
		rmsNorm2:             rmsNorm2,
		ffn:                  ffn,
	}, nil
}

// OutputShape returns the output shape, which is the same as the input shape.
func (tb *Block[T]) OutputShape() []int {
	return tb.outputShape
}

// Parameters returns all trainable parameters from its sub-layers.
func (tb *Block[T]) Parameters() []*graph.Parameter[T] {
	var params []*graph.Parameter[T]
	params = append(params, tb.rmsNorm1.Parameters()...)
	params = append(params, tb.attention.Parameters()...)
	params = append(params, tb.rmsNormPostAttention.Parameters()...)
	params = append(params, tb.rmsNorm2.Parameters()...)
	params = append(params, tb.ffn.Parameters()...)

	return params
}

// Forward computes the Transformer Block's forward pass.
func (tb *Block[T]) Forward(ctx context.Context, inputs ...*tensor.Tensor[T]) (*tensor.Tensor[T], error) {
	if len(inputs) != 1 {
		return nil, fmt.Errorf("TransformerBlock: %w, expected %d, got %d", graph.ErrInvalidInputCount, 1, len(inputs))
	}
	input := inputs[0]     // (batch_size, seq_len, model_dim)
	tb.inputForLN1 = input // Cache for backward
	tb.outputShape = input.Shape()

	// 1. Pre-Attention RMS Normalization
	norm1Output, err := tb.rmsNorm1.Forward(ctx, input)
	if err != nil {
		return nil, err
	}

	// 2. Attention
	attentionOutput, err := tb.attention.Forward(ctx, norm1Output)
	if err != nil {
		return nil, err
	}
	tb.outputFromAttention = attentionOutput // Cache for backward

	// 3. Post-Attention RMS Normalization
	postAttentionNormOutput, err := tb.rmsNormPostAttention.Forward(ctx, attentionOutput)
	if err != nil {
		return nil, err
	}
	tb.outputFromPostAttentionNorm = postAttentionNormOutput // Cache for backward

	// 4. Add Residual Connection 1 (input + postAttentionNormOutput)
	residual1Output, err := tb.engine.Add(ctx, input, postAttentionNormOutput, nil)
	if err != nil {
		return nil, err
	}
	tb.inputForLN2 = residual1Output // Cache for backward

	// 5. Pre-FFN RMS Normalization
	norm2Output, err := tb.rmsNorm2.Forward(ctx, residual1Output)
	if err != nil {
		return nil, err
	}

	// 6. Feed-Forward Network
	ffnOutput, err := tb.ffn.Forward(ctx, norm2Output)
	if err != nil {
		return nil, err
	}
	tb.outputFromFFN = ffnOutput // Cache for backward

	// 7. Add Residual Connection 2 (residual1Output + ffnOutput)
	finalOutput, err := tb.engine.Add(ctx, residual1Output, ffnOutput, nil)
	if err != nil {
		return nil, err
	}

	return finalOutput, nil
}

// Backward computes the gradients for the Transformer Block.
func (tb *Block[T]) Backward(ctx context.Context, dOut *tensor.Tensor[T], inputs ...*tensor.Tensor[T]) ([]*tensor.Tensor[T], error) {
	if len(inputs) != 1 {
		return nil, fmt.Errorf("TransformerBlock: %w, expected %d, got %d", graph.ErrInvalidInputCount, 1, len(inputs))
	}

	// 1. Backward through Residual Connection 2
	dResidual1OutputFromRes2 := dOut
	dFFNOutput := dOut

	// 2. Backward through Feed-Forward Network
	dNorm2Output, err := tb.ffn.Backward(ctx, dFFNOutput)
	if err != nil {
		return nil, err
	}
	dNorm2OutputTensor := dNorm2Output[0]

	// 3. Backward through Pre-FFN RMS Normalization 2
	dResidual1OutputFromLN2, err := tb.rmsNorm2.Backward(ctx, dNorm2OutputTensor, tb.inputForLN2)
	if err != nil {
		return nil, err
	}
	dResidual1OutputFromLN2Tensor := dResidual1OutputFromLN2[0]

	// 4. Sum gradients for first residual connection output
	dResidual1OutputTotal, err := tb.engine.Add(ctx, dResidual1OutputFromRes2, dResidual1OutputFromLN2Tensor, nil)
	if err != nil {
		return nil, err
	}

	// 5. Backward through Residual Connection 1
	dInputFromRes1 := dResidual1OutputTotal
	dPostAttentionNormOutput := dResidual1OutputTotal

	// 6. Backward through Post-Attention RMS Normalization
	dAttentionOutput, err := tb.rmsNormPostAttention.Backward(ctx, dPostAttentionNormOutput, tb.outputFromAttention)
	if err != nil {
		return nil, err
	}
	dAttentionOutputTensor := dAttentionOutput[0]

	// 7. Backward through Attention
	dNorm1Output, err := tb.attention.Backward(ctx, dAttentionOutputTensor, tb.inputForLN1)
	if err != nil {
		return nil, err
	}
	dNorm1OutputTensor := dNorm1Output[0]

	// 8. Backward through Pre-Attention RMS Normalization 1
	dInputFromLN1, err := tb.rmsNorm1.Backward(ctx, dNorm1OutputTensor, tb.inputForLN1)
	if err != nil {
		return nil, err
	}
	dInputFromLN1Tensor := dInputFromLN1[0]

	// 9. Sum gradients for original input
	dInputTotal, err := tb.engine.Add(ctx, dInputFromRes1, dInputFromLN1Tensor, nil)
	if err != nil {
		return nil, err
	}

	return []*tensor.Tensor[T]{dInputTotal}, nil
}