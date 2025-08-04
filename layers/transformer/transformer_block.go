// Package transformer provides transformer-related neural network layers.
package transformer

import (
	"context"
	"fmt"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/graph"
	"github.com/zerfoo/zerfoo/layers/activations"
	"github.com/zerfoo/zerfoo/layers/attention"
	"github.com/zerfoo/zerfoo/layers/core"
	"github.com/zerfoo/zerfoo/layers/normalization"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
)

// TransformerBlock implements a single Transformer encoder block.
type TransformerBlock[T tensor.Numeric] struct { // TransformerBlock implements a single Transformer encoder block.
	engine           compute.Engine[T]
	ops              numeric.Arithmetic[T]
	modelDim         int // d_model
	numQueryHeads    int
	numKeyValueHeads int
	ffnDim           int // Dimension of the feed-forward network

	// Sub-layers
	layerNorm1    *normalization.LayerNormalization[T]
	gqa           *attention.GroupedQueryAttention[T]
	layerNorm2    *normalization.LayerNormalization[T]
	ffnGate       *core.Dense[T] // For SwiGLU
	ffnUp         *core.Dense[T] // For SwiGLU
	ffnActivation *activations.SwiGLU[T]
	ffnDown       *core.Dense[T]

	// Cached tensors for backward pass
	inputForLN1   *tensor.Tensor[T]
	outputFromGQA *tensor.Tensor[T]
	inputForLN2   *tensor.Tensor[T]
	ffnGateOutput *tensor.Tensor[T]
	ffnUpOutput   *tensor.Tensor[T]
	outputFromFFN *tensor.Tensor[T]
}

// NewTransformerBlock creates a new Transformer encoder block.
// modelDim: The input and output dimension of the block (d_model).
// numQueryHeads: The number of query heads.
// numKeyValueHeads: The number of key/value heads.
// ffnDim: The inner dimension of the feed-forward network.
// epsilon: Epsilon for Layer Normalization.
func NewTransformerBlock[T tensor.Numeric](engine compute.Engine[T], ops numeric.Arithmetic[T], modelDim, numQueryHeads, numKeyValueHeads, ffnDim int, epsilon T) (*TransformerBlock[T], error) {
	// Layer Normalization 1
	ln1, err := normalization.NewLayerNormalization[T](engine, modelDim, epsilon)
	if err != nil {
		return nil, fmt.Errorf("failed to create LayerNorm1: %w", err)
	}

	// Grouped Query Attention
	gqa, err := attention.NewGroupedQueryAttention[T](engine, ops, modelDim, numQueryHeads, numKeyValueHeads)
	if err != nil {
		return nil, fmt.Errorf("failed to create GroupedQueryAttention: %w", err)
	}

	// Layer Normalization 2
	ln2, err := normalization.NewLayerNormalization[T](engine, modelDim, epsilon)
	if err != nil {
		return nil, fmt.Errorf("failed to create LayerNorm2: %w", err)
	}

	// Feed-Forward Network (SwiGLU based)
	// Gemma uses a structure like: input -> Linear(ffnDim * 2) -> SwiGLU -> Linear(modelDim)
	ffnGate, err := core.NewDense[T]("ffn_gate", engine, ops, modelDim, ffnDim) // This will be multiplied by the gate
	if err != nil {
		return nil, fmt.Errorf("failed to create FFN Gate Dense: %w", err)
	}
	ffnUp, err := core.NewDense[T]("ffn_up", engine, ops, modelDim, ffnDim) // This will be the input to Sigmoid
	if err != nil {
		return nil, fmt.Errorf("failed to create FFN Up Dense: %w", err)
	}
	ffnActivation := activations.NewSwiGLU[T](engine, ops)
	ffnDown, err := core.NewDense[T]("ffn_down", engine, ops, ffnDim, modelDim)
	if err != nil {
		return nil, fmt.Errorf("failed to create FFN Down Dense: %w", err)
	}

	return &TransformerBlock[T]{
		engine:           engine,
		modelDim:         modelDim,
		numQueryHeads:    numQueryHeads,
		numKeyValueHeads: numKeyValueHeads,
		ffnDim:           ffnDim,
		layerNorm1:       ln1,
		gqa:              gqa,
		layerNorm2:       ln2,
		ffnGate:          ffnGate,
		ffnUp:            ffnUp,
		ffnActivation:    ffnActivation,
		ffnDown:          ffnDown,
	}, nil
}

// OutputShape returns the output shape, which is the same as the input shape.
func (tb *TransformerBlock[T]) OutputShape(inputShapes ...[]int) ([]int, error) {
	if len(inputShapes) != 1 {
		return nil, fmt.Errorf("TransformerBlock: %w, expected %d, got %d", graph.ErrInvalidInputCount, 1, len(inputShapes))
	}
	inputShape := inputShapes[0]
	if len(inputShape) != 3 || inputShape[2] != tb.modelDim {
		return nil, fmt.Errorf("expected 3D input tensor (batch, seq_len, model_dim) with model_dim %d, got %v", tb.modelDim, inputShape)
	}

	return inputShape, nil
}

// Parameters returns all trainable parameters from its sub-layers.
func (tb *TransformerBlock[T]) Parameters() []graph.Parameter[T] {
	var params []graph.Parameter[T]
	params = append(params, tb.layerNorm1.Parameters()...)
	params = append(params, tb.gqa.Parameters()...)
	params = append(params, tb.layerNorm2.Parameters()...)
	for _, p := range tb.ffnGate.Parameters() {
		params = append(params, *p)
	}
	for _, p := range tb.ffnUp.Parameters() {
		params = append(params, *p)
	}
	for _, p := range tb.ffnDown.Parameters() {
		params = append(params, *p)
	}

	return params
}

// Forward computes the Transformer Block's forward pass.
func (tb *TransformerBlock[T]) Forward(ctx context.Context, inputs ...*tensor.Tensor[T]) (*tensor.Tensor[T], error) {
	if len(inputs) != 1 {
		return nil, fmt.Errorf("TransformerBlock: %w, expected %d, got %d", graph.ErrInvalidInputCount, 1, len(inputs))
	}
	input := inputs[0]     // (batch_size, seq_len, model_dim)
	tb.inputForLN1 = input // Cache for backward

	// 1. Layer Normalization 1
	norm1Output, err := tb.layerNorm1.Forward(ctx, input)
	if err != nil {
		return nil, err
	}

	// 2. Grouped Query Attention (Self-Attention)
	// For self-attention, Q, K, V are all the same input (norm1Output)
	gqaOutput, err := tb.gqa.Forward(ctx, norm1Output)
	if err != nil {
		return nil, err
	}
	tb.outputFromGQA = gqaOutput // Cache for backward

	// 3. Add Residual Connection 1 (input + gqaOutput)
	residual1Output, err := tb.engine.Add(ctx, input, gqaOutput, nil)
	if err != nil {
		return nil, err
	}
	tb.inputForLN2 = residual1Output // Cache for backward

	// 4. Layer Normalization 2
	norm2Output, err := tb.layerNorm2.Forward(ctx, residual1Output)
	if err != nil {
		return nil, err
	}

	// 5. Feed-Forward Network (SwiGLU based)
	// Gemma FFN: input -> Linear(ffnDim * 2) -> SwiGLU -> Linear(modelDim)
	// Here, we have ffnGate (for x1) and ffnUp (for x2 in SwiGLU)
	ffnGateOutput, err := tb.ffnGate.Forward(norm2Output)
	if err != nil {
		return nil, err
	}
	tb.ffnGateOutput = ffnGateOutput // Cache for backward

	ffnUpOutput, err := tb.ffnUp.Forward(norm2Output)
	if err != nil {
		return nil, err
	}
	tb.ffnUpOutput = ffnUpOutput // Cache for backward

	// Concatenate ffnGateOutput and ffnUpOutput for SwiGLU input
	swiGLUInput, err := tb.engine.Concat(ctx, []*tensor.Tensor[T]{ffnGateOutput, ffnUpOutput}, len(ffnGateOutput.Shape())-1, nil)
	if err != nil {
		return nil, err
	}

	swiGLUOutput, err := tb.ffnActivation.Forward(ctx, swiGLUInput)
	if err != nil {
		return nil, err
	}

	ffnOutput, err := tb.ffnDown.Forward(swiGLUOutput)
	if err != nil {
		return nil, err
	}
	tb.outputFromFFN = ffnOutput // Cache for backward

	// 6. Add Residual Connection 2 (residual1Output + ffnOutput)
	finalOutput, err := tb.engine.Add(ctx, residual1Output, ffnOutput, nil)
	if err != nil {
		return nil, err
	}

	return finalOutput, nil
}

// Backward computes the gradients for the Transformer Block.
func (tb *TransformerBlock[T]) Backward(ctx context.Context, dOut *tensor.Tensor[T], inputs ...*tensor.Tensor[T]) ([]*tensor.Tensor[T], error) {
	if len(inputs) != 1 {
		return nil, fmt.Errorf("TransformerBlock: %w, expected %d, got %d", graph.ErrInvalidInputCount, 1, len(inputs))
	}
	// dOut is the gradient from the subsequent layer/loss
	// We need to propagate it backward through the block.

	// 1. Backward through Residual Connection 2
	dResidual1OutputFromRes2 := dOut
	dFFNOutput := dOut

	// 2. Backward through Feed-Forward Network (SwiGLU based)
	dSwiGLUOutput, err := tb.ffnDown.Backward(dFFNOutput)
	if err != nil {
		return nil, err
	}
	dSwiGLUOutputTensor := dSwiGLUOutput[0]

	// Reconstruct swiGLUInput for backward pass

	// Reconstruct swiGLUInput for backward pass
	swiGLUInput, err := tb.engine.Concat(ctx, []*tensor.Tensor[T]{tb.ffnGateOutput, tb.ffnUpOutput}, len(tb.ffnGateOutput.Shape())-1, nil)
	if err != nil {
		return nil, err
	}
	dSwiGLUInput, err := tb.ffnActivation.Backward(ctx, dSwiGLUOutputTensor, swiGLUInput)
	if err != nil {
		return nil, err
	}
	dSwiGLUInputTensor := dSwiGLUInput[0]

	splitTensors, err := tb.engine.Split(ctx, dSwiGLUInputTensor, 2, len(dSwiGLUInputTensor.Shape())-1)
	if err != nil {
		return nil, err
	}
	dFFNGateOutput, dFFNUpOutput := splitTensors[0], splitTensors[1]

	dNorm2OutputFromFFNGate, err := tb.ffnGate.Backward(dFFNGateOutput)
	if err != nil {
		return nil, err
	}
	dNorm2OutputFromFFNGateTensor := dNorm2OutputFromFFNGate[0]

	dNorm2OutputFromFFNUp, err := tb.ffnUp.Backward(dFFNUpOutput)
	if err != nil {
		return nil, err
	}
	dNorm2OutputFromFFNUpTensor := dNorm2OutputFromFFNUp[0]

	// Sum gradients for norm2Output from both FFN paths
	dNorm2OutputTotal, err := tb.engine.Add(ctx, dNorm2OutputFromFFNGateTensor, dNorm2OutputFromFFNUpTensor, nil)
	if err != nil {
		return nil, err
	}

	// 3. Backward through Layer Normalization 2
	dResidual1OutputFromLN2, err := tb.layerNorm2.Backward(ctx, dNorm2OutputTotal, tb.inputForLN2)
	if err != nil {
		return nil, err
	}
	dResidual1OutputFromLN2Tensor := dResidual1OutputFromLN2[0]

	// 4. Sum gradients for Residual Connection 1
	// dL/dResidual1Output = dL/dResidual1Output_from_res2 + dL/dResidual1Output_from_LN2
	dResidual1OutputTotal, err := tb.engine.Add(ctx, dResidual1OutputFromRes2, dResidual1OutputFromLN2Tensor, nil)
	if err != nil {
		return nil, err
	}

	// 5. Backward through Grouped Query Attention
	dNorm1OutputFromGQA, err := tb.gqa.Backward(ctx, dResidual1OutputTotal, tb.inputForLN1)
	if err != nil {
		return nil, err
	}
	dNorm1OutputFromGQATensor := dNorm1OutputFromGQA[0]

	// 6. Backward through Layer Normalization 1
	dInputFromLN1, err := tb.layerNorm1.Backward(ctx, dNorm1OutputFromGQATensor, tb.inputForLN1)
	if err != nil {
		return nil, err
	}
	dInputFromLN1Tensor := dInputFromLN1[0]

	// 7. Sum gradients for original input (from residual connection 1 and LN1 path)
	// dL/dInput = dL/dInput_from_res1 + dL/dInput_from_LN1
	// The dL/dInput_from_res1 is simply dResidual1OutputTotal (as it was added directly)
	dInputTotal, err := tb.engine.Add(ctx, dResidual1OutputTotal, dInputFromLN1Tensor, nil)
	if err != nil {
		return nil, err
	}

	return []*tensor.Tensor[T]{dInputTotal}, nil
}
