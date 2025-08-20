package training

import (
	"context"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/graph"
	"github.com/zerfoo/zerfoo/layers/core"
	"github.com/zerfoo/zerfoo/layers/embeddings"
	"github.com/zerfoo/zerfoo/layers/transformer"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
)

// GemmaModel combines the token embedding, Gemma stack, and LM head.
type GemmaModel[T tensor.Numeric] struct {
	embedding *embeddings.TokenEmbedding[T]
	stack     *transformer.GemmaStack[T]
	lmHead    *core.LMHead[T]
}

// NewGemmaModel creates a new GemmaModel.
func NewGemmaModel[T tensor.Numeric](
	engine compute.Engine[T],
	ops numeric.Arithmetic[T],
	vocabSize, modelDim, numQueryHeads, numKeyValueHeads, ffnDim int,
	epsilon T,
	base float64,
	maxSeqLen, numLayers, localWindowSize, globalInterval int,
) (*GemmaModel[T], error) {
	embedding, err := embeddings.NewTokenEmbedding[T](engine, vocabSize, modelDim)
	if err != nil {
		return nil, err
	}

	stack, err := transformer.NewGemmaStack[T](engine, ops, modelDim, numQueryHeads, numKeyValueHeads, ffnDim, epsilon, base, maxSeqLen, numLayers, localWindowSize, globalInterval)
	if err != nil {
		return nil, err
	}

	lmHead, err := core.NewLMHead[T](engine, ops, modelDim, vocabSize)
	if err != nil {
		return nil, err
	}

	// Share weights between embedding and lm_head
	embeddingWeights := embedding.Parameters()[0].Value
	transposedWeights, err := engine.Transpose(context.Background(), embeddingWeights, []int{1, 0})
	if err != nil {
		return nil, err
	}
	lmHead.SetWeights(transposedWeights)

	return &GemmaModel[T]{
		embedding: embedding,
		stack:     stack,
		lmHead:    lmHead,
	}, nil
}

// Forward computes the forward pass of the GemmaModel.
func (m *GemmaModel[T]) Forward(ctx context.Context, inputs *tensor.Tensor[int]) (*tensor.Tensor[T], error) {
	// Note: The input to the model is a tensor of token IDs (int), but the graph nodes expect T.
	// The embedding layer handles this conversion.
	embedded, err := m.embedding.Forward(ctx, inputs)
	if err != nil {
		return nil, err
	}

	stackOutput, err := m.stack.Forward(ctx, embedded)
	if err != nil {
		return nil, err
	}

	logits, err := m.lmHead.Forward(ctx, stackOutput)
	if err != nil {
		return nil, err
	}

	return logits, nil
}

// Parameters returns all parameters of the GemmaModel.
func (m *GemmaModel[T]) Parameters() []*graph.Parameter[T] {
	var params []*graph.Parameter[T]
	params = append(params, m.embedding.Parameters()...)
	params = append(params, m.stack.Parameters()...)
	// LM head parameters are shared with embedding, so we don't add them again.
	return params
}