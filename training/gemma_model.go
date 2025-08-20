package training

import (
	"context"
	"fmt"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/graph"
	"github.com/zerfoo/zerfoo/layers/core"
	"github.com/zerfoo/zerfoo/layers/embeddings"
	"github.com/zerfoo/zerfoo/layers/transformer"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
)

// GemmaModel is a decoder-only transformer model.
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
		return nil, fmt.Errorf("failed to create token embedding: %w", err)
	}

	stack, err := transformer.NewGemmaStack[T](engine, ops, modelDim, numQueryHeads, numKeyValueHeads, ffnDim, epsilon, base, maxSeqLen, numLayers, localWindowSize, globalInterval)
	if err != nil {
		return nil, fmt.Errorf("failed to create gemma stack: %w", err)
	}

	lmHead, err := core.NewLMHead[T]("lm_head", engine, ops, modelDim, vocabSize)
	if err != nil {
		return nil, fmt.Errorf("failed to create lm head: %w", err)
	}

	return &GemmaModel[T]{
		embedding: embedding,
		stack:     stack,
		lmHead:    lmHead,
	}, nil
}

// Forward performs the forward pass of the GemmaModel.
func (m *GemmaModel[T]) Forward(ctx context.Context, inputs ...*tensor.Tensor[int]) (*tensor.Tensor[T], error) {
	embedded, err := m.embedding.Forward(ctx, inputs[0])
	if err != nil {
		return nil, err
	}

	hidden, err := m.stack.Forward(ctx, embedded)
	if err != nil {
		return nil, err
	}

	logits, err := m.lmHead.Forward(ctx, hidden)
	if err != nil {
		return nil, err
	}

	return logits, nil
}

// Backward computes the gradients for the GemmaModel.
func (m *GemmaModel[T]) Backward(ctx context.Context, dOut *tensor.Tensor[T], inputs ...*tensor.Tensor[T]) ([]*tensor.Tensor[T], error) {
	// This is a simplified backward pass. A real implementation would need to
	// correctly chain the gradients through the layers.
	return nil, fmt.Errorf("GemmaModel backward pass not yet implemented")
}

// Parameters returns the parameters of the GemmaModel.
func (m *GemmaModel[T]) Parameters() []*graph.Parameter[T] {
	var params []*graph.Parameter[T]
	params = append(params, m.embedding.Parameters()...)
	params = append(params, m.stack.Parameters()...)
	params = append(params, m.lmHead.Parameters()...)
	return params
}
