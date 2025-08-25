package model

import (
	"context"

	"github.com/zerfoo/zerfoo/graph"
	"github.com/zerfoo/zerfoo/layers/embeddings"
	"github.com/zerfoo/zerfoo/tensor"
)

// Model represents a complete model, including a token embedding layer and a computation graph.
type Model[T tensor.Numeric] struct {
	Embedding  *embeddings.TokenEmbedding[T]
	Graph      *graph.Graph[T]
	ZMFVersion string
}

// NewModel creates a new model.
func NewModel[T tensor.Numeric](embedding *embeddings.TokenEmbedding[T], g *graph.Graph[T]) *Model[T] {
	return &Model[T]{
		Embedding: embedding,
		Graph:     g,
	}
}

// Forward performs the forward pass of the model.
func (m *Model[T]) Forward(ctx context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	// 1. Get embeddings
	embeddingTensors, err := m.Embedding.Forward(ctx, inputs...)
	if err != nil {
		return nil, err
	}

	// 2. Pass through the graph
	output, err := m.Graph.Forward(ctx, embeddingTensors)
	if err != nil {
		return nil, err
	}

	return output, nil
}
