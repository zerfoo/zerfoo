package retrieval

import (
	"context"
	"fmt"

	"github.com/zerfoo/zerfoo/inference"
)

// ContextualEmbedder adapts a decoder embedding model to retrieval roles.
// Prefixes are caller-owned instructions and may be empty.
type ContextualEmbedder struct {
	Model          *inference.EmbeddingModel
	QueryPrefix    string
	DocumentPrefix string
}

var _ Embedder = (*ContextualEmbedder)(nil)

func (e *ContextualEmbedder) EmbedQuery(ctx context.Context, query string) ([]float32, error) {
	if e == nil || e.Model == nil {
		return nil, fmt.Errorf("contextual embedder has no model")
	}
	return e.Model.EmbedText(ctx, e.QueryPrefix+query)
}

func (e *ContextualEmbedder) EmbedDocuments(ctx context.Context, texts []string) ([][]float32, error) {
	if e == nil || e.Model == nil {
		return nil, fmt.Errorf("contextual embedder has no model")
	}
	vectors := make([][]float32, len(texts))
	for i, text := range texts {
		vector, err := e.Model.EmbedText(ctx, e.DocumentPrefix+text)
		if err != nil {
			return nil, fmt.Errorf("embed document %d: %w", i, err)
		}
		vectors[i] = vector
	}
	return vectors, nil
}
