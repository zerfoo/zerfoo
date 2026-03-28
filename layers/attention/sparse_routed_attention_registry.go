package attention

import (
	"context"
	"errors"
	"fmt"

	"github.com/zerfoo/zerfoo/layers/embeddings"
	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
)

// BuildSparseRoutedAttention constructs a SparseRoutedAttention node for the
// model builder. It reads num_heads, num_kv_heads, head_dim, segment_size,
// top_k, max_seq_len, and rope_base from attributes. A nil KV cache is used
// since cache binding happens at generation time.
func BuildSparseRoutedAttention[T tensor.Numeric](
	engine compute.Engine[T],
	ops numeric.Arithmetic[T],
	name string,
	params map[string]*graph.Parameter[T],
	attributes map[string]interface{},
) (graph.Node[T], error) {
	numHeads, ok := attributes["num_heads"].(int)
	if !ok {
		return nil, errors.New("missing or invalid attribute: num_heads")
	}

	numKVHeads, ok := attributes["num_kv_heads"].(int)
	if !ok {
		return nil, errors.New("missing or invalid attribute: num_kv_heads")
	}

	headDim, ok := attributes["head_dim"].(int)
	if !ok {
		return nil, errors.New("missing or invalid attribute: head_dim")
	}

	segmentSize, ok := attributes["segment_size"].(int)
	if !ok {
		return nil, errors.New("missing or invalid attribute: segment_size")
	}

	topK, ok := attributes["top_k"].(int)
	if !ok {
		return nil, errors.New("missing or invalid attribute: top_k")
	}

	maxSeqLen, ok := attributes["max_seq_len"].(int)
	if !ok {
		return nil, errors.New("missing or invalid attribute: max_seq_len")
	}

	base := 10000.0
	if b, ok := attributes["rope_base"].(float64); ok {
		base = b
	}

	rope, err := embeddings.NewRotaryPositionalEmbedding[T](
		context.Background(), engine, headDim, maxSeqLen,
		embeddings.WithRotaryBase(base),
	)
	if err != nil {
		return nil, fmt.Errorf("failed to create RotaryPositionalEmbedding: %w", err)
	}

	// KV cache is nil at graph-build time; it is bound during generation.
	return NewSparseRoutedAttention(engine, ops, rope, nil, numHeads, numKVHeads, headDim, segmentSize, topK)
}
