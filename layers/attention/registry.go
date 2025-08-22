// Package attention provides attention mechanisms for neural networks.
package attention

import (
	"context"
	"fmt"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/graph"
	"github.com/zerfoo/zerfoo/layers/core"
	"github.com/zerfoo/zerfoo/layers/embeddings"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
)

func buildGroupedQueryAttention[T tensor.Numeric](
	engine compute.Engine[T],
	ops numeric.Arithmetic[T],
	name string,
	params map[string]*graph.Parameter[T],
	attributes map[string]interface{},
) (graph.Node[T], error) {
	// Extract attributes
	modelDim, ok := attributes["model_dim"].(int)
	if !ok {
		return nil, fmt.Errorf("missing or invalid attribute: model_dim")
	}
	numQueryHeads, ok := attributes["num_query_heads"].(int)
	if !ok {
		return nil, fmt.Errorf("missing or invalid attribute: num_query_heads")
	}
	numKeyValueHeads, ok := attributes["num_key_value_heads"].(int)
	if !ok {
		return nil, fmt.Errorf("missing or invalid attribute: num_key_value_heads")
	}
	base, ok := attributes["rope_base"].(float64)
	if !ok {
		return nil, fmt.Errorf("missing or invalid attribute: rope_base")
	}
	maxSeqLen, ok := attributes["max_seq_len"].(int)
	if !ok {
		return nil, fmt.Errorf("missing or invalid attribute: max_seq_len")
	}

	// Extract parameters for sub-layers
	wq, ok := params[name+"_wq"]
	if !ok {
		return nil, fmt.Errorf("missing required parameter: %s_wq", name)
	}
	wk, ok := params[name+"_wk"]
	if !ok {
		return nil, fmt.Errorf("missing required parameter: %s_wk", name)
	}
	wv, ok := params[name+"_wv"]
	if !ok {
		return nil, fmt.Errorf("missing required parameter: %s_wv", name)
	}
	wo, ok := params[name+"_wo"]
	if !ok {
		return nil, fmt.Errorf("missing required parameter: %s_wo", name)
	}

	// Construct sub-layers from parameters
	wqDense := core.NewDenseFromParams(core.NewLinearFromParam(engine, wq), nil) // Assuming no bias for now
	wkDense := core.NewDenseFromParams(core.NewLinearFromParam(engine, wk), nil)
	wvDense := core.NewDenseFromParams(core.NewLinearFromParam(engine, wv), nil)
	woDense := core.NewDenseFromParams(core.NewLinearFromParam(engine, wo), nil)

	headDim := modelDim / numQueryHeads
	rope, err := embeddings.NewRotaryPositionalEmbedding[T](context.Background(), engine, headDim, maxSeqLen, embeddings.WithRotaryBase(base))
	if err != nil {
		return nil, fmt.Errorf("failed to create RotaryPositionalEmbedding: %w", err)
	}

	return NewGroupedQueryAttentionFromParams(
		engine, ops, modelDim, numQueryHeads, numKeyValueHeads,
		wqDense, wkDense, wvDense, woDense, rope,
	)
}

func buildGlobalAttention[T tensor.Numeric](
	engine compute.Engine[T],
	ops numeric.Arithmetic[T],
	name string,
	params map[string]*graph.Parameter[T],
	attributes map[string]interface{},
) (graph.Node[T], error) {
	// GlobalAttention is a wrapper around GroupedQueryAttention.
	// We can build the inner GQA and then wrap it.
	gqa, err := buildGroupedQueryAttention[T](engine, ops, name, params, attributes)
	if err != nil {
		return nil, err
	}

	return NewGlobalAttentionFromParams(gqa.(*GroupedQueryAttention[T])), nil
}
