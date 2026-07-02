// Package attention provides attention mechanisms for neural networks.
package attention

import (
	"context"
	"errors"
	"fmt"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/zerfoo/layers/core"
	"github.com/zerfoo/zerfoo/layers/embeddings"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
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
		return nil, errors.New("missing or invalid attribute: model_dim")
	}

	numQueryHeads, ok := attributes["num_query_heads"].(int)
	if !ok {
		return nil, errors.New("missing or invalid attribute: num_query_heads")
	}

	numKeyValueHeads, ok := attributes["num_key_value_heads"].(int)
	if !ok {
		return nil, errors.New("missing or invalid attribute: num_key_value_heads")
	}

	base, ok := attributes["rope_base"].(float64)
	if !ok {
		return nil, errors.New("missing or invalid attribute: rope_base")
	}

	maxSeqLen, ok := attributes["max_seq_len"].(int)
	if !ok {
		return nil, errors.New("missing or invalid attribute: max_seq_len")
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

	// Construct sub-layers from parameters, with optional bias
	wqDense := core.NewDenseFromParams(core.NewLinearFromParam(engine, wq), optionalBias(engine, ops, params, name+"_wq_bias"))
	wkDense := core.NewDenseFromParams(core.NewLinearFromParam(engine, wk), optionalBias(engine, ops, params, name+"_wk_bias"))
	wvDense := core.NewDenseFromParams(core.NewLinearFromParam(engine, wv), optionalBias(engine, ops, params, name+"_wv_bias"))
	woDense := core.NewDenseFromParams(core.NewLinearFromParam(engine, wo), optionalBias(engine, ops, params, name+"_wo_bias"))

	headDim := modelDim / numQueryHeads

	// Build RoPE options
	ropeOpts := []embeddings.RotaryPositionalEmbeddingOption{embeddings.WithRotaryBase(base)}
	if scalingType, _ := attributes["rope_scaling_type"].(string); scalingType == "yarn" {
		factor, _ := attributes["rope_scaling_factor"].(float64)
		origMaxLen, _ := attributes["rope_scaling_orig_max_len"].(int)
		if factor > 0 && origMaxLen > 0 {
			ropeOpts = append(ropeOpts, embeddings.WithYaRNScaling(factor, origMaxLen))
		}
	}
	if fraction, ok := attributes["partial_rotary_factor"].(float64); ok && fraction > 0 && fraction < 1.0 {
		ropeOpts = append(ropeOpts, embeddings.WithRotaryDimFraction(fraction))
	}

	rope, err := embeddings.NewRotaryPositionalEmbedding[T](context.Background(), engine, headDim, maxSeqLen, ropeOpts...)
	if err != nil {
		return nil, fmt.Errorf("failed to create RotaryPositionalEmbedding: %w", err)
	}

	return NewGroupedQueryAttentionFromParams(
		engine, ops, modelDim, numQueryHeads, numKeyValueHeads,
		wqDense, wkDense, wvDense, woDense, rope,
	)
}

// optionalBias returns a *core.Bias[T] if the named parameter exists, nil otherwise.
func optionalBias[T tensor.Numeric](
	engine compute.Engine[T],
	ops numeric.Arithmetic[T],
	params map[string]*graph.Parameter[T],
	key string,
) *core.Bias[T] {
	p, ok := params[key]
	if !ok {
		return nil
	}
	return core.NewBiasFromParam(engine, ops, p)
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
