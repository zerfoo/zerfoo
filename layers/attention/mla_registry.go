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

// BuildMultiHeadLatentAttention constructs a MultiHeadLatentAttention node
// for the model builder. It reads kv_lora_dim, num_heads, head_dim, and
// max_seq_len from attributes, and loads W_Q, W_DKV, W_UK, W_UV, W_O
// from node parameters.
func BuildMultiHeadLatentAttention[T tensor.Numeric](
	engine compute.Engine[T],
	ops numeric.Arithmetic[T],
	name string,
	params map[string]*graph.Parameter[T],
	attributes map[string]any,
) (graph.Node[T], error) {
	numHeads, ok := attributes["num_heads"].(int)
	if !ok {
		return nil, errors.New("missing or invalid attribute: num_heads")
	}

	headDim, ok := attributes["head_dim"].(int)
	if !ok {
		return nil, errors.New("missing or invalid attribute: head_dim")
	}

	kvLoraDim, ok := attributes["kv_lora_dim"].(int)
	if !ok {
		return nil, errors.New("missing or invalid attribute: kv_lora_dim")
	}

	maxSeqLen, ok := attributes["max_seq_len"].(int)
	if !ok {
		return nil, errors.New("missing or invalid attribute: max_seq_len")
	}

	// Load projection parameters.
	wqParam, ok := params[name+"_wq"]
	if !ok {
		return nil, fmt.Errorf("missing required parameter: %s_wq", name)
	}
	wdkvParam, ok := params[name+"_wdkv"]
	if !ok {
		return nil, fmt.Errorf("missing required parameter: %s_wdkv", name)
	}
	wukParam, ok := params[name+"_wuk"]
	if !ok {
		return nil, fmt.Errorf("missing required parameter: %s_wuk", name)
	}
	wuvParam, ok := params[name+"_wuv"]
	if !ok {
		return nil, fmt.Errorf("missing required parameter: %s_wuv", name)
	}
	woParam, ok := params[name+"_wo"]
	if !ok {
		return nil, fmt.Errorf("missing required parameter: %s_wo", name)
	}

	// Construct Dense layers from parameters (no bias for MLA projections).
	wQ := core.NewDenseFromParams(core.NewLinearFromParam(engine, wqParam), nil)
	wDKV := core.NewDenseFromParams(core.NewLinearFromParam(engine, wdkvParam), nil)
	wUK := core.NewDenseFromParams(core.NewLinearFromParam(engine, wukParam), nil)
	wUV := core.NewDenseFromParams(core.NewLinearFromParam(engine, wuvParam), nil)
	wO := core.NewDenseFromParams(core.NewLinearFromParam(engine, woParam), nil)

	// Partial RoPE dimension (0 means full headDim).
	ropeHeadDim, _ := attributes["rope_head_dim"].(int)

	// Build RoPE — sized to ropeHeadDim when partial RoPE is configured.
	ropeDim := headDim
	if ropeHeadDim > 0 && ropeHeadDim < headDim {
		ropeDim = ropeHeadDim
	}

	base := 10000.0
	if b, ok := attributes["rope_base"].(float64); ok {
		base = b
	}

	rope, err := embeddings.NewRotaryPositionalEmbedding[T](
		context.Background(), engine, ropeDim, maxSeqLen,
		embeddings.WithRotaryBase(base),
	)
	if err != nil {
		return nil, fmt.Errorf("failed to create RotaryPositionalEmbedding: %w", err)
	}

	return NewMultiHeadLatentAttention(engine, ops, numHeads, headDim, kvLoraDim, ropeHeadDim, wQ, wDKV, wUK, wUV, wO, rope), nil
}
