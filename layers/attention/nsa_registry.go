package attention

import (
	"errors"
	"fmt"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
)

// BuildNativeSparseAttention constructs a NativeSparseAttention node for the
// model builder. It reads model_dim, num_heads, num_kv_heads, block_size,
// top_blocks, top_tokens, and window_size from attributes, and loads
// gate_coarse, gate_fine, gate_window from node parameters.
func BuildNativeSparseAttention[T tensor.Numeric](
	engine compute.Engine[T],
	ops numeric.Arithmetic[T],
	name string,
	params map[string]*graph.Parameter[T],
	attributes map[string]interface{},
) (graph.Node[T], error) {
	modelDim, ok := attributes["model_dim"].(int)
	if !ok {
		return nil, errors.New("missing or invalid attribute: model_dim")
	}

	numHeads, ok := attributes["num_heads"].(int)
	if !ok {
		return nil, errors.New("missing or invalid attribute: num_heads")
	}

	numKVHeads, ok := attributes["num_kv_heads"].(int)
	if !ok {
		return nil, errors.New("missing or invalid attribute: num_kv_heads")
	}

	blockSize, ok := attributes["block_size"].(int)
	if !ok {
		return nil, errors.New("missing or invalid attribute: block_size")
	}

	topBlocks, ok := attributes["top_blocks"].(int)
	if !ok {
		return nil, errors.New("missing or invalid attribute: top_blocks")
	}

	topTokens, ok := attributes["top_tokens"].(int)
	if !ok {
		return nil, errors.New("missing or invalid attribute: top_tokens")
	}

	windowSize, ok := attributes["window_size"].(int)
	if !ok {
		return nil, errors.New("missing or invalid attribute: window_size")
	}

	nsa, err := NewNativeSparseAttention[T](engine, ops, modelDim, numHeads, numKVHeads, blockSize, topBlocks, topTokens, windowSize)
	if err != nil {
		return nil, fmt.Errorf("failed to create NativeSparseAttention: %w", err)
	}

	// Override gate parameters from GGUF if provided.
	if p, ok := params[name+"_gate_coarse"]; ok {
		nsa.gateCoarse = p
	}
	if p, ok := params[name+"_gate_fine"]; ok {
		nsa.gateFine = p
	}
	if p, ok := params[name+"_gate_window"]; ok {
		nsa.gateWindow = p
	}

	return nsa, nil
}
