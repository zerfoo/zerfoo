package inference

import (
	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/zerfoo/model/gguf"
	"github.com/zerfoo/ztensor/tensor"
)

// buildStarCoder2Graph constructs a computation graph for the StarCoder2 (BigCode)
// architecture from pre-loaded GGUF tensors. It returns the graph and the embedding
// table tensor (needed by the generator for token lookup).
//
// StarCoder2 is structurally similar to Mistral:
//   - GQA (grouped-query) or MQA (multi-query) attention with RoPE.
//   - Sliding window attention (causal mask restricted to last N positions).
//   - SwiGLU FFN.
//   - LM head may be tied to the embedding weight.
//
// The StarCoder2 architecture is:
//
//	Embed -> [RMSNorm -> GQA(sliding window) -> Add -> RMSNorm -> FFN(SwiGLU) -> Add] x N -> RMSNorm -> LMHead
func buildStarCoder2Graph(
	tensors map[string]*tensor.TensorNumeric[float32],
	cfg *gguf.ModelConfig,
	engine compute.Engine[float32],
) (*graph.Graph[float32], *tensor.TensorNumeric[float32], error) {
	tl := newTensorLookup(tensors)

	embedWeight, err := tl.Lookup("model.embed_tokens.weight")
	if err != nil {
		return nil, nil, err
	}

	// StarCoder2 may tie lm_head to embedding weights.
	lmHeadWeight, ok := tl.Optional("lm_head.weight")
	if !ok {
		lmHeadWeight = embedWeight
	}

	opts := transformerGraphOpts{
		slidingWindowSize: cfg.SlidingWindow,
	}

	g, err := buildTransformerGraph(tensors, cfg, engine, embedWeight, lmHeadWeight, opts)
	if err != nil {
		return nil, nil, err
	}

	return g, embedWeight, nil
}
