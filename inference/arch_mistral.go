package inference

import (
	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/zerfoo/model/gguf"
	"github.com/zerfoo/ztensor/tensor"
)

// buildMistralGraph constructs a computation graph for the Mistral architecture
// from pre-loaded GGUF tensors. It returns the graph and the embedding table
// tensor (needed by the generator for token lookup).
//
// Mistral differences from Llama:
//   - All layers use sliding window attention (causal mask restricted to last N positions).
//   - LM head may be tied to the embedding weight (no separate lm_head.weight).
//
// The Mistral architecture is:
//
//	Embed -> [RMSNorm -> GQA(sliding window) -> Add -> RMSNorm -> FFN(SwiGLU) -> Add] x N -> RMSNorm -> LMHead
func buildMistralGraph(
	tensors map[string]*tensor.TensorNumeric[float32],
	cfg *gguf.ModelConfig,
	engine compute.Engine[float32],
) (*graph.Graph[float32], *tensor.TensorNumeric[float32], error) {
	tl := newTensorLookup(tensors)

	embedWeight, err := tl.Lookup("model.embed_tokens.weight")
	if err != nil {
		return nil, nil, err
	}

	// Mistral may tie lm_head to embedding weights.
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
