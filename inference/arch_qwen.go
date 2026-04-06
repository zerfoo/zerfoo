package inference

import (
	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/zerfoo/model/gguf"
	"github.com/zerfoo/ztensor/tensor"
)

// buildQwenGraph constructs a computation graph for the Qwen 2 architecture
// from pre-loaded GGUF tensors. It returns the graph and the embedding table
// tensor (needed by the generator for token lookup).
//
// Qwen 2 differences from Llama:
//   - Attention bias: Q/K/V projections include bias vectors.
//   - RoPE theta defaults to 1M (set in cfg.RopeTheta from GGUF metadata).
//   - LM head may be tied to embedding weights.
//
// The Qwen 2 architecture is:
//
//	Embed -> [RMSNorm -> GQA(bias) -> Add -> RMSNorm -> FFN(SwiGLU) -> Add] x N -> RMSNorm -> LMHead
func buildQwenGraph(
	tensors map[string]*tensor.TensorNumeric[float32],
	cfg *gguf.ModelConfig,
	engine compute.Engine[float32],
) (*graph.Graph[float32], *tensor.TensorNumeric[float32], error) {
	tl := newTensorLookup(tensors)

	embedWeight, err := tl.Lookup("model.embed_tokens.weight")
	if err != nil {
		return nil, nil, err
	}

	// Qwen can tie lm_head to embedding weights.
	lmHeadWeight, ok := tl.Optional("lm_head.weight")
	if !ok {
		lmHeadWeight = embedWeight
	}

	opts := transformerGraphOpts{
		attnBias: true, // Qwen 2 always uses attention bias
	}

	g, err := buildTransformerGraph(tensors, cfg, engine, embedWeight, lmHeadWeight, opts)
	if err != nil {
		return nil, nil, err
	}

	return g, embedWeight, nil
}
