package inference

import (
	"math"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/zerfoo/model/gguf"
	"github.com/zerfoo/ztensor/tensor"
)

// buildGemmaGraph constructs a computation graph for the Gemma architecture
// from pre-loaded GGUF tensors. It returns the graph and the embedding table
// tensor (needed by the generator for token lookup and tied LM head).
//
// Gemma differences from Llama:
//   - LM head is always tied to the embedding weight (no separate lm_head.weight).
//   - Embedding inputs are scaled by sqrt(hidden_size) before entering the transformer.
//
// The Gemma architecture is:
//
//	Embed*sqrt(d) -> [RMSNorm -> GQA -> Add -> RMSNorm -> FFN(SwiGLU) -> Add] x N -> RMSNorm -> LMHead(tied)
func buildGemmaGraph(
	tensors map[string]*tensor.TensorNumeric[float32],
	cfg *gguf.ModelConfig,
	engine compute.Engine[float32],
) (*graph.Graph[float32], *tensor.TensorNumeric[float32], error) {
	tl := newTensorLookup(tensors)

	embedWeight, err := tl.Lookup("model.embed_tokens.weight")
	if err != nil {
		return nil, nil, err
	}

	// Gemma always ties LM head to embedding weights.
	// Gemma scales embeddings by sqrt(hidden_size).
	scale := float32(math.Sqrt(float64(cfg.HiddenSize)))
	opts := transformerGraphOpts{
		embedScale: scale,
	}
	// Gemma 3 has post-attention/post-FFN norms, Q/K norms, and logit softcapping.
	if cfg.Architecture == "gemma3" {
		opts.postNorm = true
		opts.qkNorm = true
		opts.logitSoftcap = cfg.LogitSoftcap
	}
	g, err := buildTransformerGraph(tensors, cfg, engine, embedWeight, embedWeight, opts)
	if err != nil {
		return nil, nil, err
	}

	return g, embedWeight, nil
}
