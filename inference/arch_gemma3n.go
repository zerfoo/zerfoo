package inference

import (
	"math"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/zerfoo/model/gguf"
	"github.com/zerfoo/ztensor/tensor"
)

func init() {
	RegisterArchitecture("gemma3n", buildGemma3nGraph)
}

// buildGemma3nGraph constructs a computation graph for the Gemma 3n architecture
// from pre-loaded GGUF tensors. It returns the graph and the embedding table
// tensor (needed by the generator for token lookup and tied LM head).
//
// Gemma 3n is a mobile-optimized variant of Gemma 3 designed for on-device
// deployment. It shares the Gemma 3 transformer backbone:
//   - LM head is always tied to the embedding weight (no separate lm_head.weight).
//   - Embedding inputs are scaled by sqrt(hidden_size).
//   - Post-attention and post-FFN norms (4 norms per layer).
//   - Q/K norms after projection.
//   - Logit softcapping.
//
// Gemma 3n-specific features:
//   - Smaller hidden dimensions and fewer layers for mobile efficiency.
//   - MatFormer-style nested architecture for variable-width inference.
//   - Per-Layer Embeddings (PLE) for parameter-efficient layer specialization.
//
// The architecture is:
//
//	Embed*sqrt(d) -> [RMSNorm -> GQA(QKNorm) -> PostAttnNorm -> Add -> RMSNorm -> FFN(SwiGLU) -> PostFFNNorm -> Add] x N -> RMSNorm -> LMHead(tied, softcap)
func buildGemma3nGraph(
	tensors map[string]*tensor.TensorNumeric[float32],
	cfg *gguf.ModelConfig,
	engine compute.Engine[float32],
) (*graph.Graph[float32], *tensor.TensorNumeric[float32], error) {
	tl := newTensorLookup(tensors)

	embedWeight, err := tl.Lookup("model.embed_tokens.weight")
	if err != nil {
		return nil, nil, err
	}

	// Gemma 3n always ties LM head to embedding weights.
	// Gemma 3n scales embeddings by sqrt(hidden_size), same as Gemma 3.
	scale := float32(math.Sqrt(float64(cfg.HiddenSize)))
	opts := transformerGraphOpts{
		embedScale:  scale,
		postNorm:    true,
		qkNorm:      true,
		logitSoftcap: cfg.LogitSoftcap,
	}

	g, err := buildTransformerGraph(tensors, cfg, engine, embedWeight, embedWeight, opts)
	if err != nil {
		return nil, nil, err
	}

	return g, embedWeight, nil
}
