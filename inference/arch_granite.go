package inference

import (
	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/zerfoo/model/gguf"
	"github.com/zerfoo/ztensor/tensor"
)

// buildGraniteGraph constructs a computation graph for the IBM Granite
// architecture from pre-loaded GGUF tensors. It returns the graph and the
// embedding table tensor (needed by the generator for token lookup).
//
// Granite differences from Llama:
//   - Embedding multiplier: Granite may scale embeddings by a configurable
//     factor (granite.embedding_multiplier).
//   - Attention bias: Some Granite variants include bias on QKV projections.
//     Detected by checking for bias tensors in the tensor map.
//   - Logit softcapping: Guardian 3.3 variants may use logit softcap
//     (granite.logit_scale / granite.final_logit_softcapping).
//   - LM head may be tied to embedding weights.
//
// The Granite architecture is:
//
//	Embed*scale -> [RMSNorm -> GQA -> Add -> RMSNorm -> FFN(SwiGLU) -> Add] x N -> RMSNorm -> LMHead
func buildGraniteGraph(
	tensors map[string]*tensor.TensorNumeric[float32],
	cfg *gguf.ModelConfig,
	engine compute.Engine[float32],
) (*graph.Graph[float32], *tensor.TensorNumeric[float32], error) {
	tl := newTensorLookup(tensors)

	embedWeight, err := tl.Lookup("model.embed_tokens.weight")
	if err != nil {
		return nil, nil, err
	}

	// Granite may tie lm_head to embedding weights.
	lmHeadWeight, ok := tl.Optional("lm_head.weight")
	if !ok {
		lmHeadWeight = embedWeight
	}

	opts := transformerGraphOpts{
		residual: ResidualConfigFromGGUF(cfg.ResidualMode, cfg.AttnResNumBlocks),
	}

	// Embedding multiplier: use config value if set.
	if cfg.EmbeddingMultiplier > 0 {
		opts.embedScale = cfg.EmbeddingMultiplier
	}

	// Logit softcapping from config.
	if cfg.LogitSoftcap > 0 {
		opts.logitSoftcap = cfg.LogitSoftcap
	}

	// Detect attention bias by checking if bias tensors exist for layer 0.
	if tl.Has("model.layers.0.self_attn.q_proj.bias") {
		opts.attnBias = true
	}

	g, err := buildTransformerGraph(tensors, cfg, engine, embedWeight, lmHeadWeight, opts)
	if err != nil {
		return nil, nil, err
	}

	return g, embedWeight, nil
}
