package inference

import (
	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/zerfoo/model/gguf"
	"github.com/zerfoo/ztensor/tensor"
)

// buildExaoneGraph constructs a computation graph for the EXAONE architecture
// (LG AI Research) from pre-loaded GGUF tensors. It returns the graph and the
// embedding table tensor (needed by the generator for token lookup).
//
// EXAONE is structurally identical to Llama: GQA attention with RoPE and
// SwiGLU FFN. It uses standard GGUF tensor naming and supports tied embeddings.
//
// The EXAONE architecture is:
//
//	Embed -> [RMSNorm -> GQA -> Add -> RMSNorm -> FFN(SwiGLU) -> Add] x N -> RMSNorm -> LMHead
func buildExaoneGraph(
	tensors map[string]*tensor.TensorNumeric[float32],
	cfg *gguf.ModelConfig,
	engine compute.Engine[float32],
) (*graph.Graph[float32], *tensor.TensorNumeric[float32], error) {
	tl := newTensorLookup(tensors)

	embedWeight, err := tl.Lookup("model.embed_tokens.weight")
	if err != nil {
		return nil, nil, err
	}

	// EXAONE can tie lm_head to embedding weights.
	lmHeadWeight, ok := tl.Optional("lm_head.weight")
	if !ok {
		lmHeadWeight = embedWeight
	}

	g, err := buildTransformerGraph(tensors, cfg, engine, embedWeight, lmHeadWeight, transformerGraphOpts{
		residual: ResidualConfigFromGGUF(cfg.ResidualMode, cfg.AttnResNumBlocks),
	})
	if err != nil {
		return nil, nil, err
	}

	return g, embedWeight, nil
}
