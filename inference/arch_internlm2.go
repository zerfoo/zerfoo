package inference

import (
	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/zerfoo/model/gguf"
	"github.com/zerfoo/ztensor/tensor"
)

// buildInternLM2Graph constructs a computation graph for the InternLM2 architecture
// (Shanghai AI Lab) from pre-loaded GGUF tensors. It returns the graph and the
// embedding table tensor (needed by the generator for token lookup).
//
// InternLM2 is structurally identical to Llama: GQA attention with RoPE, SwiGLU FFN,
// RMSNorm, and optional weight tying. The GGUF tensor names follow the standard
// Llama convention so no architecture-specific name mapping is needed.
//
// The InternLM2 architecture is:
//
//	Embed -> [RMSNorm -> GQA -> Add -> RMSNorm -> FFN(SiLU-gate) -> Add] x N -> RMSNorm -> LMHead
func buildInternLM2Graph(
	tensors map[string]*tensor.TensorNumeric[float32],
	cfg *gguf.ModelConfig,
	engine compute.Engine[float32],
) (*graph.Graph[float32], *tensor.TensorNumeric[float32], error) {
	tl := newTensorLookup(tensors)

	embedWeight, err := tl.Lookup("model.embed_tokens.weight")
	if err != nil {
		return nil, nil, err
	}

	// InternLM2 may tie lm_head to embedding weights.
	lmHeadWeight, ok := tl.Optional("lm_head.weight")
	if !ok {
		lmHeadWeight = embedWeight
	}

	g, err := buildTransformerGraph(tensors, cfg, engine, embedWeight, lmHeadWeight, transformerGraphOpts{})
	if err != nil {
		return nil, nil, err
	}

	return g, embedWeight, nil
}
