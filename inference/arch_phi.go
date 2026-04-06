package inference

import (
	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/zerfoo/model/gguf"
	"github.com/zerfoo/ztensor/tensor"
)

// buildPhiGraph constructs a computation graph for the Phi architecture
// from pre-loaded GGUF tensors. It returns the graph and the embedding table
// tensor (needed by the generator for token lookup).
//
// Phi differs from Llama primarily in partial rotary embeddings: RoPE is
// applied to only a fraction of each head's dimensions (e.g., 0.5 for Phi-3
// mini means RoPE on 48 of 96 dims). The remaining dimensions pass through
// unchanged, preserving absolute positional information.
//
// The Phi architecture is:
//
//	Embed -> [RMSNorm -> GQA(partial RoPE) -> Add -> RMSNorm -> FFN(SwiGLU) -> Add] x N -> RMSNorm -> LMHead
func buildPhiGraph(
	tensors map[string]*tensor.TensorNumeric[float32],
	cfg *gguf.ModelConfig,
	engine compute.Engine[float32],
) (*graph.Graph[float32], *tensor.TensorNumeric[float32], error) {
	tl := newTensorLookup(tensors)

	embedWeight, err := tl.Lookup("model.embed_tokens.weight")
	if err != nil {
		return nil, nil, err
	}

	// Phi can tie lm_head to embedding weights.
	lmHeadWeight, ok := tl.Optional("lm_head.weight")
	if !ok {
		lmHeadWeight = embedWeight
	}

	opts := transformerGraphOpts{
		partialRotaryFactor: cfg.PartialRotaryFactor,
	}

	g, err := buildTransformerGraph(tensors, cfg, engine, embedWeight, lmHeadWeight, opts)
	if err != nil {
		return nil, nil, err
	}

	return g, embedWeight, nil
}
