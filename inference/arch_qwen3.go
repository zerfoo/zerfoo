package inference

import (
	"github.com/zerfoo/zerfoo/model/gguf"
	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/tensor"
)

// buildQwen3Graph constructs a computation graph for the Qwen 3 architecture
// from pre-loaded GGUF tensors. It returns the graph and the embedding table
// tensor (needed by the generator for token lookup).
//
// Qwen 3 differences from Qwen 2, verified against the tensor list of
// Qwen/Qwen3-0.6B-GGUF (Qwen3-0.6B-Q8_0.gguf, GGUF architecture "qwen3"):
//
//   - QK RMSNorm: every block carries attn_q_norm.weight and attn_k_norm.weight
//     of shape [headDim] (128 for Qwen3-0.6B). Qwen 2 has no such tensors.
//   - No attention bias: the GGUF contains no attn_q.bias / attn_k.bias /
//     attn_v.bias tensors at all, where Qwen 2 always carries them.
//   - Head dimension is decoupled from HiddenSize/NumHeads. Qwen3-0.6B has
//     hiddenSize 1024 and 16 query heads, but headDim 128 (attn_q is
//     [2048, 1024] = 16*128 rows), not 1024/16 = 64. The head dimension comes
//     from the GGUF key qwen3.attention.key_length and is honored by
//     buildTransformerGraph via cfg.HeadDim.
//   - LM head is tied to the embedding table on the smaller sizes (the 0.6B
//     GGUF has no output.weight tensor), so lm_head falls back to the
//     embedding weight exactly as Qwen 2 does.
//
// RoPE theta is read from GGUF metadata as for every other architecture; Qwen 3
// GGUFs declare 1000000 for the sizes checked, the same default Qwen 2 uses.
//
// The Qwen 3 architecture is:
//
//	Embed -> [RMSNorm -> GQA(QK-norm, no bias) -> Add -> RMSNorm -> FFN(SwiGLU) -> Add] x N -> RMSNorm -> LMHead
//
// This builder covers the dense Qwen 3 sizes (0.6B, 1.7B, 4B, 8B, 14B, 32B).
// The mixture-of-experts sizes declare the distinct GGUF architecture
// "qwen3moe" and are deliberately not handled here, so they still fail with
// the "unsupported architecture" error from buildArchGraph's default branch.
func buildQwen3Graph(
	tensors map[string]*tensor.TensorNumeric[float32],
	cfg *gguf.ModelConfig,
	engine compute.Engine[float32],
) (*graph.Graph[float32], *tensor.TensorNumeric[float32], error) {
	tl := newTensorLookup(tensors)

	embedWeight, err := tl.Lookup("model.embed_tokens.weight")
	if err != nil {
		return nil, nil, err
	}

	// Qwen 3 ties lm_head to the embedding table on the smaller sizes.
	lmHeadWeight, ok := tl.Optional("lm_head.weight")
	if !ok {
		lmHeadWeight = embedWeight
	}

	opts := transformerGraphOpts{
		qkNorm:   true,  // Qwen 3 applies RMSNorm to Q and K after projection
		attnBias: false, // Qwen 3 dropped Qwen 2's Q/K/V projection biases
	}

	g, err := buildTransformerGraph(tensors, cfg, engine, embedWeight, lmHeadWeight, opts)
	if err != nil {
		return nil, nil, err
	}

	return g, embedWeight, nil
}
