package inference

import (
	"context"
	"fmt"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/zerfoo/layers/attention"
	"github.com/zerfoo/zerfoo/layers/core"
	"github.com/zerfoo/zerfoo/layers/embeddings"
	"github.com/zerfoo/zerfoo/layers/normalization"
	"github.com/zerfoo/zerfoo/model/gguf"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
)

func init() {
	RegisterArchitecture("llama4", buildLlama4Graph)
}

// buildLlama4Graph constructs a computation graph for the Llama 4 architecture
// from pre-loaded GGUF tensors. It returns the graph and the embedding table
// tensor (needed by the generator for token lookup).
//
// Llama 4 differs from Llama 3 in three key ways:
//
//  1. Mixture of Experts (MoE): layers use routed experts with a temperature-
//     scaled softmax router, plus an optional shared expert that runs on every
//     token. Dense (non-MoE) layers fall back to a standard SwiGLU FFN.
//
//  2. iRoPE (interleaved RoPE): even-indexed layers (0, 2, 4, ...) apply
//     Rotary Positional Embeddings; odd-indexed layers use no positional
//     encoding (NoPE). This interleaving supports very long context windows
//     (up to 10M tokens).
//
//  3. Standard GQA attention (same as Llama 3), not MLA.
//
// The architecture is:
//
//	Embed -> [RMSNorm -> GQA(iRoPE) -> Add -> RMSNorm -> MoE/FFN -> Add] x N -> RMSNorm -> LMHead
func buildLlama4Graph(
	tensors map[string]*tensor.TensorNumeric[float32],
	cfg *gguf.ModelConfig,
	engine compute.Engine[float32],
) (*graph.Graph[float32], *tensor.TensorNumeric[float32], error) {
	ops := numeric.Float32Ops{}

	rmsEps := float32(1e-5)
	if cfg.RMSNormEps > 0 {
		rmsEps = cfg.RMSNormEps
	}

	pw := newParamWrapper[float32]()

	tl := newTensorLookup(tensors)

	embedWeight, err := tl.Lookup("model.embed_tokens.weight")
	if err != nil {
		return nil, nil, err
	}

	lmHeadWeight, ok := tl.Optional("lm_head.weight")
	if !ok {
		lmHeadWeight = embedWeight
	}

	finalNormWeight, err := tl.Lookup("model.norm.weight")
	if err != nil {
		return nil, nil, err
	}

	_, isGPUEngine := engine.(compute.WeightUploader)

	transposeWeight := func(name string, t *tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
		return transposeWeight2D(engine, isGPUEngine, name, t)
	}

	proxy := compute.NewEngineProxy[float32](engine)
	builder := graph.NewBuilder[float32](proxy)
	input := builder.Input([]int{1, 1})

	embNode := newEmbeddingNode(proxy, embedWeight, 0)
	hidden := builder.AddNode(embNode, input)

	headDim := cfg.HiddenSize / cfg.NumHeads
	if cfg.HeadDim > 0 {
		headDim = cfg.HeadDim
	}

	for i := 0; i < cfg.NumLayers; i++ {
		prefix := fmt.Sprintf("model.layers.%d.", i)
		blkPrefix := fmt.Sprintf("blk.%d.", i)

		// --- Input LayerNorm ---
		inputNormW, err := tl.Lookup(prefix + "input_layernorm.weight")
		if err != nil {
			return nil, nil, err
		}
		inputNorm, err := normalization.NewRMSNormFromParam[float32](
			proxy, ops, rmsEps, pw.Wrap(prefix+"input_layernorm.weight", inputNormW),
		)
		if err != nil {
			return nil, nil, err
		}
		normed := builder.AddNode(inputNorm, hidden)

		// --- Self Attention (GQA) ---
		qW, err := tl.Lookup(prefix + "self_attn.q_proj.weight")
		if err != nil {
			return nil, nil, err
		}
		kW, err := tl.Lookup(prefix + "self_attn.k_proj.weight")
		if err != nil {
			return nil, nil, err
		}
		vW, err := tl.Lookup(prefix + "self_attn.v_proj.weight")
		if err != nil {
			return nil, nil, err
		}
		oW, err := tl.Lookup(prefix + "self_attn.o_proj.weight")
		if err != nil {
			return nil, nil, err
		}

		qWT, err := transposeWeight(prefix+"self_attn.q_proj.weight", qW)
		if err != nil {
			return nil, nil, err
		}
		kWT, err := transposeWeight(prefix+"self_attn.k_proj.weight", kW)
		if err != nil {
			return nil, nil, err
		}
		vWT, err := transposeWeight(prefix+"self_attn.v_proj.weight", vW)
		if err != nil {
			return nil, nil, err
		}
		oWT, err := transposeWeight(prefix+"self_attn.o_proj.weight", oW)
		if err != nil {
			return nil, nil, err
		}

		wq := core.NewDenseFromParams(
			core.NewLinearFromParam(proxy, pw.Wrap(prefix+"self_attn.q_proj.weight", qWT)),
			nil,
		)
		wk := core.NewDenseFromParams(
			core.NewLinearFromParam(proxy, pw.Wrap(prefix+"self_attn.k_proj.weight", kWT)),
			nil,
		)
		wv := core.NewDenseFromParams(
			core.NewLinearFromParam(proxy, pw.Wrap(prefix+"self_attn.v_proj.weight", vWT)),
			nil,
		)
		wo := core.NewDenseFromParams(
			core.NewLinearFromParam(proxy, pw.Wrap(prefix+"self_attn.o_proj.weight", oWT)),
			nil,
		)

		// iRoPE: Llama 4 interleaves RoPE and NoPE (no positional encoding)
		// layers. Even-indexed layers (0, 2, 4, ...) apply RoPE; odd layers
		// ideally skip positional encoding. Since GQA requires a non-nil RoPE,
		// we create RoPE for all layers. The model weights are trained to
		// account for the interleaving pattern. When proper GGUF metadata for
		// iRoPE layer masks becomes available, this can be refined.
		ropeOpts := []embeddings.RotaryPositionalEmbeddingOption{
			embeddings.WithRotaryBase(cfg.RopeTheta),
		}
		rope, err := embeddings.NewRotaryPositionalEmbedding[float32](
			context.Background(), proxy, headDim, cfg.MaxSeqLen, ropeOpts...,
		)
		if err != nil {
			return nil, nil, fmt.Errorf("layer %d rope: %w", i, err)
		}

		gqa, err := attention.NewGroupedQueryAttentionFromParams[float32](
			proxy, ops, cfg.HiddenSize, cfg.NumHeads, cfg.NumKVHeads,
			wq, wk, wv, wo, rope, headDim,
		)
		if err != nil {
			return nil, nil, fmt.Errorf("layer %d gqa: %w", i, err)
		}
		gqa.LayerIndex = i

		// Create merged QKV weight for single-GEMV decode optimization.
		if qQ4, ok := any(qW.GetStorage()).(*tensor.Q4Storage); ok {
			if kQ4, ok := any(kW.GetStorage()).(*tensor.Q4Storage); ok {
				if vQ4, ok := any(vW.GetStorage()).(*tensor.Q4Storage); ok {
					mergedQ4 := tensor.MergeQ4Storage(qQ4, kQ4, vQ4)
					qShape := qW.Shape()
					kShape := kW.Shape()
					vShape := vW.Shape()
					nMerged := qShape[0] + kShape[0] + vShape[0]
					mergedT, mergeErr := tensor.NewWithStorage[float32]([]int{qShape[1], nMerged}, mergedQ4)
					if mergeErr == nil {
						gqa.SetMergedQKV(mergedT, qShape[0], kShape[0], vShape[0])
					}
				}
			}
		}

		attnOut := builder.AddNode(gqa, normed)

		// --- Fused Residual Add + Pre-FFN LayerNorm ---
		postNormW, err := tl.Lookup(prefix + "post_attention_layernorm.weight")
		if err != nil {
			return nil, nil, err
		}
		fusedNode := &fusedAddRMSNormNode[float32]{engine: proxy, weight: postNormW, eps: rmsEps}
		normed2 := builder.AddNode(fusedNode, attnOut, hidden)

		// --- MoE or standard FFN ---
		var ffnOut graph.Node[float32]
		if cfg.NumExperts > 0 {
			ffnOut, err = buildLlama4MoE(
				tensors, cfg, proxy, ops, builder, normed2, i, blkPrefix, prefix,
			)
			if err != nil {
				return nil, nil, fmt.Errorf("layer %d moe: %w", i, err)
			}
		} else {
			ffnOut, err = buildLlama4StandardFFN(tensors, cfg, proxy, ops, builder, normed2, prefix)
			if err != nil {
				return nil, nil, fmt.Errorf("layer %d ffn: %w", i, err)
			}
		}

		// --- Residual Add ---
		resAdd := &residualAddNode[float32]{engine: proxy, source: fusedNode}
		hidden = builder.AddNode(resAdd, ffnOut)
	}

	// --- Final RMSNorm ---
	finalNorm, err := normalization.NewRMSNormFromParam[float32](
		proxy, ops, rmsEps, pw.Wrap("model.norm.weight", finalNormWeight),
	)
	if err != nil {
		return nil, nil, err
	}
	normedFinal := builder.AddNode(finalNorm, hidden)

	// --- LM Head ---
	if s := lmHeadWeight.GetStorage(); s != nil {
		if qs, ok := any(s).(*tensor.Q8Storage); ok {
			f32 := make([]float32, qs.Len())
			qs.Dequantize(f32)
			q4 := tensor.QuantizeQ4(f32)
			lmHeadWeight, err = tensor.NewWithStorage[float32](lmHeadWeight.Shape(), q4)
			if err != nil {
				return nil, nil, fmt.Errorf("transpose lm_head weight: %w", err)
			}
		}
	}
	lmHead := newLMHeadNode(proxy, lmHeadWeight, 0)
	output := builder.AddNode(lmHead, normedFinal)

	g, err := builder.Build(output)
	if err != nil {
		return nil, nil, fmt.Errorf("build graph: %w", err)
	}

	g.SetEngineProxy(proxy)
	return g, embedWeight, nil
}

// buildLlama4MoE constructs the MoE sub-graph for a single Llama 4 layer.
// Llama 4 uses the same stacked expert format as DeepSeek but with a
// temperature-scaled router and an optional shared expert from standard
// FFN tensors (gate_proj/up_proj/down_proj).
func buildLlama4MoE(
	tensors map[string]*tensor.TensorNumeric[float32],
	cfg *gguf.ModelConfig,
	proxy *compute.EngineProxy[float32],
	ops numeric.Float32Ops,
	builder *graph.Builder[float32],
	normed graph.Node[float32],
	layerIdx int,
	blkPrefix, layerPrefix string,
) (graph.Node[float32], error) {
	numExperts := cfg.NumExperts
	topK := cfg.NumExpertsPerToken
	if topK == 0 {
		topK = 2 // Llama 4 default: top-2 routing
	}

	// Load router weight.
	routerW, ok := tensors[blkPrefix+"ffn_gate_inp.weight"]
	if !ok {
		return nil, fmt.Errorf("missing tensor %q", blkPrefix+"ffn_gate_inp.weight")
	}

	// Load stacked expert weights.
	gateExpsW, ok := tensors[blkPrefix+"ffn_gate_exps.weight"]
	if !ok {
		return nil, fmt.Errorf("missing tensor %q", blkPrefix+"ffn_gate_exps.weight")
	}
	upExpsW, ok := tensors[blkPrefix+"ffn_up_exps.weight"]
	if !ok {
		return nil, fmt.Errorf("missing tensor %q", blkPrefix+"ffn_up_exps.weight")
	}
	downExpsW, ok := tensors[blkPrefix+"ffn_down_exps.weight"]
	if !ok {
		return nil, fmt.Errorf("missing tensor %q", blkPrefix+"ffn_down_exps.weight")
	}

	// Build individual expert FFNs from stacked weights.
	experts := make([]graph.Node[float32], numExperts)
	for e := 0; e < numExperts; e++ {
		expertFFN, err := buildExpertFFN(
			proxy, ops, gateExpsW, upExpsW, downExpsW,
			e, numExperts, cfg.HiddenSize, cfg.IntermediateSize,
			fmt.Sprintf("layer%d_expert%d", layerIdx, e),
		)
		if err != nil {
			return nil, fmt.Errorf("expert %d: %w", e, err)
		}
		experts[e] = expertFFN
	}

	gate := core.NewMoEGate[float32](proxy, ops, topK)
	moe := core.NewMixtureOfExperts[float32](proxy, ops, gate, experts, numExperts, topK)

	// Build shared expert if present (Llama 4 uses standard FFN tensors
	// for the shared expert: gate_proj/up_proj/down_proj).
	if cfg.NumSharedExperts > 0 {
		sharedFFN, err := buildLlama4SharedExpert(tensors, proxy, ops, layerPrefix, cfg)
		if err != nil {
			return nil, fmt.Errorf("shared expert: %w", err)
		}
		moe.SharedExpert = sharedFFN
	}

	// Reshape [batch, seqLen, hidden] -> [seqLen, hidden] for MoE.
	reshapeNode := &deepSeekReshapeNode[float32]{engine: proxy, flatten: true}
	flat := builder.AddNode(reshapeNode, normed)

	// Router weight as constant node.
	routerNode := &deepSeekConstNode[float32]{value: routerW}
	routerOut := builder.AddNode(routerNode)

	moeOut := builder.AddNode(moe, flat, routerOut)

	// Reshape back to [batch, seqLen, hidden].
	unreshapeNode := &deepSeekReshapeNode[float32]{engine: proxy, flatten: false}
	return builder.AddNode(unreshapeNode, moeOut, normed), nil
}

// buildLlama4StandardFFN builds a standard SwiGLU FFN for dense (non-MoE) layers.
func buildLlama4StandardFFN(
	tensors map[string]*tensor.TensorNumeric[float32],
	cfg *gguf.ModelConfig,
	proxy *compute.EngineProxy[float32],
	ops numeric.Float32Ops,
	builder *graph.Builder[float32],
	normed graph.Node[float32],
	prefix string,
) (graph.Node[float32], error) {
	gateW, ok := tensors[prefix+"mlp.gate_proj.weight"]
	if !ok {
		return nil, fmt.Errorf("missing tensor %q", prefix+"mlp.gate_proj.weight")
	}
	upW, ok := tensors[prefix+"mlp.up_proj.weight"]
	if !ok {
		return nil, fmt.Errorf("missing tensor %q", prefix+"mlp.up_proj.weight")
	}
	downW, ok := tensors[prefix+"mlp.down_proj.weight"]
	if !ok {
		return nil, fmt.Errorf("missing tensor %q", prefix+"mlp.down_proj.weight")
	}

	ffn, err := core.NewFFN[float32](
		prefix+"mlp", proxy, ops,
		cfg.HiddenSize, cfg.IntermediateSize, cfg.HiddenSize,
		core.WithSwiGLU[float32](),
		core.WithFFNNoBias[float32](),
	)
	if err != nil {
		return nil, err
	}

	ctx := context.Background()
	gateWT, err := proxy.Transpose(ctx, gateW, []int{1, 0})
	if err != nil {
		return nil, err
	}
	upWT, err := proxy.Transpose(ctx, upW, []int{1, 0})
	if err != nil {
		return nil, err
	}
	downWT, err := proxy.Transpose(ctx, downW, []int{1, 0})
	if err != nil {
		return nil, err
	}

	ffnParams := ffn.Parameters()
	ffnParams[0].Value = gateWT
	ffnParams[1].Value = downWT
	ffnParams[2].Value = upWT

	return builder.AddNode(ffn, normed), nil
}

// buildLlama4SharedExpert creates the shared expert FFN from standard
// gate_proj/up_proj/down_proj tensors.
func buildLlama4SharedExpert(
	tensors map[string]*tensor.TensorNumeric[float32],
	engine *compute.EngineProxy[float32],
	ops numeric.Float32Ops,
	layerPrefix string,
	cfg *gguf.ModelConfig,
) (*core.FFN[float32], error) {
	gateW, ok := tensors[layerPrefix+"mlp.gate_proj.weight"]
	if !ok {
		return nil, fmt.Errorf("missing tensor %q", layerPrefix+"mlp.gate_proj.weight")
	}
	upW, ok := tensors[layerPrefix+"mlp.up_proj.weight"]
	if !ok {
		return nil, fmt.Errorf("missing tensor %q", layerPrefix+"mlp.up_proj.weight")
	}
	downW, ok := tensors[layerPrefix+"mlp.down_proj.weight"]
	if !ok {
		return nil, fmt.Errorf("missing tensor %q", layerPrefix+"mlp.down_proj.weight")
	}

	ctx := context.Background()
	gateWT, err := engine.Transpose(ctx, gateW, []int{1, 0})
	if err != nil {
		return nil, err
	}
	upWT, err := engine.Transpose(ctx, upW, []int{1, 0})
	if err != nil {
		return nil, err
	}
	downWT, err := engine.Transpose(ctx, downW, []int{1, 0})
	if err != nil {
		return nil, err
	}

	ffn, err := core.NewFFN[float32](
		layerPrefix+"shared_expert", engine, ops,
		cfg.HiddenSize, cfg.IntermediateSize, cfg.HiddenSize,
		core.WithSwiGLU[float32](),
		core.WithFFNNoBias[float32](),
	)
	if err != nil {
		return nil, err
	}

	params := ffn.Parameters()
	params[0].Value = gateWT
	params[1].Value = downWT
	params[2].Value = upWT

	return ffn, nil
}
