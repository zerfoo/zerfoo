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
	RegisterArchitecture("chatglm", buildGLM4Graph)
	RegisterArchitecture("glm4", buildGLM4Graph)
	RegisterArchitecture("glm-dsa", buildGLM4Graph)
	RegisterArchitecture("glm4moe", buildGLM4MoEGraph)
	RegisterArchitecture("deepseek2-ocr", buildDeepSeekGraph)
}

// buildGLM4Graph constructs a computation graph for the GLM4/ChatGLM
// architecture from pre-loaded GGUF tensors. It returns the graph and the
// embedding table tensor.
//
// GLM4 uses a Llama-like decoder-only transformer:
//   - GQA attention with RoPE
//   - SwiGLU FFN
//   - RMSNorm layer normalization
//   - Tied or separate LM head weights
//
// The architecture is:
//
//	Embed -> [RMSNorm -> GQA(RoPE) -> Add -> RMSNorm -> FFN(SwiGLU) -> Add] x N -> RMSNorm -> LMHead
func buildGLM4Graph(
	tensors map[string]*tensor.TensorNumeric[float32],
	cfg *gguf.ModelConfig,
	engine compute.Engine[float32],
) (*graph.Graph[float32], *tensor.TensorNumeric[float32], error) {
	tl := newTensorLookup(tensors)

	embedWeight, err := tl.Lookup("model.embed_tokens.weight")
	if err != nil {
		return nil, nil, err
	}

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

// buildGLM4MoEGraph constructs a computation graph for the GLM4-MoE
// architecture from pre-loaded GGUF tensors. GLM4-MoE extends GLM4 by
// replacing the dense FFN with a Mixture of Experts layer, following the
// same MoE pattern as Mixtral (top-K routing, stacked expert weights).
//
// The architecture is:
//
//	Embed -> [RMSNorm -> GQA(RoPE) -> Add -> RMSNorm -> MoE -> Add] x N -> RMSNorm -> LMHead
func buildGLM4MoEGraph(
	tensors map[string]*tensor.TensorNumeric[float32],
	cfg *gguf.ModelConfig,
	engine compute.Engine[float32],
) (*graph.Graph[float32], *tensor.TensorNumeric[float32], error) {
	ops := numeric.Float32Ops{}

	rmsEps := float32(1e-5)
	if cfg.RMSNormEps > 0 {
		rmsEps = cfg.RMSNormEps
	}

	tl := newTensorLookup(tensors)
	pw := newParamWrapper[float32]()

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

	numExperts := cfg.NumExperts
	if numExperts == 0 {
		numExperts = 8
	}
	topK := cfg.NumExpertsPerToken
	if topK == 0 {
		topK = 2
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

		// --- Input RMSNorm ---
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

		// --- Self Attention (GQA with RoPE) ---
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

		ctx := context.Background()
		qWT, err := engine.Transpose(ctx, qW, []int{1, 0})
		if err != nil {
			return nil, nil, fmt.Errorf("layer %d transpose q: %w", i, err)
		}
		kWT, err := engine.Transpose(ctx, kW, []int{1, 0})
		if err != nil {
			return nil, nil, fmt.Errorf("layer %d transpose k: %w", i, err)
		}
		vWT, err := engine.Transpose(ctx, vW, []int{1, 0})
		if err != nil {
			return nil, nil, fmt.Errorf("layer %d transpose v: %w", i, err)
		}
		oWT, err := engine.Transpose(ctx, oW, []int{1, 0})
		if err != nil {
			return nil, nil, fmt.Errorf("layer %d transpose o: %w", i, err)
		}

		wq := core.NewDenseFromParams(
			core.NewLinearFromParam(proxy, pw.Wrap(prefix+"self_attn.q_proj.weight", qWT)), nil,
		)
		wk := core.NewDenseFromParams(
			core.NewLinearFromParam(proxy, pw.Wrap(prefix+"self_attn.k_proj.weight", kWT)), nil,
		)
		wv := core.NewDenseFromParams(
			core.NewLinearFromParam(proxy, pw.Wrap(prefix+"self_attn.v_proj.weight", vWT)), nil,
		)
		wo := core.NewDenseFromParams(
			core.NewLinearFromParam(proxy, pw.Wrap(prefix+"self_attn.o_proj.weight", oWT)), nil,
		)

		ropeOpts := []embeddings.RotaryPositionalEmbeddingOption{
			embeddings.WithRotaryBase(cfg.RopeTheta),
		}
		rope, err := embeddings.NewRotaryPositionalEmbedding[float32](
			ctx, proxy, headDim, cfg.MaxSeqLen, ropeOpts...,
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

		attnOut := builder.AddNode(gqa, normed)

		// --- Fused Residual Add + Pre-MoE RMSNorm ---
		postNormW, err := tl.Lookup(prefix + "post_attention_layernorm.weight")
		if err != nil {
			return nil, nil, err
		}
		fusedNode := &fusedAddRMSNormNode[float32]{engine: proxy, weight: postNormW, eps: rmsEps}
		normed2 := builder.AddNode(fusedNode, attnOut, hidden)

		// --- MoE ---
		ffnOut, err := buildGLM4MoE(
			tensors, cfg, proxy, ops, builder, normed2,
			i, blkPrefix, numExperts, topK,
		)
		if err != nil {
			return nil, nil, fmt.Errorf("layer %d moe: %w", i, err)
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
	lmHead := newLMHeadNode(proxy, lmHeadWeight, 0)
	output := builder.AddNode(lmHead, normedFinal)

	g, err := builder.Build(output)
	if err != nil {
		return nil, nil, fmt.Errorf("build graph: %w", err)
	}

	g.SetEngineProxy(proxy)
	return g, embedWeight, nil
}

// buildGLM4MoE constructs the MoE sub-graph for a single GLM4-MoE layer.
// Follows the Mixtral stacked-expert format with top-K routing.
func buildGLM4MoE(
	tensors map[string]*tensor.TensorNumeric[float32],
	cfg *gguf.ModelConfig,
	proxy *compute.EngineProxy[float32],
	ops numeric.Float32Ops,
	builder *graph.Builder[float32],
	normed graph.Node[float32],
	layerIdx int,
	blkPrefix string,
	numExperts, topK int,
) (graph.Node[float32], error) {
	tl := newTensorLookup(tensors)

	routerW, err := tl.Lookup(blkPrefix + "ffn_gate_inp.weight")
	if err != nil {
		return nil, err
	}

	gateExpsW, err := tl.Lookup(blkPrefix + "ffn_gate_exps.weight")
	if err != nil {
		return nil, err
	}
	upExpsW, err := tl.Lookup(blkPrefix + "ffn_up_exps.weight")
	if err != nil {
		return nil, err
	}
	downExpsW, err := tl.Lookup(blkPrefix + "ffn_down_exps.weight")
	if err != nil {
		return nil, err
	}

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

	reshapeNode := &deepSeekReshapeNode[float32]{engine: proxy, flatten: true}
	flat := builder.AddNode(reshapeNode, normed)

	routerNode := &deepSeekConstNode[float32]{value: routerW}
	routerOut := builder.AddNode(routerNode)

	moeOut := builder.AddNode(moe, flat, routerOut)

	unreshapeNode := &deepSeekReshapeNode[float32]{engine: proxy, flatten: false}
	return builder.AddNode(unreshapeNode, moeOut, normed), nil
}
