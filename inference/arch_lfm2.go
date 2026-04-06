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
	RegisterArchitecture("lfm2", buildLFM2Graph)
	RegisterArchitecture("lfm2moe", buildLFM2MoEGraph)
}

// buildLFM2Graph constructs a computation graph for the Liquid Foundation
// Model 2 (LFM2) architecture from pre-loaded GGUF tensors. LFM2 is a
// dense transformer using GQA attention with RoPE, SwiGLU FFN, and RMSNorm.
//
// The architecture is:
//
//	Embed -> [RMSNorm -> GQA(RoPE) -> Add -> RMSNorm -> FFN(SwiGLU) -> Add] x N -> RMSNorm -> LMHead
func buildLFM2Graph(
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

// buildLFM2MoEGraph constructs a computation graph for the LFM2-MoE variant.
// LFM2-MoE is a hybrid architecture that combines dense transformer attention
// layers with MoE FFN layers (24B total parameters, 2B active per token).
//
// Layer dispatch: all layers use GQA attention, but the FFN is replaced with
// a top-K MoE router when expert tensors are present for that layer. Layers
// without expert tensors fall back to a standard dense SwiGLU FFN.
//
// The architecture is:
//
//	Embed -> [RMSNorm -> GQA(RoPE) -> Add -> RMSNorm -> (MoE|FFN) -> Add] x N -> RMSNorm -> LMHead
func buildLFM2MoEGraph(
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

		// --- Fused Residual Add + Pre-FFN RMSNorm ---
		postNormW, err := tl.Lookup(prefix + "post_attention_layernorm.weight")
		if err != nil {
			return nil, nil, err
		}
		fusedNode := &fusedAddRMSNormNode[float32]{engine: proxy, weight: postNormW, eps: rmsEps}
		normed2 := builder.AddNode(fusedNode, attnOut, hidden)

		// --- MoE or Dense FFN ---
		// Dispatch based on whether expert tensors exist for this layer.
		var ffnOut graph.Node[float32]
		if tl.Has(blkPrefix + "ffn_gate_inp.weight") {
			ffnOut, err = buildLFM2MoE(
				tensors, cfg, proxy, ops, builder, normed2,
				i, blkPrefix, numExperts, topK,
			)
			if err != nil {
				return nil, nil, fmt.Errorf("layer %d moe: %w", i, err)
			}
		} else {
			ffnOut, err = buildLFM2DenseFFN(
				tensors, cfg, proxy, ops, builder, normed2, prefix,
			)
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
	lmHead := newLMHeadNode(proxy, lmHeadWeight, 0)
	output := builder.AddNode(lmHead, normedFinal)

	g, err := builder.Build(output)
	if err != nil {
		return nil, nil, fmt.Errorf("build graph: %w", err)
	}

	g.SetEngineProxy(proxy)
	return g, embedWeight, nil
}

// buildLFM2DenseFFN builds a standard SwiGLU FFN for dense LFM2 layers.
func buildLFM2DenseFFN(
	tensors map[string]*tensor.TensorNumeric[float32],
	cfg *gguf.ModelConfig,
	proxy *compute.EngineProxy[float32],
	ops numeric.Float32Ops,
	builder *graph.Builder[float32],
	normed graph.Node[float32],
	prefix string,
) (graph.Node[float32], error) {
	tl := newTensorLookup(tensors)

	gateW, err := tl.Lookup(prefix + "mlp.gate_proj.weight")
	if err != nil {
		return nil, err
	}
	upW, err := tl.Lookup(prefix + "mlp.up_proj.weight")
	if err != nil {
		return nil, err
	}
	downW, err := tl.Lookup(prefix + "mlp.down_proj.weight")
	if err != nil {
		return nil, err
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

// buildLFM2MoE constructs the MoE sub-graph for a single LFM2-MoE layer.
func buildLFM2MoE(
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
