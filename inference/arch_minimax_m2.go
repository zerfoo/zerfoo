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

// buildMiniMaxM2Graph constructs a computation graph for the MiniMax-M2
// architecture from pre-loaded GGUF tensors. It returns the graph and the
// embedding table tensor.
//
// MiniMax-M2 is a pure full-softmax-attention MoE transformer. All layers are
// identical: GQA attention (with QK norms and partial RoPE) followed by a
// sigmoid-gated MoE FFN with routing bias.
//
// The architecture is:
//
//	Embed -> [RMSNorm -> GQA(QKNorm, PartialRoPE) -> Add -> RMSNorm -> MoE(sigmoid+bias) -> Add] x N -> RMSNorm -> LMHead
func buildMiniMaxM2Graph(
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

	// Global tensors use GGUF names.
	embedWeight, ok := tensors["token_embd.weight"]
	if !ok {
		return nil, nil, fmt.Errorf("missing tensor %q", "token_embd.weight")
	}

	lmHeadWeight, ok := tensors["output.weight"]
	if !ok {
		lmHeadWeight = embedWeight
	}

	finalNormWeight, err := tl.Lookup("output_norm.weight")
	if err != nil {
		return nil, nil, err
	}

	proxy := compute.NewEngineProxy[float32](engine)
	builder := graph.NewBuilder[float32](proxy)
	input := builder.Input([]int{1, 1})

	// Embedding lookup.
	embNode := newEmbeddingNode(proxy, embedWeight, 0)
	hidden := builder.AddNode(embNode, input)

	headDim := cfg.HiddenSize / cfg.NumHeads
	if cfg.HeadDim > 0 {
		headDim = cfg.HeadDim
	}

	// Partial RoPE: rotary_dim=64 of head_dim=128 -> fraction=0.5.
	partialFraction := float64(0)
	if cfg.PartialRotaryFactor > 0 && cfg.PartialRotaryFactor < 1 {
		partialFraction = float64(cfg.PartialRotaryFactor)
	}

	numExperts := cfg.NumExperts
	topK := cfg.NumExpertsPerToken
	if topK == 0 {
		topK = 8
	}

	for i := 0; i < cfg.NumLayers; i++ {
		blk := fmt.Sprintf("blk.%d.", i)

		// --- Input LayerNorm ---
		inputNormW, err := tl.Lookup(blk + "attn_norm.weight")
		if err != nil {
			return nil, nil, err
		}
		inputNorm, err := normalization.NewRMSNormFromParam[float32](
			proxy, ops, rmsEps, pw.Wrap(blk+"attn_norm.weight", inputNormW),
		)
		if err != nil {
			return nil, nil, err
		}
		normed := builder.AddNode(inputNorm, hidden)

		// --- GQA Attention with QK Norms and Partial RoPE ---
		qW, err := tl.Lookup(blk + "attn_q.weight")
		if err != nil {
			return nil, nil, err
		}
		kW, err := tl.Lookup(blk + "attn_k.weight")
		if err != nil {
			return nil, nil, err
		}
		vW, err := tl.Lookup(blk + "attn_v.weight")
		if err != nil {
			return nil, nil, err
		}
		oW, err := tl.Lookup(blk + "attn_output.weight")
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
			core.NewLinearFromParam(proxy, pw.Wrap(blk+"attn_q.weight", qWT)), nil,
		)
		wk := core.NewDenseFromParams(
			core.NewLinearFromParam(proxy, pw.Wrap(blk+"attn_k.weight", kWT)), nil,
		)
		wv := core.NewDenseFromParams(
			core.NewLinearFromParam(proxy, pw.Wrap(blk+"attn_v.weight", vWT)), nil,
		)
		wo := core.NewDenseFromParams(
			core.NewLinearFromParam(proxy, pw.Wrap(blk+"attn_output.weight", oWT)), nil,
		)

		ropeOpts := []embeddings.RotaryPositionalEmbeddingOption{
			embeddings.WithRotaryBase(cfg.RopeTheta),
		}
		if partialFraction > 0 && partialFraction < 1 {
			ropeOpts = append(ropeOpts, embeddings.WithRotaryDimFraction(partialFraction))
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

		// QK norms: apply per-layer RMSNorm to Q and K projections.
		qNormW, err := tl.Lookup(blk + "attn_q_norm.weight")
		if err != nil {
			return nil, nil, err
		}
		qNorm, err := normalization.NewRMSNormFromParam[float32](
			proxy, ops, rmsEps, pw.Wrap(blk+"attn_q_norm.weight", qNormW),
		)
		if err != nil {
			return nil, nil, err
		}
		kNormW, err := tl.Lookup(blk + "attn_k_norm.weight")
		if err != nil {
			return nil, nil, err
		}
		kNorm, err := normalization.NewRMSNormFromParam[float32](
			proxy, ops, rmsEps, pw.Wrap(blk+"attn_k_norm.weight", kNormW),
		)
		if err != nil {
			return nil, nil, err
		}
		gqa.SetQKNorms(qNorm, kNorm)
		gqa.SetQKNormWeights(qNormW, kNormW, rmsEps)
		// MiniMax-M2 stores q_norm weight as [nH*hD], not [hD]. Apply norm
		// before the head reshape so the weight broadcasts correctly.
		qNormShape := qNormW.Shape()
		if len(qNormShape) > 0 && qNormShape[len(qNormShape)-1] > headDim {
			gqa.SetQKNormPreReshape(true)
		}

		attnOut := builder.AddNode(gqa, normed)

		// --- Fused Residual Add + Pre-FFN LayerNorm ---
		ffnNormW, err := tl.Lookup(blk + "ffn_norm.weight")
		if err != nil {
			return nil, nil, err
		}
		fusedNode := &fusedAddRMSNormNode[float32]{engine: proxy, weight: ffnNormW, eps: rmsEps}
		normed2 := builder.AddNode(fusedNode, attnOut, hidden)

		// --- MoE FFN with Sigmoid Gating and Routing Bias ---
		ffnOut, err := buildMiniMaxM2MoE(
			tensors, cfg, proxy, ops, builder, normed2, i, blk, numExperts, topK,
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
		proxy, ops, rmsEps, pw.Wrap("output_norm.weight", finalNormWeight),
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

// buildMiniMaxM2MoE constructs the MoE sub-graph for a single MiniMax-M2 layer.
// It uses sigmoid gating with an optional routing bias tensor.
func buildMiniMaxM2MoE(
	tensors map[string]*tensor.TensorNumeric[float32],
	cfg *gguf.ModelConfig,
	proxy *compute.EngineProxy[float32],
	ops numeric.Float32Ops,
	builder *graph.Builder[float32],
	normed graph.Node[float32],
	layerIdx int,
	blk string,
	numExperts, topK int,
) (graph.Node[float32], error) {
	// Load router weight.
	routerW, ok := tensors[blk+"ffn_gate_inp.weight"]
	if !ok {
		return nil, fmt.Errorf("missing tensor %q", blk+"ffn_gate_inp.weight")
	}

	// Load stacked expert weights.
	gateExpsW, ok := tensors[blk+"ffn_gate_exps.weight"]
	if !ok {
		return nil, fmt.Errorf("missing tensor %q", blk+"ffn_gate_exps.weight")
	}
	upExpsW, ok := tensors[blk+"ffn_up_exps.weight"]
	if !ok {
		return nil, fmt.Errorf("missing tensor %q", blk+"ffn_up_exps.weight")
	}
	downExpsW, ok := tensors[blk+"ffn_down_exps.weight"]
	if !ok {
		return nil, fmt.Errorf("missing tensor %q", blk+"ffn_down_exps.weight")
	}

	// Split stacked expert weights into individual expert FFNs.
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

	// Build gate with sigmoid gating and optional routing bias.
	gateOpts := []core.MoEGateOption[float32]{
		core.WithSigmoidGating[float32](),
	}
	if routingBias, ok := tensors[blk+"exp_probs_b"]; ok {
		gateOpts = append(gateOpts, core.WithRoutingBias(routingBias))
	}
	gate := core.NewMoEGate[float32](proxy, ops, topK, gateOpts...)
	moe := core.NewMixtureOfExperts[float32](proxy, ops, gate, experts, numExperts, topK)

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
