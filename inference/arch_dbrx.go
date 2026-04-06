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

// buildDBRXGraph constructs a computation graph for the DBRX (Databricks) architecture
// from pre-loaded GGUF tensors. It returns the graph and the embedding table
// tensor (needed by the generator for token lookup).
//
// DBRX is a fine-grained Mixture of Experts model:
//   - 16 experts with top-4 routing (configurable via GGUF metadata).
//   - GQA attention with RoPE positional encoding.
//   - RMSNorm layer normalization.
//   - SwiGLU activation within each expert FFN.
//   - No shared expert.
//   - No sliding window attention.
//
// Expert tensor naming (stacked format, same as Mixtral):
//
//	blk.N.ffn_gate_inp.weight  — router weight [numExperts, hiddenSize]
//	blk.N.ffn_gate_exps.weight — stacked gate projections [numExperts, interSize, hiddenSize]
//	blk.N.ffn_up_exps.weight   — stacked up projections   [numExperts, interSize, hiddenSize]
//	blk.N.ffn_down_exps.weight — stacked down projections [numExperts, hiddenSize, interSize]
//
// The architecture is:
//
//	Embed -> [RMSNorm -> GQA(RoPE) -> Add -> RMSNorm -> MoE -> Add] x N -> RMSNorm -> LMHead
func buildDBRXGraph(
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

	numExperts := cfg.NumExperts
	if numExperts == 0 {
		numExperts = 16 // DBRX default: 16 experts
	}
	topK := cfg.NumExpertsPerToken
	if topK == 0 {
		topK = 4 // DBRX default: top-4 routing
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

		// --- Fused Residual Add + Pre-MoE LayerNorm ---
		postNormW, err := tl.Lookup(prefix + "post_attention_layernorm.weight")
		if err != nil {
			return nil, nil, err
		}
		fusedNode := &fusedAddRMSNormNode[float32]{engine: proxy, weight: postNormW, eps: rmsEps}
		normed2 := builder.AddNode(fusedNode, attnOut, hidden)

		// --- MoE (all DBRX layers are MoE) ---
		ffnOut, err := buildDBRXMoE(
			tensors, cfg, proxy, ops, builder, normed2,
			i, blkPrefix,
			numExperts, topK,
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

// buildDBRXMoE constructs the MoE sub-graph for a single DBRX layer.
// DBRX uses the same stacked expert format as Mixtral, with 16 experts
// and top-4 routing by default. No shared expert.
func buildDBRXMoE(
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
	// Load router weight: [numExperts, hiddenSize].
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
