package inference

import (
	"context"
	"fmt"
	"math"

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

// buildGemma4MoEGraph constructs a computation graph for the Gemma 4 26B-A4B
// MoE architecture from pre-loaded GGUF tensors. It extends the dense Gemma 4
// builder with Mixture of Experts routing for FFN layers.
//
// Gemma 4 26B-A4B has MoE on some layers and dense FFN on others. MoE layers
// are detected by the presence of "model.layers.{i}.mlp.gate.weight" in the
// tensor map. Dense layers use the standard gate_proj/up_proj/down_proj weights.
//
// The architecture is:
//
//	Embed*sqrt(d) -> [RMSNorm -> GQA -> PostNorm -> FusedAdd+RMSNorm -> MoE/FFN(GELU) -> FusedNorm+Add] x N -> RMSNorm -> LMHead(tied)
func buildGemma4MoEGraph(
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

	_, isGPUEngine := engine.(compute.WeightUploader)
	tw := func(name string, t *tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
		return transposeWeight2D(engine, isGPUEngine, name, t)
	}

	embedWeight, err := tl.Lookup("model.embed_tokens.weight")
	if err != nil {
		return nil, nil, err
	}

	finalNormWeight, err := tl.Lookup("model.norm.weight")
	if err != nil {
		return nil, nil, err
	}

	proxy := compute.NewEngineProxy[float32](engine)
	builder := graph.NewBuilder[float32](proxy)
	input := builder.Input([]int{1, 1})

	// Embedding lookup with sqrt(hidden_size) scaling (Gemma family convention).
	scale := float32(math.Sqrt(float64(cfg.HiddenSize)))
	embNode := newEmbeddingNode(proxy, embedWeight, scale)
	hidden := builder.AddNode(embNode, input)

	defaultHeadDim := cfg.HiddenSize / cfg.NumHeads
	if cfg.HeadDim > 0 {
		defaultHeadDim = cfg.HeadDim
	}

	for i := 0; i < cfg.NumLayers; i++ {
		prefix := fmt.Sprintf("model.layers.%d.", i)

		// Determine if this is a global or sliding layer.
		isGlobal := cfg.SlidingWindowPattern > 0 && (i+1)%cfg.SlidingWindowPattern == 0

		// Select per-layer attention configuration.
		var numKVHeads, headDim int
		var ropeTheta float64
		var partialRotaryFactor float32

		if isGlobal {
			numKVHeads = cfg.GlobalNumKVHeads
			if numKVHeads == 0 {
				numKVHeads = cfg.NumKVHeads
			}
			headDim = cfg.GlobalHeadDim
			if headDim == 0 {
				headDim = defaultHeadDim
			}
			ropeTheta = cfg.RopeTheta
			partialRotaryFactor = cfg.GlobalPartialRotaryFactor
		} else {
			numKVHeads = cfg.SlidingNumKVHeads
			if numKVHeads == 0 {
				numKVHeads = cfg.NumKVHeads
			}
			headDim = cfg.SlidingHeadDim
			if headDim == 0 {
				headDim = defaultHeadDim
			}
			if cfg.LocalRopeTheta > 0 {
				ropeTheta = cfg.LocalRopeTheta
			} else {
				ropeTheta = cfg.RopeTheta
			}
			// Sliding layers use full RoPE.
		}

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

		// --- Self-Attention (GQA) ---
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

		qWT, err := tw(prefix+"self_attn.q_proj.weight", qW)
		if err != nil {
			return nil, nil, err
		}
		kWT, err := tw(prefix+"self_attn.k_proj.weight", kW)
		if err != nil {
			return nil, nil, err
		}
		vWT, err := tw(prefix+"self_attn.v_proj.weight", vW)
		if err != nil {
			return nil, nil, err
		}
		oWT, err := tw(prefix+"self_attn.o_proj.weight", oW)
		if err != nil {
			return nil, nil, err
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

		// RoPE with per-layer theta and optional partial rotation.
		ropeOpts := []embeddings.RotaryPositionalEmbeddingOption{
			embeddings.WithRotaryBase(ropeTheta),
		}
		if partialRotaryFactor > 0 && partialRotaryFactor < 1 {
			ropeOpts = append(ropeOpts, embeddings.WithRotaryDimFraction(float64(partialRotaryFactor)))
		}
		rope, err := embeddings.NewRotaryPositionalEmbedding[float32](
			context.Background(), proxy, headDim, cfg.MaxSeqLen, ropeOpts...,
		)
		if err != nil {
			return nil, nil, fmt.Errorf("layer %d rope: %w", i, err)
		}

		gqa, err := attention.NewGroupedQueryAttentionFromParams[float32](
			proxy, ops, cfg.HiddenSize, cfg.NumHeads, numKVHeads,
			wq, wk, wv, wo, rope, headDim,
		)
		if err != nil {
			return nil, nil, fmt.Errorf("layer %d gqa: %w", i, err)
		}
		gqa.LayerIndex = i

		// Sliding window attention for non-global layers.
		if !isGlobal && cfg.SlidingWindow > 0 {
			gqa.SlidingWindowSize = cfg.SlidingWindow
		}

		// K=V optimization for global layers.
		if isGlobal && cfg.AttentionKEqV {
			gqa.SetKEqV(true)
		}

		// Q/K norms (always on for Gemma 4).
		qNormW, err := tl.Lookup(prefix + "self_attn.q_norm.weight")
		if err != nil {
			return nil, nil, err
		}
		qNorm, err := normalization.NewRMSNormFromParam[float32](
			proxy, ops, rmsEps, pw.Wrap(prefix+"self_attn.q_norm.weight", qNormW),
		)
		if err != nil {
			return nil, nil, err
		}
		kNormW, err := tl.Lookup(prefix + "self_attn.k_norm.weight")
		if err != nil {
			return nil, nil, err
		}
		kNorm, err := normalization.NewRMSNormFromParam[float32](
			proxy, ops, rmsEps, pw.Wrap(prefix+"self_attn.k_norm.weight", kNormW),
		)
		if err != nil {
			return nil, nil, err
		}
		gqa.SetQKNorms(qNorm, kNorm)
		gqa.SetQKNormWeights(qNormW, kNormW, rmsEps)

		// Merged QKV for Q4 quantized weights.
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
		// Merged QKV for Q4_K quantized weights.
		if qQ4K, ok := any(qW.GetStorage()).(*tensor.Q4KStorage); ok {
			if kQ4K, ok := any(kW.GetStorage()).(*tensor.Q4KStorage); ok {
				if vQ4K, ok := any(vW.GetStorage()).(*tensor.Q4KStorage); ok {
					mergedQ4K := tensor.MergeQ4KStorage(qQ4K, kQ4K, vQ4K)
					qShape := qW.Shape()
					kShape := kW.Shape()
					vShape := vW.Shape()
					nMerged := qShape[0] + kShape[0] + vShape[0]
					mergedT, mergeErr := tensor.NewWithStorage[float32]([]int{qShape[1], nMerged}, mergedQ4K)
					if mergeErr == nil {
						gqa.SetMergedQKV(mergedT, qShape[0], kShape[0], vShape[0])
					}
				}
			}
		}

		attnOut := builder.AddNode(gqa, normed)

		// --- Post-Attention Norm (Gemma 4 always has post-norms) ---
		postAttnNormW, err := tl.Lookup(prefix + "post_attention_layernorm.weight")
		if err != nil {
			return nil, nil, err
		}
		postAttnNorm, err := normalization.NewRMSNormFromParam[float32](
			proxy, ops, rmsEps, pw.Wrap(prefix+"post_attention_layernorm.weight", postAttnNormW),
		)
		if err != nil {
			return nil, nil, err
		}
		attnOut = builder.AddNode(postAttnNorm, attnOut)

		// --- Fused Residual Add + Pre-FFN LayerNorm ---
		preFfnNormW, err := tl.Lookup(prefix + "pre_feedforward_layernorm.weight")
		if err != nil {
			return nil, nil, err
		}
		fusedNode := &fusedAddRMSNormNode[float32]{engine: proxy, weight: preFfnNormW, eps: rmsEps}
		normed2 := builder.AddNode(fusedNode, attnOut, hidden)

		// --- FFN: MoE or Dense (GELU) ---
		var ffnOut graph.Node[float32]

		// Detect MoE layer: if router weight exists, this is a MoE layer.
		if _, isMoE := tensors[prefix+"mlp.gate.weight"]; isMoE {
			ffnOut, err = buildGemma4MoE(tensors, cfg, proxy, ops, builder, normed2, i, prefix, pw, tw)
			if err != nil {
				return nil, nil, fmt.Errorf("layer %d moe: %w", i, err)
			}
		} else {
			ffnOut, err = buildGemma4DenseFFN(tensors, cfg, proxy, ops, builder, normed2, prefix, pw, tw)
			if err != nil {
				return nil, nil, fmt.Errorf("layer %d ffn: %w", i, err)
			}
		}

		// --- Fused Post-FFN Norm + Residual Add ---
		postFfnNormW, err := tl.Lookup(prefix + "post_feedforward_layernorm.weight")
		if err != nil {
			return nil, nil, err
		}
		fusedNormAdd := &fusedNormAddNode[float32]{engine: proxy, weight: postFfnNormW, eps: rmsEps}
		resNode := &residualRefNode[float32]{source: fusedNode}
		residualRef := builder.AddNode(resNode)
		hidden = builder.AddNode(fusedNormAdd, ffnOut, residualRef)
	}

	// --- Final RMSNorm ---
	finalNorm, err := normalization.NewRMSNormFromParam[float32](
		proxy, ops, rmsEps, pw.Wrap("model.norm.weight", finalNormWeight),
	)
	if err != nil {
		return nil, nil, err
	}
	normedFinal := builder.AddNode(finalNorm, hidden)

	// --- LM Head (tied to embedding) ---
	lmHead := newLMHeadNode(proxy, embedWeight, cfg.LogitSoftcap)
	output := builder.AddNode(lmHead, normedFinal)

	g, err := builder.Build(output)
	if err != nil {
		return nil, nil, fmt.Errorf("build graph: %w", err)
	}

	g.SetEngineProxy(proxy)
	return g, embedWeight, nil
}

// buildGemma4MoE constructs the MoE sub-graph for a single Gemma 4 layer.
// Expert weights are stored per-expert (model.layers.{i}.mlp.experts.{j}.{gate,up,down}_proj.weight).
func buildGemma4MoE(
	tensors map[string]*tensor.TensorNumeric[float32],
	cfg *gguf.ModelConfig,
	proxy *compute.EngineProxy[float32],
	ops numeric.Float32Ops,
	builder *graph.Builder[float32],
	normed graph.Node[float32],
	layerIdx int,
	prefix string,
	pw paramWrapper[float32],
	tw func(string, *tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error),
) (graph.Node[float32], error) {
	numExperts := cfg.NumExperts
	topK := cfg.NumExpertsPerToken
	if topK == 0 {
		topK = 2
	}

	tl := newTensorLookup(tensors)

	// Load router weight: [num_experts, hidden_size].
	routerW, err := tl.Lookup(prefix + "mlp.gate.weight")
	if err != nil {
		return nil, err
	}

	// Build expert FFNs with per-expert weights.
	experts := make([]graph.Node[float32], numExperts)
	for j := 0; j < numExperts; j++ {
		expertPrefix := fmt.Sprintf("%smlp.experts.%d.", prefix, j)
		name := fmt.Sprintf("layer%d_expert%d", layerIdx, j)

		gateW, err := tl.Lookup(expertPrefix + "gate_proj.weight")
		if err != nil {
			return nil, fmt.Errorf("expert %d gate_proj: %w", j, err)
		}
		upW, err := tl.Lookup(expertPrefix + "up_proj.weight")
		if err != nil {
			return nil, fmt.Errorf("expert %d up_proj: %w", j, err)
		}
		downW, err := tl.Lookup(expertPrefix + "down_proj.weight")
		if err != nil {
			return nil, fmt.Errorf("expert %d down_proj: %w", j, err)
		}

		gateWT, err := tw(expertPrefix+"gate_proj.weight", gateW)
		if err != nil {
			return nil, err
		}
		upWT, err := tw(expertPrefix+"up_proj.weight", upW)
		if err != nil {
			return nil, err
		}
		downWT, err := tw(expertPrefix+"down_proj.weight", downW)
		if err != nil {
			return nil, err
		}

		gateParam := pw.Wrap(name+"_gate", gateWT)
		upParam := pw.Wrap(name+"_up", upWT)
		downParam := pw.Wrap(name+"_down", downWT)

		w1 := core.NewDenseFromParams(core.NewLinearFromParam(proxy, gateParam), nil)
		w2 := core.NewDenseFromParams(core.NewLinearFromParam(proxy, downParam), nil)
		w3 := core.NewDenseFromParams(core.NewLinearFromParam(proxy, upParam), nil)

		expertFFN, err := core.NewFFNFromDense[float32](name, proxy, ops, w1, w2, w3,
			core.WithGELU[float32](), core.WithFFNNoBias[float32]())
		if err != nil {
			return nil, fmt.Errorf("expert %d: %w", j, err)
		}
		experts[j] = expertFFN
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

// buildGemma4DenseFFN builds a standard GELU FFN for dense layers in the MoE model.
func buildGemma4DenseFFN(
	tensors map[string]*tensor.TensorNumeric[float32],
	cfg *gguf.ModelConfig,
	proxy *compute.EngineProxy[float32],
	ops numeric.Float32Ops,
	builder *graph.Builder[float32],
	normed graph.Node[float32],
	prefix string,
	pw paramWrapper[float32],
	tw func(string, *tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error),
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
		core.WithGELU[float32](),
		core.WithFFNNoBias[float32](),
	)
	if err != nil {
		return nil, err
	}

	gateWT, err := tw(prefix+"mlp.gate_proj.weight", gateW)
	if err != nil {
		return nil, err
	}
	upWT, err := tw(prefix+"mlp.up_proj.weight", upW)
	if err != nil {
		return nil, err
	}
	downWT, err := tw(prefix+"mlp.down_proj.weight", downW)
	if err != nil {
		return nil, err
	}

	ffnParams := ffn.Parameters()
	ffnParams[0].Value = gateWT // w1 = gate_proj
	ffnParams[1].Value = downWT // w2 = down_proj
	ffnParams[2].Value = upWT   // w3 = up_proj

	// Merged Gate+Up for Q4 quantized weights.
	if gateQ4, ok := any(gateW.GetStorage()).(*tensor.Q4Storage); ok {
		if upQ4, ok := any(upW.GetStorage()).(*tensor.Q4Storage); ok {
			mergedGateUpQ4 := tensor.MergeQ4Storage(gateQ4, upQ4)
			gateShape := gateW.Shape()
			upShape := upW.Shape()
			nMerged := gateShape[0] + upShape[0]
			mergedT, mergeErr := tensor.NewWithStorage[float32]([]int{gateShape[1], nMerged}, mergedGateUpQ4)
			if mergeErr == nil {
				ffn.SetMergedGateUp(mergedT, gateShape[0], upShape[0])
			}
		}
	}
	// Merged Gate+Up for Q4_K quantized weights.
	if gateQ4K, ok := any(gateW.GetStorage()).(*tensor.Q4KStorage); ok {
		if upQ4K, ok := any(upW.GetStorage()).(*tensor.Q4KStorage); ok {
			mergedGateUpQ4K := tensor.MergeQ4KStorage(gateQ4K, upQ4K)
			gateShape := gateW.Shape()
			upShape := upW.Shape()
			nMerged := gateShape[0] + upShape[0]
			mergedT, mergeErr := tensor.NewWithStorage[float32]([]int{gateShape[1], nMerged}, mergedGateUpQ4K)
			if mergeErr == nil {
				ffn.SetMergedGateUp(mergedT, gateShape[0], upShape[0])
			}
		}
	}

	return builder.AddNode(ffn, normed), nil
}
