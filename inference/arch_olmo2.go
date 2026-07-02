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

// buildOLMo2Graph constructs a computation graph for the OLMo2 architecture
// (AI2's Open Language Model 2) from pre-loaded GGUF tensors.
//
// OLMo2 differs from Llama in norm placement:
//   - No pre-attention or pre-FFN norms (no input_layernorm).
//   - Post-attention norm: RMSNorm applied to attention output before residual add.
//   - Post-FFN norm: RMSNorm applied to FFN output before residual add.
//   - QK norm: RMSNorm applied to Q and K projections inside attention.
//
// The OLMo2 layer structure is:
//
//	Embed -> [GQA(QKNorm) -> PostAttnNorm -> Add -> FFN(SiLU-gate) -> PostFFNNorm -> Add] x N -> RMSNorm -> LMHead
func buildOLMo2Graph(
	tensors map[string]*tensor.TensorNumeric[float32],
	cfg *gguf.ModelConfig,
	engine compute.Engine[float32],
) (*graph.Graph[float32], *tensor.TensorNumeric[float32], error) {
	tl := newTensorLookup(tensors)

	embedWeight, err := tl.Lookup("model.embed_tokens.weight")
	if err != nil {
		return nil, nil, err
	}

	// OLMo2 may tie lm_head to embedding weights.
	lmHeadWeight, ok := tl.Optional("lm_head.weight")
	if !ok {
		lmHeadWeight = embedWeight
	}

	g, err := buildOLMo2TransformerGraph(tensors, cfg, engine, embedWeight, lmHeadWeight)
	if err != nil {
		return nil, nil, err
	}

	return g, embedWeight, nil
}

// buildOLMo2TransformerGraph constructs the OLMo2-specific transformer graph
// with post-norm placement and QK norms.
func buildOLMo2TransformerGraph(
	tensors map[string]*tensor.TensorNumeric[float32],
	cfg *gguf.ModelConfig,
	engine compute.Engine[float32],
	embedWeight *tensor.TensorNumeric[float32],
	lmHeadWeight *tensor.TensorNumeric[float32],
) (*graph.Graph[float32], error) {
	ops := numeric.Float32Ops{}

	rmsEps := float32(1e-5)
	if cfg.RMSNormEps > 0 {
		rmsEps = cfg.RMSNormEps
	}

	tl := newTensorLookup(tensors)

	pw := newParamWrapper[float32]()

	_, isGPUEngine := engine.(compute.WeightUploader)

	transposeWeight := func(name string, t *tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
		return transposeWeight2D(engine, isGPUEngine, name, t)
	}

	finalNormWeight, err := tl.Lookup("model.norm.weight")
	if err != nil {
		return nil, err
	}

	proxy := compute.NewEngineProxy[float32](engine)
	builder := graph.NewBuilder[float32](proxy)
	input := builder.Input([]int{1, 1})

	// Embedding lookup: token IDs -> [1, seqLen, hiddenSize].
	embNode := &embeddingLookupNode[float32]{
		engine: proxy,
		weight: embedWeight,
	}
	hidden := builder.AddNode(embNode, input)

	headDim := cfg.HiddenSize / cfg.NumHeads
	if cfg.HeadDim > 0 {
		headDim = cfg.HeadDim
	}

	for i := 0; i < cfg.NumLayers; i++ {
		prefix := fmt.Sprintf("model.layers.%d.", i)

		// --- Self Attention (no pre-norm in OLMo2) ---
		qW, err := tl.Lookup(prefix + "self_attn.q_proj.weight")
		if err != nil {
			return nil, err
		}
		kW, err := tl.Lookup(prefix + "self_attn.k_proj.weight")
		if err != nil {
			return nil, err
		}
		vW, err := tl.Lookup(prefix + "self_attn.v_proj.weight")
		if err != nil {
			return nil, err
		}
		oW, err := tl.Lookup(prefix + "self_attn.o_proj.weight")
		if err != nil {
			return nil, err
		}

		qWT, err := transposeWeight(prefix+"self_attn.q_proj.weight", qW)
		if err != nil {
			return nil, err
		}
		kWT, err := transposeWeight(prefix+"self_attn.k_proj.weight", kW)
		if err != nil {
			return nil, err
		}
		vWT, err := transposeWeight(prefix+"self_attn.v_proj.weight", vW)
		if err != nil {
			return nil, err
		}
		oWT, err := transposeWeight(prefix+"self_attn.o_proj.weight", oW)
		if err != nil {
			return nil, err
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

		rope, err := embeddings.NewRotaryPositionalEmbedding[float32](
			context.Background(), proxy, headDim, cfg.MaxSeqLen,
			embeddings.WithRotaryBase(cfg.RopeTheta),
		)
		if err != nil {
			return nil, fmt.Errorf("layer %d rope: %w", i, err)
		}

		gqa, err := attention.NewGroupedQueryAttentionFromParams[float32](
			proxy, ops, cfg.HiddenSize, cfg.NumHeads, cfg.NumKVHeads,
			wq, wk, wv, wo, rope, headDim,
		)
		if err != nil {
			return nil, fmt.Errorf("layer %d gqa: %w", i, err)
		}
		gqa.LayerIndex = i

		// Merged QKV for Q4 storage.
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
		// Merged QKV for Q4_K storage.
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

		// QK norms.
		qNormW, err := tl.Lookup(prefix + "self_attn.q_norm.weight")
		if err != nil {
			return nil, err
		}
		qNorm, err := normalization.NewRMSNormFromParam[float32](
			proxy, ops, rmsEps, pw.Wrap(prefix+"self_attn.q_norm.weight", qNormW),
		)
		if err != nil {
			return nil, err
		}
		kNormW, err := tl.Lookup(prefix + "self_attn.k_norm.weight")
		if err != nil {
			return nil, err
		}
		kNorm, err := normalization.NewRMSNormFromParam[float32](
			proxy, ops, rmsEps, pw.Wrap(prefix+"self_attn.k_norm.weight", kNormW),
		)
		if err != nil {
			return nil, err
		}
		gqa.SetQKNorms(qNorm, kNorm)
		gqa.SetQKNormWeights(qNormW, kNormW, rmsEps)

		attnOut := builder.AddNode(gqa, hidden)

		// --- Post-Attention Norm: RMSNorm(attnOut) + residual ---
		postAttnNormW, err := tl.Lookup(prefix + "post_attention_layernorm.weight")
		if err != nil {
			return nil, err
		}
		postAttnFused := &fusedNormAddNode[float32]{engine: proxy, weight: postAttnNormW, eps: rmsEps}
		hidden = builder.AddNode(postAttnFused, attnOut, hidden)

		// --- FFN (SwiGLU, no pre-norm in OLMo2) ---
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
			return nil, fmt.Errorf("layer %d ffn: %w", i, err)
		}

		gateWT, err := transposeWeight(prefix+"mlp.gate_proj.weight", gateW)
		if err != nil {
			return nil, err
		}
		upWT, err := transposeWeight(prefix+"mlp.up_proj.weight", upW)
		if err != nil {
			return nil, err
		}
		downWT, err := transposeWeight(prefix+"mlp.down_proj.weight", downW)
		if err != nil {
			return nil, err
		}

		ffnParams := ffn.Parameters()
		ffnParams[0].Value = gateWT // w1 = gate_proj
		ffnParams[1].Value = downWT // w2 = down_proj
		ffnParams[2].Value = upWT   // w3 = up_proj

		// Merged Gate+Up for Q4 storage.
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
		// Merged Gate+Up for Q4_K storage.
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

		ffnOut := builder.AddNode(ffn, hidden)

		// --- Post-FFN Norm: RMSNorm(ffnOut) + residual ---
		postFfnNormW, err := tl.Lookup(prefix + "post_feedforward_layernorm.weight")
		if err != nil {
			return nil, err
		}
		postFfnFused := &fusedNormAddNode[float32]{engine: proxy, weight: postFfnNormW, eps: rmsEps}
		hidden = builder.AddNode(postFfnFused, ffnOut, hidden)
	}

	// --- Final RMSNorm ---
	finalNorm, err := normalization.NewRMSNormFromParam[float32](
		proxy, ops, rmsEps, pw.Wrap("model.norm.weight", finalNormWeight),
	)
	if err != nil {
		return nil, err
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
				return nil, fmt.Errorf("quantize lm_head weight: %w", err)
			}
		}
	}
	lmHead := newLMHeadNode(proxy, lmHeadWeight, 0)
	output := builder.AddNode(lmHead, normedFinal)

	g, err := builder.Build(output)
	if err != nil {
		return nil, fmt.Errorf("build graph: %w", err)
	}

	g.SetEngineProxy(proxy)
	return g, nil
}
