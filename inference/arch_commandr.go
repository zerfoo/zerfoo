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
	RegisterArchitecture("command-r", buildCommandRGraph)
}

// buildCommandRGraph constructs a computation graph for the Command R architecture
// from pre-loaded GGUF tensors. It returns the graph and the embedding table
// tensor (needed by the generator for token lookup).
//
// Command R is a decoder-only transformer with the following characteristics:
//   - Standard GQA (grouped query attention) with RoPE positional encoding
//   - SwiGLU feed-forward network
//   - LayerNorm (NOT RMSNorm) applied before attention and FFN
//   - 128K context window via extended RoPE
//   - No tied embeddings (separate lm_head.weight)
//
// The graph is:
//
//	Embed -> [LayerNorm -> GQA(RoPE) -> ResAdd -> LayerNorm -> FFN(SwiGLU) -> ResAdd] x N -> LayerNorm -> LMHead
func buildCommandRGraph(
	tensors map[string]*tensor.TensorNumeric[float32],
	cfg *gguf.ModelConfig,
	engine compute.Engine[float32],
) (*graph.Graph[float32], *tensor.TensorNumeric[float32], error) {
	ops := numeric.Float32Ops{}

	const layerNormEps = float32(1e-5)

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

	ropeTheta := cfg.RopeTheta
	if ropeTheta == 0 {
		ropeTheta = 10000.0
	}

	for i := 0; i < cfg.NumLayers; i++ {
		prefix := fmt.Sprintf("model.layers.%d.", i)

		// --- Pre-attention LayerNorm ---
		inputNormGamma, err := tl.Lookup(prefix + "input_layernorm.weight")
		if err != nil {
			return nil, nil, err
		}
		inputNormBeta, err := tl.Lookup(prefix + "input_layernorm.bias")
		if err != nil {
			return nil, nil, err
		}
		inputNorm, err := normalization.NewLayerNormalization[float32](
			proxy, cfg.HiddenSize,
			normalization.WithLayerNormEpsilon[float32](layerNormEps),
		)
		if err != nil {
			return nil, nil, fmt.Errorf("layer %d input_layernorm: %w", i, err)
		}
		{
			p := inputNorm.Parameters()
			p[0].Value = inputNormGamma
			p[1].Value = inputNormBeta
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

		rope, err := embeddings.NewRotaryPositionalEmbedding[float32](
			context.Background(), proxy, headDim, cfg.MaxSeqLen,
			embeddings.WithRotaryBase(ropeTheta),
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

		// Merged QKV for single-GEMV decode optimization.
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

		// --- Residual add after attention ---
		resAdd1 := &elementwiseAddNode[float32]{engine: proxy}
		hidden = builder.AddNode(resAdd1, attnOut, hidden)

		// --- Pre-FFN LayerNorm ---
		postNormGamma, err := tl.Lookup(prefix + "post_attention_layernorm.weight")
		if err != nil {
			return nil, nil, err
		}
		postNormBeta, err := tl.Lookup(prefix + "post_attention_layernorm.bias")
		if err != nil {
			return nil, nil, err
		}
		postNorm, err := normalization.NewLayerNormalization[float32](
			proxy, cfg.HiddenSize,
			normalization.WithLayerNormEpsilon[float32](layerNormEps),
		)
		if err != nil {
			return nil, nil, fmt.Errorf("layer %d post_attention_layernorm: %w", i, err)
		}
		{
			p := postNorm.Parameters()
			p[0].Value = postNormGamma
			p[1].Value = postNormBeta
		}
		normed2 := builder.AddNode(postNorm, hidden)

		// --- FFN (SwiGLU) ---
		gateW, err := tl.Lookup(prefix + "mlp.gate_proj.weight")
		if err != nil {
			return nil, nil, err
		}
		upW, err := tl.Lookup(prefix + "mlp.up_proj.weight")
		if err != nil {
			return nil, nil, err
		}
		downW, err := tl.Lookup(prefix + "mlp.down_proj.weight")
		if err != nil {
			return nil, nil, err
		}

		ffn, err := core.NewFFN[float32](
			prefix+"mlp", proxy, ops,
			cfg.HiddenSize, cfg.IntermediateSize, cfg.HiddenSize,
			core.WithSwiGLU[float32](),
			core.WithFFNNoBias[float32](),
		)
		if err != nil {
			return nil, nil, fmt.Errorf("layer %d ffn: %w", i, err)
		}

		gateWT, err := transposeWeight(prefix+"mlp.gate_proj.weight", gateW)
		if err != nil {
			return nil, nil, err
		}
		upWT, err := transposeWeight(prefix+"mlp.up_proj.weight", upW)
		if err != nil {
			return nil, nil, err
		}
		downWT, err := transposeWeight(prefix+"mlp.down_proj.weight", downW)
		if err != nil {
			return nil, nil, err
		}

		ffnParams := ffn.Parameters()
		ffnParams[0].Value = gateWT
		ffnParams[1].Value = downWT
		ffnParams[2].Value = upWT

		// Merged gate+up for single-GEMV decode optimization.
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

		ffnOut := builder.AddNode(ffn, normed2)

		// --- Residual add after FFN ---
		resAdd2 := &elementwiseAddNode[float32]{engine: proxy}
		hidden = builder.AddNode(resAdd2, ffnOut, hidden)
	}

	// --- Final LayerNorm ---
	finalNormGamma, err := tl.Lookup("model.norm.weight")
	if err != nil {
		return nil, nil, err
	}
	finalNormBeta, err := tl.Lookup("model.norm.bias")
	if err != nil {
		return nil, nil, err
	}
	finalNorm, err := normalization.NewLayerNormalization[float32](
		proxy, cfg.HiddenSize,
		normalization.WithLayerNormEpsilon[float32](layerNormEps),
	)
	if err != nil {
		return nil, nil, err
	}
	{
		p := finalNorm.Parameters()
		p[0].Value = finalNormGamma
		p[1].Value = finalNormBeta
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

