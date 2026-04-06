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

// ResidualConfig controls the residual connection strategy used by
// architecture graph builders. The default mode ("standard" or "") preserves
// existing behaviour. "attnres" and "block_attnres" enable attention-weighted
// residual connections from the layers/residual package (arXiv:2603.15031).
//
// # GGUF metadata convention
//
// Models opt into attention residuals via two GGUF general-metadata keys:
//
//   - general.residual_mode (string): one of "standard" (default), "attnres",
//     or "block_attnres". When absent or empty, standard additive residuals
//     are used and no extra memory is allocated.
//
//   - general.attnres_blocks (uint32): number of blocks for the block_attnres
//     variant. Ignored when residual_mode is not "block_attnres". Defaults to
//     8 when unset, which recovers most of the benefit of full AttnRes.
//
// These keys follow the GGUF general metadata namespace convention
// (see https://github.com/ggerganov/ggml/blob/master/docs/gguf.md).
// [ResidualConfigFromGGUF] parses these values into a ResidualConfig.
type ResidualConfig struct {
	Mode      string // "standard" (default), "attnres", or "block_attnres"
	NumBlocks int    // block count for "block_attnres" mode (default 8)
}

// DefaultResidualConfig returns a ResidualConfig with standard (no-op) residuals.
func DefaultResidualConfig() ResidualConfig {
	return ResidualConfig{Mode: "standard"}
}

// ResidualConfigFromGGUF builds a ResidualConfig from GGUF model metadata.
// Missing keys produce the backward-compatible "standard" default.
func ResidualConfigFromGGUF(mode string, numBlocks int) ResidualConfig {
	cfg := DefaultResidualConfig()
	if mode != "" {
		cfg.Mode = mode
	}
	if numBlocks > 0 {
		cfg.NumBlocks = numBlocks
	}
	if cfg.Mode == "block_attnres" && cfg.NumBlocks == 0 {
		cfg.NumBlocks = 8
	}
	return cfg
}

// BuildResidualConnection returns a residual handler appropriate for the given
// config. For "standard" mode (the default), it returns nil — callers should
// fall through to existing residual-add logic. For "attnres" and
// "block_attnres" modes, it returns a placeholder (nil for now); the actual
// implementation will be wired once layers/residual/ ships AttnRes types.
func BuildResidualConnection[T tensor.Numeric](config ResidualConfig, engine compute.Engine[T]) any {
	switch config.Mode {
	case "", "standard":
		return nil
	case "attnres", "block_attnres":
		// Placeholder: actual AttnRes wiring will be added when models ship
		// with AttnRes GGUF metadata and layers/residual/ is integrated.
		return nil
	default:
		return nil
	}
}

// transformerGraphOpts configures architecture-specific differences.
type transformerGraphOpts struct {
	embedScale          float32        // multiply embeddings by this factor (0 = no scaling)
	postNorm            bool           // if true, apply post-attention and post-FFN norms (Gemma 3)
	qkNorm              bool           // if true, apply RMSNorm to Q/K after projection (Gemma 3)
	logitSoftcap        float32        // if > 0, apply logit softcapping: cap * tanh(logit/cap)
	slidingWindowSize   int            // if > 0, apply causal sliding window attention mask
	attnBias            bool           // if true, add bias to Q/K/V projections (Qwen 2)
	partialRotaryFactor float32        // fraction of head dims to apply RoPE (0 or 1 = full RoPE)
	residual            ResidualConfig // residual connection strategy
	// penultimateNode will capture the hidden state at layer NumLayers-2
	// for EAGLE head training once graph-level intermediate output capture
	// is implemented. For now, EAGLE training uses synthetic pairs.
}

// buildTransformerGraph constructs a computation graph for a decoder-only
// transformer from pre-loaded GGUF tensors. Both Llama and Gemma share the
// same transformer body; they differ only in LM head weight tying and
// embedding scaling.
//
// The graph accepts token IDs as input [1, seqLen] and performs embedding
// lookup internally. lmHeadWeight is used for the final logit projection.
func buildTransformerGraph(
	tensors map[string]*tensor.TensorNumeric[float32],
	cfg *gguf.ModelConfig,
	engine compute.Engine[float32],
	embedWeight *tensor.TensorNumeric[float32],
	lmHeadWeight *tensor.TensorNumeric[float32],
	opts transformerGraphOpts,
) (*graph.Graph[float32], error) {
	ops := numeric.Float32Ops{}

	// Use model-specified RMS norm epsilon, defaulting to 1e-5.
	rmsEps := float32(1e-5)
	if cfg.RMSNormEps > 0 {
		rmsEps = cfg.RMSNormEps
	}

	// Use canonical helpers from builder_helpers.go.
	tl := newTensorLookup(tensors)
	pw := newParamWrapper[float32]()

	_, isGPUEngine := engine.(compute.WeightUploader)

	tw := func(name string, t *tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
		return transposeWeight2D(engine, isGPUEngine, name, t)
	}

	finalNormWeight, err := tl.Lookup("model.norm.weight")
	if err != nil {
		return nil, err
	}

	// Wrap engine in EngineProxy for tracing compiler support.
	proxy := compute.NewEngineProxy[float32](engine)

	builder := graph.NewBuilder[float32](proxy)
	// Input: token IDs as [1, seqLen].
	input := builder.Input([]int{1, 1})

	// Embedding lookup: token IDs -> [1, seqLen, hiddenSize].
	embNode := newEmbeddingNode(proxy, embedWeight, opts.embedScale)
	hidden := builder.AddNode(embNode, input)
	headDim := cfg.HiddenSize / cfg.NumHeads
	if cfg.HeadDim > 0 {
		headDim = cfg.HeadDim
	}

	for i := 0; i < cfg.NumLayers; i++ {
		prefix := fmt.Sprintf("model.layers.%d.", i)

		// --- Input LayerNorm ---
		inputNormW, err := tl.Lookup(prefix + "input_layernorm.weight")
		if err != nil {
			return nil, err
		}
		inputNorm, err := normalization.NewRMSNormFromParam[float32](
			proxy, ops, rmsEps, pw.Wrap(prefix+"input_layernorm.weight", inputNormW),
		)
		if err != nil {
			return nil, err
		}
		normed := builder.AddNode(inputNorm, hidden)

		// --- Self Attention ---
		// Check for TransMLA tensors: if present, use MLA instead of GQA.
		transMLAPrefix := fmt.Sprintf("transmla.%d.", i)
		hasTransMLA := tl.Has(transMLAPrefix + "wDKV")

		// Select RoPE base: global vs local based on layer pattern.
		ropeBase := cfg.RopeTheta
		if cfg.SlidingWindowPattern > 0 && cfg.LocalRopeTheta > 0 {
			isGlobal := (i+1)%cfg.SlidingWindowPattern == 0
			if !isGlobal {
				ropeBase = cfg.LocalRopeTheta
			}
		}
		ropeOpts := []embeddings.RotaryPositionalEmbeddingOption{
			embeddings.WithRotaryBase(ropeBase),
		}
		if opts.partialRotaryFactor > 0 && opts.partialRotaryFactor < 1 {
			ropeOpts = append(ropeOpts, embeddings.WithRotaryDimFraction(float64(opts.partialRotaryFactor)))
		}
		rope, err := embeddings.NewRotaryPositionalEmbedding[float32](
			context.Background(), proxy, headDim, cfg.MaxSeqLen, ropeOpts...,
		)
		if err != nil {
			return nil, fmt.Errorf("layer %d rope: %w", i, err)
		}

		var attnOut graph.Node[float32]
		if hasTransMLA && cfg.TransMLAKVLoraDim > 0 {
			// --- TransMLA path: use MultiHeadLatentAttention ---
			qW, qErr := tl.Lookup(prefix + "self_attn.q_proj.weight")
			if qErr != nil {
				return nil, qErr
			}
			oW, oErr := tl.Lookup(prefix + "self_attn.o_proj.weight")
			if oErr != nil {
				return nil, oErr
			}
			dkvW, dkvErr := tl.Lookup(transMLAPrefix + "wDKV")
			if dkvErr != nil {
				return nil, dkvErr
			}
			ukW, ukErr := tl.Lookup(transMLAPrefix + "wUK")
			if ukErr != nil {
				return nil, ukErr
			}
			uvW, uvErr := tl.Lookup(transMLAPrefix + "wUV")
			if uvErr != nil {
				return nil, uvErr
			}

			qWT, tErr := tw(prefix+"self_attn.q_proj.weight", qW)
			if tErr != nil {
				return nil, tErr
			}
			oWT, tErr := tw(prefix+"self_attn.o_proj.weight", oW)
			if tErr != nil {
				return nil, tErr
			}

			// TransMLA tensors from the converter:
			//   wDKV: [dModel, rank] — already [inputDim, outputDim], no transpose needed
			//   wUK:  [dK, rank]     — stored as [outputDim, inputDim], needs transpose to [rank, dK]
			//   wUV:  [dV, rank]     — stored as [outputDim, inputDim], needs transpose to [rank, dV]
			ukWT, tErr := tw(transMLAPrefix+"wUK", ukW)
			if tErr != nil {
				return nil, tErr
			}
			uvWT, tErr := tw(transMLAPrefix+"wUV", uvW)
			if tErr != nil {
				return nil, tErr
			}

			wQ := core.NewDenseFromParams(
				core.NewLinearFromParam(proxy, pw.Wrap(prefix+"self_attn.q_proj.weight", qWT)), nil,
			)
			wDKV := core.NewDenseFromParams(
				core.NewLinearFromParam(proxy, pw.Wrap(transMLAPrefix+"wDKV", dkvW)), nil,
			)
			wUK := core.NewDenseFromParams(
				core.NewLinearFromParam(proxy, pw.Wrap(transMLAPrefix+"wUK", ukWT)), nil,
			)
			wUV := core.NewDenseFromParams(
				core.NewLinearFromParam(proxy, pw.Wrap(transMLAPrefix+"wUV", uvWT)), nil,
			)
			wO := core.NewDenseFromParams(
				core.NewLinearFromParam(proxy, pw.Wrap(prefix+"self_attn.o_proj.weight", oWT)), nil,
			)

			mla := attention.NewMultiHeadLatentAttention[float32](
				proxy, ops, cfg.NumHeads, headDim, cfg.TransMLAKVLoraDim, 0,
				wQ, wDKV, wUK, wUV, wO, rope,
			)
			attnOut = builder.AddNode(mla, normed)
		} else {
			// --- Standard GQA path ---
			qW, qErr := tl.Lookup(prefix + "self_attn.q_proj.weight")
			if qErr != nil {
				return nil, qErr
			}
			kW, kErr := tl.Lookup(prefix + "self_attn.k_proj.weight")
			if kErr != nil {
				return nil, kErr
			}
			vW, vErr := tl.Lookup(prefix + "self_attn.v_proj.weight")
			if vErr != nil {
				return nil, vErr
			}
			oW, oErr := tl.Lookup(prefix + "self_attn.o_proj.weight")
			if oErr != nil {
				return nil, oErr
			}

			qWT, tErr := tw(prefix+"self_attn.q_proj.weight", qW)
			if tErr != nil {
				return nil, tErr
			}
			kWT, tErr := tw(prefix+"self_attn.k_proj.weight", kW)
			if tErr != nil {
				return nil, tErr
			}
			vWT, tErr := tw(prefix+"self_attn.v_proj.weight", vW)
			if tErr != nil {
				return nil, tErr
			}
			oWT, tErr := tw(prefix+"self_attn.o_proj.weight", oW)
			if tErr != nil {
				return nil, tErr
			}

			// Build Q/K/V/O Dense layers, optionally with attention bias (Qwen 2).
			var qBias, kBias, vBias *core.Bias[float32]
			if opts.attnBias {
				if qB, ok := tl.Optional(prefix + "self_attn.q_proj.bias"); ok {
					qBias = core.NewBiasFromParam(proxy, ops, pw.Wrap(prefix+"self_attn.q_proj.bias", qB))
				}
				if kB, ok := tl.Optional(prefix + "self_attn.k_proj.bias"); ok {
					kBias = core.NewBiasFromParam(proxy, ops, pw.Wrap(prefix+"self_attn.k_proj.bias", kB))
				}
				if vB, ok := tl.Optional(prefix + "self_attn.v_proj.bias"); ok {
					vBias = core.NewBiasFromParam(proxy, ops, pw.Wrap(prefix+"self_attn.v_proj.bias", vB))
				}
			}
			wq := core.NewDenseFromParams(
				core.NewLinearFromParam(proxy, pw.Wrap(prefix+"self_attn.q_proj.weight", qWT)),
				qBias,
			)
			wk := core.NewDenseFromParams(
				core.NewLinearFromParam(proxy, pw.Wrap(prefix+"self_attn.k_proj.weight", kWT)),
				kBias,
			)
			wv := core.NewDenseFromParams(
				core.NewLinearFromParam(proxy, pw.Wrap(prefix+"self_attn.v_proj.weight", vWT)),
				vBias,
			)
			wo := core.NewDenseFromParams(
				core.NewLinearFromParam(proxy, pw.Wrap(prefix+"self_attn.o_proj.weight", oWT)),
				nil,
			)

			gqa, gqaErr := attention.NewGroupedQueryAttentionFromParams[float32](
				proxy, ops, cfg.HiddenSize, cfg.NumHeads, cfg.NumKVHeads,
				wq, wk, wv, wo, rope, headDim,
			)
			if gqaErr != nil {
				return nil, fmt.Errorf("layer %d gqa: %w", i, gqaErr)
			}
			gqa.LayerIndex = i
			if opts.slidingWindowSize > 0 {
				gqa.SlidingWindowSize = opts.slidingWindowSize
			}

			// Create merged QKV weight for single-GEMV decode optimization.
			// Concatenates Q, K, V Q4 blocks row-wise so a single GEMV replaces
			// three separate projections during decode (seqLen=1).
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
			// Note: In Q4_K_M models, V may use Q6_K while Q/K use Q4_K. We only
			// merge when all three are Q4KStorage to keep the storage type uniform.
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

			// Set Q/K norms if enabled (Gemma 3).
			if opts.qkNorm {
				qNormW, lookupErr := tl.Lookup(prefix + "self_attn.q_norm.weight")
				if lookupErr != nil {
					return nil, lookupErr
				}
				qNorm, normErr := normalization.NewRMSNormFromParam[float32](
					proxy, ops, rmsEps, pw.Wrap(prefix+"self_attn.q_norm.weight", qNormW),
				)
				if normErr != nil {
					return nil, normErr
				}
				kNormW, lookupErr := tl.Lookup(prefix + "self_attn.k_norm.weight")
				if lookupErr != nil {
					return nil, lookupErr
				}
				kNorm, normErr := normalization.NewRMSNormFromParam[float32](
					proxy, ops, rmsEps, pw.Wrap(prefix+"self_attn.k_norm.weight", kNormW),
				)
				if normErr != nil {
					return nil, normErr
				}
				gqa.SetQKNorms(qNorm, kNorm)
				gqa.SetQKNormWeights(qNormW, kNormW, rmsEps)
			}

			attnOut = builder.AddNode(gqa, normed)
		}

		// --- Post-Attention Norm (Gemma 3: normalize before residual add) ---
		if opts.postNorm {
			postAttnNormW, lookupErr := tl.Lookup(prefix + "post_attention_layernorm.weight")
			if lookupErr != nil {
				return nil, lookupErr
			}
			postAttnNorm, normErr := normalization.NewRMSNormFromParam[float32](
				proxy, ops, rmsEps, pw.Wrap(prefix+"post_attention_layernorm.weight", postAttnNormW),
			)
			if normErr != nil {
				return nil, normErr
			}
			attnOut = builder.AddNode(postAttnNorm, attnOut)
		}

		// --- Fused Residual Add + Pre-FFN LayerNorm ---
		// Fuses Add(attnOut, hidden) + RMSNorm into a single GPU kernel launch,
		// saving one kernel launch per layer. The stored residual is reused by
		// residualAddNode below.
		var preFfnNormKey string
		if opts.postNorm {
			preFfnNormKey = prefix + "pre_feedforward_layernorm.weight"
		} else {
			preFfnNormKey = prefix + "post_attention_layernorm.weight"
		}
		postNormW, err := tl.Lookup(preFfnNormKey)
		if err != nil {
			return nil, err
		}
		fusedNode := &fusedAddRMSNormNode[float32]{engine: proxy, weight: postNormW, eps: rmsEps}
		normed2 := builder.AddNode(fusedNode, attnOut, hidden)

		// --- FFN (SwiGLU) ---
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

		// Create merged Gate+Up weight for single-GEMV decode optimization.
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

		ffnOut := builder.AddNode(ffn, normed2)

		// --- Fused Post-FFN Norm + Residual Add ---
		// When postNorm is enabled (Gemma 3), fuse RMSNorm(ffnOut) + Add(result, residual)
		// into a single kernel launch, replacing 2 separate launches.
		if opts.postNorm {
			postFfnNormW, lookupErr := tl.Lookup(prefix + "post_feedforward_layernorm.weight")
			if lookupErr != nil {
				return nil, lookupErr
			}
			fusedNormAdd := &fusedNormAddNode[float32]{engine: proxy, weight: postFfnNormW, eps: rmsEps}
			// residualAddNode retrieves the stored residual from fusedNode.
			// fusedNormAddNode needs the residual as a graph input, so use a
			// residualAddNode that just returns the residual without adding.
			resNode := &residualRefNode[float32]{source: fusedNode}
			residualRef := builder.AddNode(resNode)
			hidden = builder.AddNode(fusedNormAdd, ffnOut, residualRef)
		} else {
			// Non-postNorm path: just add ffnOut + residual.
			resAdd := &residualAddNode[float32]{engine: proxy, source: fusedNode}
			hidden = builder.AddNode(resAdd, ffnOut)
		}
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
	// Convert Q8 lmHead weight to Q4 for fast GEMV on both CPU and GPU.
	// CPU: Q4 enables fast NEON GEMV. GPU: Q4 reduces weight read from
	// 1.2 GB (F32) to 0.17 GB, using the optimized Q4 GEMV kernel.
	if s := lmHeadWeight.GetStorage(); s != nil {
		if qs, ok := any(s).(*tensor.Q8Storage); ok {
			f32 := make([]float32, qs.Len())
			qs.Dequantize(f32)
			q4 := tensor.QuantizeQ4(f32)
			lmHeadWeight, err = tensor.NewWithStorage[float32](lmHeadWeight.Shape(), q4)
			if err != nil {
				return nil, fmt.Errorf("transpose lm_head weight: %w", err)
			}
		}
	}
	lmHead := newLMHeadNode(proxy, lmHeadWeight, opts.logitSoftcap)
	output := builder.AddNode(lmHead, normedFinal)

	g, err := builder.Build(output)
	if err != nil {
		return nil, fmt.Errorf("build graph: %w", err)
	}

	g.SetEngineProxy(proxy)
	return g, nil
}
