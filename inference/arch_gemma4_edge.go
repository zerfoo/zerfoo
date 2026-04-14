package inference

import (
	"context"
	"fmt"
	"math"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/zerfoo/layers/activations"
	"github.com/zerfoo/zerfoo/layers/attention"
	"github.com/zerfoo/zerfoo/layers/core"
	"github.com/zerfoo/zerfoo/layers/embeddings"
	"github.com/zerfoo/zerfoo/layers/normalization"
	"github.com/zerfoo/zerfoo/model/gguf"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
)

// buildGemma4EdgeGraph constructs a computation graph for the Gemma 4 edge
// variants (E2B and E4B) from pre-loaded GGUF tensors. Per ADR-086 and ADR-087
// it implements the canonical HuggingFace transformers modeling_gemma4.py
// layout:
//
//   1. Per-Layer Embeddings (PLE). Shared `model.ple_embed_tokens.weight`
//      [vocab, numLayers*pleDim] and `model.ple_model_proj.weight`
//      [hidden, numLayers*pleDim] are computed once per forward, sliced per
//      layer, combined via the shared `model.ple_proj_norm.weight` [pleDim]
//      RMSNorm, and scaled by 1/sqrt(2). Per-block `input_gate`
//      (hidden->pleDim), `ple_layer_proj` (pleDim->hidden), `post_layernorm`
//      (hidden), and `layer_output_scale` [1] weights complete the sub-block.
//
//   2. Shared-KV attention. The last `KVSharedLayers` layers skip their own
//      wk/wv/k_norm and reuse the K/V activations from the nearest non-shared
//      layer of the same attention type (sliding vs global), wired through
//      `KVReuseNode` into a consumer GQA configured with
//      `SetExternalKV(true)`.
//
//   3. Dual-RoPE hybrid attention: global layers (every
//      `SlidingWindowPattern`-th) use `RopeTheta`; sliding layers use
//      `LocalRopeTheta` and a per-layer sliding window.
//
// Per-layer flow (canonical post-mapping names):
//
//	input_layernorm -> GQA (donor or shared-via-KVReuseNode)
//	  -> post_attention_layernorm
//	  -> fusedAddRMSNorm(pre_feedforward_layernorm, +pre-attn residual)
//	  -> FFN (GELU, intermediate = gate_proj.Shape()[0])
//	  -> fusedNormAdd(post_feedforward_layernorm, +pre-FFN residual)
//	  -> PLE block: input_gate -> GELU -> * per_layer_inputs[i]
//	       -> ple_layer_proj -> post_layernorm -> + residual
//	  -> layer_output_scale (scalar broadcast)
func buildGemma4EdgeGraph(
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

	pleDim := cfg.PLEHiddenSize
	if pleDim <= 0 {
		return nil, nil, fmt.Errorf("gemma4-edge: PLEHiddenSize must be positive (got %d)", pleDim)
	}

	pleEmbed, err := tl.Lookup("model.ple_embed_tokens.weight")
	if err != nil {
		return nil, nil, fmt.Errorf("gemma4-edge: %w", err)
	}
	pleModelProjRaw, err := tl.Lookup("model.ple_model_proj.weight")
	if err != nil {
		return nil, nil, fmt.Errorf("gemma4-edge: %w", err)
	}
	pleProjNormGain, err := tl.Lookup("model.ple_proj_norm.weight")
	if err != nil {
		return nil, nil, fmt.Errorf("gemma4-edge: %w", err)
	}
	pleModelProj, err := tw("model.ple_model_proj.weight", pleModelProjRaw)
	if err != nil {
		return nil, nil, err
	}

	proxy := compute.NewEngineProxy[float32](engine)
	builder := graph.NewBuilder[float32](proxy)
	input := builder.Input([]int{1, 1})

	// Main embedding: lookup and scale by sqrt(hidden_size) (Gemma convention).
	hiddenScale := float32(math.Sqrt(float64(cfg.HiddenSize)))
	embNode := newEmbeddingNode(proxy, embedWeight, hiddenScale)
	hidden := builder.AddNode(embNode, input)

	// PLE combined producer: caches token-PLE and model-projection tensors
	// for consumption by per-layer pleSliceNode. Forward returns the hidden
	// input unchanged so downstream nodes keep a well-formed data edge.
	pleProducer, err := newPLECombinedProducer[float32](proxy, pleEmbed, pleModelProj,
		cfg.NumLayers, pleDim, cfg.HiddenSize)
	if err != nil {
		return nil, nil, fmt.Errorf("gemma4-edge: %w", err)
	}
	hidden = builder.AddNode(pleProducer, input, hidden)

	defaultHeadDim := cfg.HiddenSize / cfg.NumHeads
	if cfg.HeadDim > 0 {
		defaultHeadDim = cfg.HeadDim
	}

	// Classify layers upfront for ResolveKVDonor.
	layerTypes := make([]LayerType, cfg.NumLayers)
	for i := 0; i < cfg.NumLayers; i++ {
		if cfg.SlidingWindowPattern > 0 && (i+1)%cfg.SlidingWindowPattern == 0 {
			layerTypes[i] = LayerTypeGlobal
		} else {
			layerTypes[i] = LayerTypeSliding
		}
	}
	firstSharedIdx := cfg.NumLayers - cfg.KVSharedLayers
	if firstSharedIdx < 0 {
		firstSharedIdx = 0
	}
	donorGQA := make(map[int]*attention.GroupedQueryAttention[float32], cfg.NumLayers)
	donorKVGeom := make(map[int][2]int, cfg.NumLayers) // donorIdx -> [numKVHeads, headDim]

	for i := 0; i < cfg.NumLayers; i++ {
		prefix := fmt.Sprintf("model.layers.%d.", i)
		isGlobal := layerTypes[i] == LayerTypeGlobal
		isShared := cfg.KVSharedLayers > 0 && i >= firstSharedIdx

		// Per-layer attention configuration.
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

		// --- Q / O projections (always present) ---
		qW, err := tl.Lookup(prefix + "self_attn.q_proj.weight")
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
		oWT, err := tw(prefix+"self_attn.o_proj.weight", oW)
		if err != nil {
			return nil, nil, err
		}
		wq := core.NewDenseFromParams(
			core.NewLinearFromParam(proxy, pw.Wrap(prefix+"self_attn.q_proj.weight", qWT)), nil,
		)
		wo := core.NewDenseFromParams(
			core.NewLinearFromParam(proxy, pw.Wrap(prefix+"self_attn.o_proj.weight", oWT)), nil,
		)

		// --- K / V projections: only for non-shared (donor) layers ---
		var wk, wv *core.Dense[float32]
		if !isShared {
			kW, err := tl.Lookup(prefix + "self_attn.k_proj.weight")
			if err != nil {
				return nil, nil, err
			}
			vW, err := tl.Lookup(prefix + "self_attn.v_proj.weight")
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
			wk = core.NewDenseFromParams(
				core.NewLinearFromParam(proxy, pw.Wrap(prefix+"self_attn.k_proj.weight", kWT)), nil,
			)
			wv = core.NewDenseFromParams(
				core.NewLinearFromParam(proxy, pw.Wrap(prefix+"self_attn.v_proj.weight", vWT)), nil,
			)
		}

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
		if !isGlobal && cfg.SlidingWindow > 0 {
			gqa.SlidingWindowSize = cfg.SlidingWindow
		}

		// Q norm is present on every layer; K norm only on donor layers.
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

		var attnOut graph.Node[float32]
		if isShared {
			gqa.SetExternalKV(true)
			gqa.SetQKNorms(qNorm, nil)
			gqa.SetQKNormWeights(qNormW, nil, rmsEps)

			donorIdx := ResolveKVDonor(i, firstSharedIdx, layerTypes)
			donor, ok := donorGQA[donorIdx]
			if !ok {
				return nil, nil, fmt.Errorf("layer %d: donor GQA for index %d not registered", i, donorIdx)
			}
			geom, ok := donorKVGeom[donorIdx]
			if !ok {
				return nil, nil, fmt.Errorf("layer %d: donor geometry for index %d not registered", i, donorIdx)
			}
			donorKVHeads, donorHeadDim := geom[0], geom[1]
			kReuse, err := NewKVReuseNode[float32](proxy, donor.KPort(), donorKVHeads, donorHeadDim, true)
			if err != nil {
				return nil, nil, fmt.Errorf("layer %d: K reuse: %w", i, err)
			}
			vReuse, err := NewKVReuseNode[float32](proxy, donor.VPort(), donorKVHeads, donorHeadDim, false)
			if err != nil {
				return nil, nil, fmt.Errorf("layer %d: V reuse: %w", i, err)
			}
			kIn := builder.AddNode(kReuse)
			vIn := builder.AddNode(vReuse)
			attnOut = builder.AddNode(gqa, normed, kIn, vIn)
		} else {
			if isGlobal && cfg.AttentionKEqV {
				gqa.SetKEqV(true)
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
			donorGQA[i] = gqa
			donorKVGeom[i] = [2]int{numKVHeads, headDim}
			attnOut = builder.AddNode(gqa, normed)
		}

		// --- Post-Attention Norm ---
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

		// --- Fused Residual Add + Pre-FFN LayerNorm (residual = pre-attn hidden) ---
		preFfnNormW, err := tl.Lookup(prefix + "pre_feedforward_layernorm.weight")
		if err != nil {
			return nil, nil, err
		}
		fusedPreFFN := &fusedAddRMSNormNode[float32]{engine: proxy, weight: preFfnNormW, eps: rmsEps}
		normed2 := builder.AddNode(fusedPreFFN, attnOut, hidden)

		// --- FFN (GELU) ---
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

		// Per-layer intermediate size from gate_proj ground truth (the GGUF
		// loader reverses GGUF's [cols, rows] to zerfoo [rows, cols], so
		// gate_proj.Shape()[0] is the intermediate dim). This correctly
		// handles HF/unsloth boundary mismatches in E2B packing.
		gateShape := gateW.Shape()
		if len(gateShape) != 2 {
			return nil, nil, fmt.Errorf("layer %d gate_proj rank %d, want 2", i, len(gateShape))
		}
		intermediateSize := gateShape[0]

		ffn, err := core.NewFFN[float32](
			prefix+"mlp", proxy, ops,
			cfg.HiddenSize, intermediateSize, cfg.HiddenSize,
			core.WithGELU[float32](),
			core.WithFFNNoBias[float32](),
		)
		if err != nil {
			return nil, nil, fmt.Errorf("layer %d ffn: %w", i, err)
		}

		gateWT, err := tw(prefix+"mlp.gate_proj.weight", gateW)
		if err != nil {
			return nil, nil, err
		}
		upWT, err := tw(prefix+"mlp.up_proj.weight", upW)
		if err != nil {
			return nil, nil, err
		}
		downWT, err := tw(prefix+"mlp.down_proj.weight", downW)
		if err != nil {
			return nil, nil, err
		}
		ffnParams := ffn.Parameters()
		ffnParams[0].Value = gateWT // w1 = gate_proj
		ffnParams[1].Value = downWT // w2 = down_proj
		ffnParams[2].Value = upWT   // w3 = up_proj

		ffnOut := builder.AddNode(ffn, normed2)

		// --- Fused Post-FFN Norm + Residual Add (residual from fusedPreFFN input) ---
		postFfnNormW, err := tl.Lookup(prefix + "post_feedforward_layernorm.weight")
		if err != nil {
			return nil, nil, err
		}
		fusedPostFFN := &fusedNormAddNode[float32]{engine: proxy, weight: postFfnNormW, eps: rmsEps}
		resNode := &residualRefNode[float32]{source: fusedPreFFN}
		residualRef := builder.AddNode(resNode)
		afterFFN := builder.AddNode(fusedPostFFN, ffnOut, residualRef)

		// --- PLE sub-block (after the standard FFN residual) ---
		inpGateW, err := tl.Lookup(prefix + "input_gate.weight")
		if err != nil {
			return nil, nil, fmt.Errorf("layer %d: %w", i, err)
		}
		pleLayerProjW, err := tl.Lookup(prefix + "ple_layer_proj.weight")
		if err != nil {
			return nil, nil, fmt.Errorf("layer %d: %w", i, err)
		}
		postLayerNormW, err := tl.Lookup(prefix + "post_layernorm.weight")
		if err != nil {
			return nil, nil, fmt.Errorf("layer %d: %w", i, err)
		}
		layerOutputScaleW, err := tl.Lookup(prefix + "layer_output_scale.weight")
		if err != nil {
			return nil, nil, fmt.Errorf("layer %d: %w", i, err)
		}

		inpGateWT, err := tw(prefix+"input_gate.weight", inpGateW)
		if err != nil {
			return nil, nil, err
		}
		pleLayerProjWT, err := tw(prefix+"ple_layer_proj.weight", pleLayerProjW)
		if err != nil {
			return nil, nil, err
		}

		inpGate := core.NewDenseFromParams(
			core.NewLinearFromParam(proxy, pw.Wrap(prefix+"input_gate.weight", inpGateWT)), nil,
		)
		pleLayerProj := core.NewDenseFromParams(
			core.NewLinearFromParam(proxy, pw.Wrap(prefix+"ple_layer_proj.weight", pleLayerProjWT)), nil,
		)

		// gelu(input_gate(afterFFN)) -- produces [B, S, pleDim]
		gateOut := builder.AddNode(inpGate, afterFFN)
		gelu := activations.NewGelu[float32](proxy, ops)
		gateActivated := builder.AddNode(gelu, gateOut)

		// Per-layer PLE slice: [B, S, pleDim], combined per HF line 1693-1696.
		sliceNode, err := newPLESliceNode[float32](proxy, pleProducer, pleProjNormGain, rmsEps, i)
		if err != nil {
			return nil, nil, fmt.Errorf("layer %d: pleSlice: %w", i, err)
		}
		pleSlice := builder.AddNode(sliceNode)

		// Elementwise multiply gated hidden by per-layer inputs.
		mulGatePLE := &elementwiseMulNode[float32]{engine: proxy}
		gated := builder.AddNode(mulGatePLE, gateActivated, pleSlice)

		// Project back to hidden and apply post_layernorm.
		projected := builder.AddNode(pleLayerProj, gated)
		postLayerNorm, err := normalization.NewRMSNormFromParam[float32](
			proxy, ops, rmsEps, pw.Wrap(prefix+"post_layernorm.weight", postLayerNormW),
		)
		if err != nil {
			return nil, nil, err
		}
		postNormed := builder.AddNode(postLayerNorm, projected)

		// Add PLE sub-block output to the FFN residual.
		addRes := &elementwiseAddNode[float32]{engine: proxy}
		combined := builder.AddNode(addRes, afterFFN, postNormed)

		// --- Layer output scale (scalar broadcast) ---
		scaleNode, err := newLayerOutputScaleNode[float32](proxy, layerOutputScaleW)
		if err != nil {
			return nil, nil, fmt.Errorf("layer %d: %w", i, err)
		}
		hidden = builder.AddNode(scaleNode, combined)
	}

	// --- Final RMSNorm ---
	finalNorm, err := normalization.NewRMSNormFromParam[float32](
		proxy, ops, rmsEps, pw.Wrap("model.norm.weight", finalNormWeight),
	)
	if err != nil {
		return nil, nil, err
	}
	normedFinal := builder.AddNode(finalNorm, hidden)

	// --- LM Head (tied to embedding, optional softcap) ---
	lmHead := newLMHeadNode(proxy, embedWeight, cfg.LogitSoftcap)
	output := builder.AddNode(lmHead, normedFinal)

	g, err := builder.Build(output)
	if err != nil {
		return nil, nil, fmt.Errorf("build graph: %w", err)
	}
	g.SetEngineProxy(proxy)
	return g, embedWeight, nil
}
