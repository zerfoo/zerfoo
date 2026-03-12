package inference

import (
	"context"
	"fmt"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/graph"
	"github.com/zerfoo/zerfoo/layers/attention"
	"github.com/zerfoo/zerfoo/layers/core"
	"github.com/zerfoo/zerfoo/layers/embeddings"
	"github.com/zerfoo/zerfoo/layers/normalization"
	"github.com/zerfoo/zerfoo/model/gguf"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
)

// transformerGraphOpts configures architecture-specific differences.
type transformerGraphOpts struct {
	embedScale    float32 // multiply embeddings by this factor (0 = no scaling)
	postNorm      bool    // if true, apply post-attention and post-FFN norms (Gemma 3)
	qkNorm        bool    // if true, apply RMSNorm to Q/K after projection (Gemma 3)
	logitSoftcap  float32 // if > 0, apply logit softcapping: cap * tanh(logit/cap)
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

	lookup := func(name string) (*tensor.TensorNumeric[float32], error) {
		t, ok := tensors[name]
		if !ok {
			return nil, fmt.Errorf("missing tensor %q", name)
		}
		return t, nil
	}

	param := func(name string, t *tensor.TensorNumeric[float32]) *graph.Parameter[float32] {
		return &graph.Parameter[float32]{Name: name, Value: t}
	}

	_, isGPUEngine := engine.(compute.WeightUploader)

	transposeWeight := func(name string, t *tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
		s := t.GetStorage()
		// GPU path: use virtual transpose for Q4 weights so the Q4Storage
		// is preserved and the Q4 GEMV kernel can be used at inference time.
		// Q8 weights are still dequantized to F32 for cuBLAS SGEMM.
		if isGPUEngine {
			if _, ok := any(s).(*tensor.Q4Storage); ok {
				shape := t.Shape()
				if len(shape) == 2 {
					return tensor.NewWithStorage[float32]([]int{shape[1], shape[0]}, s)
				}
			}
			shape := t.Shape()
			if len(shape) == 2 {
				if qs, ok := any(s).(*tensor.Q8Storage); ok {
					f32 := make([]float32, qs.Len())
					qs.Dequantize(f32)
					rows, cols := shape[0], shape[1]
					transposed := make([]float32, len(f32))
					for r := range rows {
						for c := range cols {
							transposed[c*rows+r] = f32[r*cols+c]
						}
					}
					return tensor.New([]int{cols, rows}, transposed)
				}
			}
			tr, err := engine.Transpose(context.Background(), t, []int{1, 0})
			if err != nil {
				return nil, fmt.Errorf("transpose %s: %w", name, err)
			}
			return tr, nil
		}

		// CPU path: virtual transpose for quantized storage.
		if _, ok := any(s).(*tensor.Q4Storage); ok {
			shape := t.Shape()
			if len(shape) == 2 {
				return tensor.NewWithStorage[float32]([]int{shape[1], shape[0]}, s)
			}
		}
		if _, ok := any(s).(*tensor.Q8Storage); ok {
			shape := t.Shape()
			if len(shape) == 2 {
				return tensor.NewWithStorage[float32]([]int{shape[1], shape[0]}, s)
			}
		}
		tr, err := engine.Transpose(context.Background(), t, []int{1, 0})
		if err != nil {
			return nil, fmt.Errorf("transpose %s: %w", name, err)
		}
		return tr, nil
	}

	finalNormWeight, err := lookup("model.norm.weight")
	if err != nil {
		return nil, err
	}

	// Wrap engine in EngineProxy for tracing compiler support.
	proxy := compute.NewEngineProxy[float32](engine)

	builder := graph.NewBuilder[float32](proxy)
	// Input: token IDs as [1, seqLen].
	input := builder.Input([]int{1, 1})

	// Embedding lookup: token IDs -> [1, seqLen, hiddenSize].
	embNode := &embeddingLookupNode[float32]{
		engine: proxy,
		weight: embedWeight,
		scale:  opts.embedScale,
	}
	hidden := builder.AddNode(embNode, input)
	headDim := cfg.HiddenSize / cfg.NumHeads
	if cfg.HeadDim > 0 {
		headDim = cfg.HeadDim
	}

	for i := 0; i < cfg.NumLayers; i++ {
		prefix := fmt.Sprintf("model.layers.%d.", i)

		// --- Input LayerNorm ---
		inputNormW, err := lookup(prefix + "input_layernorm.weight")
		if err != nil {
			return nil, err
		}
		inputNorm, err := normalization.NewRMSNormFromParam[float32](
			proxy, ops, rmsEps, param(prefix+"input_layernorm.weight", inputNormW),
		)
		if err != nil {
			return nil, err
		}
		normed := builder.AddNode(inputNorm, hidden)

		// --- Self Attention (GQA) ---
		qW, err := lookup(prefix + "self_attn.q_proj.weight")
		if err != nil {
			return nil, err
		}
		kW, err := lookup(prefix + "self_attn.k_proj.weight")
		if err != nil {
			return nil, err
		}
		vW, err := lookup(prefix + "self_attn.v_proj.weight")
		if err != nil {
			return nil, err
		}
		oW, err := lookup(prefix + "self_attn.o_proj.weight")
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
			core.NewLinearFromParam(proxy, param(prefix+"self_attn.q_proj.weight", qWT)),
			nil,
		)
		wk := core.NewDenseFromParams(
			core.NewLinearFromParam(proxy, param(prefix+"self_attn.k_proj.weight", kWT)),
			nil,
		)
		wv := core.NewDenseFromParams(
			core.NewLinearFromParam(proxy, param(prefix+"self_attn.v_proj.weight", vWT)),
			nil,
		)
		wo := core.NewDenseFromParams(
			core.NewLinearFromParam(proxy, param(prefix+"self_attn.o_proj.weight", oWT)),
			nil,
		)

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
		rope, err := embeddings.NewRotaryPositionalEmbedding[float32](
			context.Background(), proxy, headDim, cfg.MaxSeqLen, ropeOpts...,
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

		// Set Q/K norms if enabled (Gemma 3).
		if opts.qkNorm {
			qNormW, lookupErr := lookup(prefix + "self_attn.q_norm.weight")
			if lookupErr != nil {
				return nil, lookupErr
			}
			qNorm, normErr := normalization.NewRMSNormFromParam[float32](
				proxy, ops, rmsEps, param(prefix+"self_attn.q_norm.weight", qNormW),
			)
			if normErr != nil {
				return nil, normErr
			}
			kNormW, lookupErr := lookup(prefix + "self_attn.k_norm.weight")
			if lookupErr != nil {
				return nil, lookupErr
			}
			kNorm, normErr := normalization.NewRMSNormFromParam[float32](
				proxy, ops, rmsEps, param(prefix+"self_attn.k_norm.weight", kNormW),
			)
			if normErr != nil {
				return nil, normErr
			}
			gqa.SetQKNorms(qNorm, kNorm)
		}

		attnOut := builder.AddNode(gqa, normed)

		// --- Post-Attention Norm (Gemma 3: normalize before residual add) ---
		if opts.postNorm {
			postAttnNormW, lookupErr := lookup(prefix + "post_attention_layernorm.weight")
			if lookupErr != nil {
				return nil, lookupErr
			}
			postAttnNorm, normErr := normalization.NewRMSNormFromParam[float32](
				proxy, ops, rmsEps, param(prefix+"post_attention_layernorm.weight", postAttnNormW),
			)
			if normErr != nil {
				return nil, normErr
			}
			attnOut = builder.AddNode(postAttnNorm, attnOut)
		}

		// --- Residual Add ---
		add1 := core.NewAdd[float32](proxy)
		residual1 := builder.AddNode(add1, attnOut, hidden)

		// --- Pre-FFN LayerNorm ---
		var preFfnNormKey string
		if opts.postNorm {
			preFfnNormKey = prefix + "pre_feedforward_layernorm.weight"
		} else {
			preFfnNormKey = prefix + "post_attention_layernorm.weight"
		}
		postNormW, err := lookup(preFfnNormKey)
		if err != nil {
			return nil, err
		}
		postNorm, err := normalization.NewRMSNormFromParam[float32](
			proxy, ops, rmsEps, param(preFfnNormKey, postNormW),
		)
		if err != nil {
			return nil, err
		}
		normed2 := builder.AddNode(postNorm, residual1)

		// --- FFN (SwiGLU) ---
		gateW, err := lookup(prefix + "mlp.gate_proj.weight")
		if err != nil {
			return nil, err
		}
		upW, err := lookup(prefix + "mlp.up_proj.weight")
		if err != nil {
			return nil, err
		}
		downW, err := lookup(prefix + "mlp.down_proj.weight")
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

		ffnOut := builder.AddNode(ffn, normed2)

		// --- Post-FFN Norm (Gemma 3: normalize before residual add) ---
		if opts.postNorm {
			postFfnNormW, lookupErr := lookup(prefix + "post_feedforward_layernorm.weight")
			if lookupErr != nil {
				return nil, lookupErr
			}
			postFfnNorm, normErr := normalization.NewRMSNormFromParam[float32](
				proxy, ops, rmsEps, param(prefix+"post_feedforward_layernorm.weight", postFfnNormW),
			)
			if normErr != nil {
				return nil, normErr
			}
			ffnOut = builder.AddNode(postFfnNorm, ffnOut)
		}

		// --- Residual Add ---
		add2 := core.NewAdd[float32](proxy)
		hidden = builder.AddNode(add2, ffnOut, residual1)
	}

	// --- Final RMSNorm ---
	finalNorm, err := normalization.NewRMSNormFromParam[float32](
		proxy, ops, rmsEps, param("model.norm.weight", finalNormWeight),
	)
	if err != nil {
		return nil, err
	}
	normedFinal := builder.AddNode(finalNorm, hidden)

	// --- LM Head ---
	// On CPU: convert Q8 lmHead weight to Q4 for fast NEON GEMV.
	// On GPU: skip conversion; UploadWeights will dequantize Q8 to F32
	// for cuBLAS SGEMM.
	if _, isGPU := engine.(compute.WeightUploader); !isGPU {
		if s := lmHeadWeight.GetStorage(); s != nil {
			switch qs := any(s).(type) {
			case *tensor.Q8Storage:
				f32 := make([]float32, qs.Len())
				qs.Dequantize(f32)
				q4 := tensor.QuantizeQ4(f32)
				lmHeadWeight, _ = tensor.NewWithStorage[float32](lmHeadWeight.Shape(), q4)
			}
		}
	}
	lmHead := &lmHeadNode[float32]{engine: proxy, weight: lmHeadWeight, softcapVal: opts.logitSoftcap}
	output := builder.AddNode(lmHead, normedFinal)

	g, err := builder.Build(output)
	if err != nil {
		return nil, fmt.Errorf("build graph: %w", err)
	}

	g.SetEngineProxy(proxy)
	return g, nil
}
