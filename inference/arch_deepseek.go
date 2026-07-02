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
	"github.com/zerfoo/ztensor/types"
)

// buildDeepSeekGraph constructs a computation graph for the DeepSeek V2/V3
// architecture from pre-loaded GGUF tensors. It returns the graph and the
// embedding table tensor.
//
// DeepSeek uses Multi-head Latent Attention (MLA) which compresses KV into
// a low-rank latent space, and Mixture of Experts (MoE) with shared + routed
// experts.
//
// The architecture is:
//
//	Embed -> [RMSNorm -> MLA -> Add -> RMSNorm -> MoE -> Add] x N -> RMSNorm -> LMHead
func buildDeepSeekGraph(
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

	kvLoraDim := cfg.KVLoRADim
	if kvLoraDim == 0 {
		kvLoraDim = headDim // fallback
	}

	ropeHeadDim := cfg.QKRopeHeadDim // 0 means full headDim (handled by MLA constructor)

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

		// --- MLA (Multi-head Latent Attention) ---
		kvAProjW, err := tl.Lookup(blkPrefix + "attn_kv_a_proj_with_mqa.weight")
		if err != nil {
			return nil, nil, err
		}
		kvBProjW, err := tl.Lookup(blkPrefix + "attn_kv_b_proj.weight")
		if err != nil {
			return nil, nil, err
		}
		qProjW, err := lookupMLAQProj(tensors, blkPrefix, prefix)
		if err != nil {
			return nil, nil, err
		}
		oProjW, err := lookupDeepSeekOProj(tensors, blkPrefix, prefix)
		if err != nil {
			return nil, nil, err
		}

		// Transpose weights for Dense layers.
		ctx := context.Background()
		kvAProjWT, err := engine.Transpose(ctx, kvAProjW, []int{1, 0})
		if err != nil {
			return nil, nil, fmt.Errorf("layer %d transpose kv_a_proj: %w", i, err)
		}
		kvBProjWT, err := engine.Transpose(ctx, kvBProjW, []int{1, 0})
		if err != nil {
			return nil, nil, fmt.Errorf("layer %d transpose kv_b_proj: %w", i, err)
		}
		qProjWT, err := engine.Transpose(ctx, qProjW, []int{1, 0})
		if err != nil {
			return nil, nil, fmt.Errorf("layer %d transpose q_proj: %w", i, err)
		}
		oProjWT, err := engine.Transpose(ctx, oProjW, []int{1, 0})
		if err != nil {
			return nil, nil, fmt.Errorf("layer %d transpose o_proj: %w", i, err)
		}

		// Build Dense layers for MLA projections.
		wQ := core.NewDenseFromParams(
			core.NewLinearFromParam(proxy, pw.Wrap("q_proj.weight", qProjWT)), nil,
		)
		wDKV := core.NewDenseFromParams(
			core.NewLinearFromParam(proxy, pw.Wrap("kv_a_proj.weight", kvAProjWT)), nil,
		)

		// Split B projection into separate K and V up-projections.
		kvBShape := kvBProjWT.Shape()
		kvBHalf := kvBShape[1] / 2
		kvBData := kvBProjWT.Data()
		ukData := make([]float32, kvBShape[0]*kvBHalf)
		uvData := make([]float32, kvBShape[0]*kvBHalf)
		for r := 0; r < kvBShape[0]; r++ {
			copy(ukData[r*kvBHalf:(r+1)*kvBHalf], kvBData[r*kvBShape[1]:r*kvBShape[1]+kvBHalf])
			copy(uvData[r*kvBHalf:(r+1)*kvBHalf], kvBData[r*kvBShape[1]+kvBHalf:(r+1)*kvBShape[1]])
		}
		ukWeight, err := tensor.New([]int{kvBShape[0], kvBHalf}, ukData)
		if err != nil {
			return nil, nil, fmt.Errorf("layer %d create uk weight: %w", i, err)
		}
		uvWeight, err := tensor.New([]int{kvBShape[0], kvBHalf}, uvData)
		if err != nil {
			return nil, nil, fmt.Errorf("layer %d create uv weight: %w", i, err)
		}
		wUK := core.NewDenseFromParams(
			core.NewLinearFromParam(proxy, pw.Wrap("kv_b_k_proj.weight", ukWeight)), nil,
		)
		wUV := core.NewDenseFromParams(
			core.NewLinearFromParam(proxy, pw.Wrap("kv_b_v_proj.weight", uvWeight)), nil,
		)
		wO := core.NewDenseFromParams(
			core.NewLinearFromParam(proxy, pw.Wrap("o_proj.weight", oProjWT)), nil,
		)

		// RoPE for MLA — applied only to ropeHeadDim dimensions.
		ropeDim := headDim
		if ropeHeadDim > 0 {
			ropeDim = ropeHeadDim
		}
		ropeOpts := []embeddings.RotaryPositionalEmbeddingOption{
			embeddings.WithRotaryBase(cfg.RopeTheta),
		}
		rope, err := embeddings.NewRotaryPositionalEmbedding[float32](
			ctx, proxy, ropeDim, cfg.MaxSeqLen, ropeOpts...,
		)
		if err != nil {
			return nil, nil, fmt.Errorf("layer %d rope: %w", i, err)
		}

		mla := attention.NewMultiHeadLatentAttention[float32](
			proxy, ops, cfg.NumHeads, headDim, kvLoraDim, ropeHeadDim,
			wQ, wDKV, wUK, wUV, wO, rope,
		)
		attnOut := builder.AddNode(mla, normed)

		// --- Fused Residual Add + Pre-FFN LayerNorm ---
		postNormW, err := tl.Lookup(prefix + "post_attention_layernorm.weight")
		if err != nil {
			return nil, nil, err
		}
		fusedNode := &fusedAddRMSNormNode[float32]{engine: proxy, weight: postNormW, eps: rmsEps}
		normed2 := builder.AddNode(fusedNode, attnOut, hidden)

		// --- MoE or standard FFN ---
		var ffnOut graph.Node[float32]
		if cfg.NumExperts > 0 {
			ffnOut, err = buildDeepSeekMoE(
				tensors, cfg, proxy, ops, builder, normed2, i, blkPrefix, prefix,
			)
			if err != nil {
				return nil, nil, fmt.Errorf("layer %d moe: %w", i, err)
			}
		} else {
			ffnOut, err = buildDeepSeekStandardFFN(tensors, cfg, proxy, ops, builder, normed2, prefix)
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

// lookupMLAQProj finds the query projection weight for DeepSeek MLA.
func lookupMLAQProj(
	tensors map[string]*tensor.TensorNumeric[float32],
	blkPrefix, layerPrefix string,
) (*tensor.TensorNumeric[float32], error) {
	if t, ok := tensors[layerPrefix+"self_attn.q_proj.weight"]; ok {
		return t, nil
	}
	if t, ok := tensors[blkPrefix+"attn_q_a_proj.weight"]; ok {
		return t, nil
	}
	if t, ok := tensors[blkPrefix+"attn_q.weight"]; ok {
		return t, nil
	}
	return nil, fmt.Errorf("missing query projection tensor for %s", blkPrefix)
}

// lookupDeepSeekOProj finds the output projection weight.
func lookupDeepSeekOProj(
	tensors map[string]*tensor.TensorNumeric[float32],
	blkPrefix, layerPrefix string,
) (*tensor.TensorNumeric[float32], error) {
	if t, ok := tensors[layerPrefix+"self_attn.o_proj.weight"]; ok {
		return t, nil
	}
	if t, ok := tensors[blkPrefix+"attn_output.weight"]; ok {
		return t, nil
	}
	return nil, fmt.Errorf("missing output projection tensor for %s", blkPrefix)
}

// buildDeepSeekMoE constructs the MoE sub-graph for a single DeepSeek layer.
func buildDeepSeekMoE(
	tensors map[string]*tensor.TensorNumeric[float32],
	cfg *gguf.ModelConfig,
	proxy *compute.EngineProxy[float32],
	ops numeric.Float32Ops,
	builder *graph.Builder[float32],
	normed graph.Node[float32],
	layerIdx int,
	blkPrefix, layerPrefix string,
) (graph.Node[float32], error) {
	numExperts := cfg.NumExperts
	topK := cfg.NumExpertsPerToken
	if topK == 0 {
		topK = 6
	}

	tl := newTensorLookup(tensors)

	// Load router weight.
	routerW, err := tl.Lookup(blkPrefix + "ffn_gate_inp.weight")
	if err != nil {
		return nil, err
	}

	// Load stacked expert weights.
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

	gate := core.NewMoEGate[float32](proxy, ops, topK)
	moe := core.NewMixtureOfExperts[float32](proxy, ops, gate, experts, numExperts, topK)

	// Build shared expert if present.
	if cfg.NumSharedExperts > 0 {
		sharedFFN, err := buildSharedExpertFFN(tensors, proxy, ops, blkPrefix, cfg)
		if err != nil {
			return nil, fmt.Errorf("shared expert: %w", err)
		}
		moe.SharedExpert = sharedFFN
	}

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

// buildDeepSeekStandardFFN builds a standard SwiGLU FFN for layers without MoE.
func buildDeepSeekStandardFFN(
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

// buildExpertFFN creates a single expert FFN from stacked expert weight tensors.
func buildExpertFFN(
	engine *compute.EngineProxy[float32],
	ops numeric.Float32Ops,
	gateExpsW, upExpsW, downExpsW *tensor.TensorNumeric[float32],
	expertIdx, numExperts, hiddenDim, intermediateDim int,
	name string,
) (*core.FFN[float32], error) {
	gateSlice, err := extractExpertSlice(gateExpsW, expertIdx, numExperts)
	if err != nil {
		return nil, fmt.Errorf("gate slice: %w", err)
	}
	upSlice, err := extractExpertSlice(upExpsW, expertIdx, numExperts)
	if err != nil {
		return nil, fmt.Errorf("up slice: %w", err)
	}
	downSlice, err := extractExpertSlice(downExpsW, expertIdx, numExperts)
	if err != nil {
		return nil, fmt.Errorf("down slice: %w", err)
	}

	// Build the FFN directly from pre-existing weight slices without going
	// through NewFFN, which calls NewLinear internally and allocates
	// inputDim*outputDim random float32 values per layer — only to be
	// overwritten immediately. For MoE models with 256 experts × 62 layers,
	// that wasted allocation is ~857 GB and OOM-kills the process.
	ctx := context.Background()
	gateWT, err := engine.Transpose(ctx, gateSlice, []int{1, 0})
	if err != nil {
		return nil, err
	}
	upWT, err := engine.Transpose(ctx, upSlice, []int{1, 0})
	if err != nil {
		return nil, err
	}
	downWT, err := engine.Transpose(ctx, downSlice, []int{1, 0})
	if err != nil {
		return nil, err
	}

	pw := newParamWrapper[float32]()
	gateParam := pw.Wrap(name+"_gate", gateWT)
	upParam := pw.Wrap(name+"_up", upWT)
	downParam := pw.Wrap(name+"_down", downWT)

	w1 := core.NewDenseFromParams(core.NewLinearFromParam(engine, gateParam), nil)
	w2 := core.NewDenseFromParams(core.NewLinearFromParam(engine, downParam), nil)
	w3 := core.NewDenseFromParams(core.NewLinearFromParam(engine, upParam), nil)

	return core.NewFFNFromDense[float32](name, engine, ops, w1, w2, w3,
		core.WithSwiGLU[float32](), core.WithFFNNoBias[float32]())
}

// extractExpertSlice extracts a single expert's weight from a stacked tensor.
// For MmapStorage tensors, it slices the raw bytes at quantization-block
// boundaries — no heap allocation and no dequantization. This is critical for
// models whose stacked expert tensors exceed available RAM: calling Data() on a
// Q4_K_M stacked tensor materializes all experts at once (e.g. ~4.8 GB per
// tensor per layer × 3 types × 62 layers = ~893 GB for MiniMax-M2).
func extractExpertSlice(
	stacked *tensor.TensorNumeric[float32],
	expertIdx, numExperts int,
) (*tensor.TensorNumeric[float32], error) {
	shape := stacked.Shape()

	var rows, cols, elemStart int
	switch len(shape) {
	case 3:
		rows = shape[1]
		cols = shape[2]
		elemStart = expertIdx * rows * cols
	case 2:
		totalRows := shape[0]
		cols = shape[1]
		rows = totalRows / numExperts
		elemStart = expertIdx * rows * cols
	default:
		return nil, fmt.Errorf("unexpected stacked tensor shape %v", shape)
	}

	elemEnd := elemStart + rows*cols
	outShape := []int{rows, cols}

	// Fast path: MmapStorage → zero-copy byte slice at block boundaries.
	if ms, ok := any(stacked.GetStorage()).(*tensor.MmapStorage); ok {
		sub, err := ms.SliceElements(elemStart, elemEnd)
		if err != nil {
			// Fall through to float32 path if alignment doesn't hold.
			goto f32path
		}
		return tensor.NewWithStorage[float32](outShape, sub)
	}

f32path:
	data := stacked.Data()
	sliceData := make([]float32, rows*cols)
	copy(sliceData, data[elemStart:elemEnd])
	return tensor.New(outShape, sliceData)
}

// buildSharedExpertFFN creates the shared expert FFN from GGUF tensors.
func buildSharedExpertFFN(
	tensors map[string]*tensor.TensorNumeric[float32],
	engine *compute.EngineProxy[float32],
	ops numeric.Float32Ops,
	blkPrefix string,
	cfg *gguf.ModelConfig,
) (*core.FFN[float32], error) {
	tl := newTensorLookup(tensors)

	gateW, err := tl.Lookup(blkPrefix + "ffn_shared_expert_gate.weight")
	if err != nil {
		return nil, err
	}
	upW, err := tl.Lookup(blkPrefix + "ffn_shared_expert_up.weight")
	if err != nil {
		return nil, err
	}
	downW, err := tl.Lookup(blkPrefix + "ffn_shared_expert_down.weight")
	if err != nil {
		return nil, err
	}

	ctx := context.Background()
	gateWT, err := engine.Transpose(ctx, gateW, []int{1, 0})
	if err != nil {
		return nil, err
	}
	upWT, err := engine.Transpose(ctx, upW, []int{1, 0})
	if err != nil {
		return nil, err
	}
	downWT, err := engine.Transpose(ctx, downW, []int{1, 0})
	if err != nil {
		return nil, err
	}

	ffn, err := core.NewFFN[float32](
		blkPrefix+"shared_expert", engine, ops,
		cfg.HiddenSize, cfg.IntermediateSize, cfg.HiddenSize,
		core.WithSwiGLU[float32](),
		core.WithFFNNoBias[float32](),
	)
	if err != nil {
		return nil, err
	}

	params := ffn.Parameters()
	params[0].Value = gateWT
	params[1].Value = downWT
	params[2].Value = upWT

	return ffn, nil
}

// deepSeekReshapeNode reshapes between [batch, seqLen, hidden] and [seqLen, hidden]
// for the MoE layer which expects 2D input.
type deepSeekReshapeNode[T tensor.Numeric] struct {
	engine  compute.Engine[T]
	flatten bool // true: 3D->2D, false: 2D->3D (uses shape from second input)
}

func (n *deepSeekReshapeNode[T]) OpType() string { return "DeepSeekReshape" }

func (n *deepSeekReshapeNode[T]) Attributes() map[string]any {
	return map[string]any{"flatten": n.flatten}
}

func (n *deepSeekReshapeNode[T]) Forward(ctx context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if n.flatten {
		shape := inputs[0].Shape()
		return n.engine.Reshape(ctx, inputs[0], []int{shape[0] * shape[1], shape[2]})
	}
	refShape := inputs[1].Shape()
	return n.engine.Reshape(ctx, inputs[0], refShape)
}

func (n *deepSeekReshapeNode[T]) Backward(_ context.Context, _ types.BackwardMode, _ *tensor.TensorNumeric[T], _ ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	return nil, nil
}

func (n *deepSeekReshapeNode[T]) Parameters() []*graph.Parameter[T] { return nil }
func (n *deepSeekReshapeNode[T]) OutputShape() []int                { return nil }

// deepSeekConstNode wraps a tensor as a constant graph node.
type deepSeekConstNode[T tensor.Numeric] struct {
	value *tensor.TensorNumeric[T]
}

func (n *deepSeekConstNode[T]) OpType() string                    { return "Constant" }
func (n *deepSeekConstNode[T]) Attributes() map[string]any        { return nil }
func (n *deepSeekConstNode[T]) Parameters() []*graph.Parameter[T] { return nil }
func (n *deepSeekConstNode[T]) OutputShape() []int                { return nil }

func (n *deepSeekConstNode[T]) Forward(_ context.Context, _ ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	return n.value, nil
}

func (n *deepSeekConstNode[T]) Backward(_ context.Context, _ types.BackwardMode, _ *tensor.TensorNumeric[T], _ ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	return nil, nil
}
