package inference

import (
	"context"
	"fmt"
	"math"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/zerfoo/layers/attention"
	"github.com/zerfoo/zerfoo/layers/core"
	"github.com/zerfoo/zerfoo/layers/embeddings"
	"github.com/zerfoo/zerfoo/layers/normalization"
	"github.com/zerfoo/zerfoo/layers/ssm"
	"github.com/zerfoo/zerfoo/model/gguf"
)

// NemotronHConfig holds Nemotron-H-specific hybrid model configuration.
type NemotronHConfig struct {
	NumLayers        int
	HiddenSize       int
	IntermediateSize int
	AttnHeads        int
	KVHeads          int
	SSMStateSize     int // SSM state dimension per head
	SSMConvKernel    int // SSM convolution kernel width (default 4)
	SSMNumHeads      int // SSM number of heads
	RMSEps           float32
	VocabSize        int
	MaxSeqLen        int
	RopeTheta        float64

	// MoE fields (only used by nemotron_h_moe).
	NumExperts         int
	NumExpertsPerToken int
	NumSharedExperts   int
}

// NemotronHConfigFromGGUF extracts Nemotron-H configuration from GGUF ModelConfig.
func NemotronHConfigFromGGUF(cfg *gguf.ModelConfig) NemotronHConfig {
	eps := cfg.RMSNormEps
	if eps == 0 {
		eps = 1e-5
	}
	ropeTheta := cfg.RopeTheta
	if ropeTheta == 0 {
		ropeTheta = 10000
	}
	ssmState := cfg.SSMStateSize
	if ssmState == 0 {
		ssmState = 16
	}
	ssmConv := cfg.SSMConvKernel
	if ssmConv == 0 {
		ssmConv = 4
	}
	ssmHeads := cfg.SSMNumHeads
	if ssmHeads == 0 {
		ssmHeads = 1
	}
	kvHeads := cfg.NumKVHeads
	if kvHeads == 0 {
		kvHeads = cfg.NumHeads
	}
	sharedExperts := cfg.NumSharedExperts
	if sharedExperts == 0 {
		sharedExperts = cfg.ExpertSharedCount
	}
	return NemotronHConfig{
		NumLayers:          cfg.NumLayers,
		HiddenSize:         cfg.HiddenSize,
		IntermediateSize:   cfg.IntermediateSize,
		AttnHeads:          cfg.NumHeads,
		KVHeads:            kvHeads,
		SSMStateSize:       ssmState,
		SSMConvKernel:      ssmConv,
		SSMNumHeads:        ssmHeads,
		RMSEps:             eps,
		VocabSize:          cfg.VocabSize,
		MaxSeqLen:          cfg.MaxSeqLen,
		RopeTheta:          ropeTheta,
		NumExperts:         cfg.NumExperts,
		NumExpertsPerToken: cfg.NumExpertsPerToken,
		NumSharedExperts:   sharedExperts,
	}
}

// nemotronHLayerType represents the type of a Nemotron-H layer.
type nemotronHLayerType int

const (
	nemotronHLayerMamba nemotronHLayerType = iota
	nemotronHLayerAttn
	nemotronHLayerFFN
	nemotronHLayerMoE
)

// detectNemotronHLayerType probes tensor names to determine the layer type
// for layer i.
//
// Detection order:
//   - blk.{i}.ssm_in.weight exists -> Mamba-2 (SSM) layer
//   - blk.{i}.attn_q.weight exists -> Attention layer
//   - blk.{i}.ffn_gate_inp.weight exists -> MoE layer
//   - blk.{i}.ffn_gate.weight exists -> Dense FFN layer
func detectNemotronHLayerType(tensors map[string]*tensor.TensorNumeric[float32], i int) nemotronHLayerType {
	tl := newTensorLookup(tensors)
	prefix := fmt.Sprintf("blk.%d.", i)
	if tl.Has(prefix + "ssm_in.weight") {
		return nemotronHLayerMamba
	}
	if tl.Has(prefix + "attn_q.weight") {
		return nemotronHLayerAttn
	}
	if tl.Has(prefix + "ffn_gate_inp.weight") {
		return nemotronHLayerMoE
	}
	return nemotronHLayerFFN
}

// buildNemotronHGraph constructs a computation graph for the Nemotron-H dense
// architecture from pre-loaded GGUF tensors.
//
// Architecture:
//
//	Embed -> [LayerType-dependent block] x N -> RMSNorm -> LMHead
//
// Layer types (detected by tensor name probing):
//   - Mamba: RMSNorm -> MIMOMambaBlock -> ResidualAdd
//   - Attention: RMSNorm -> GQA -> ResidualAdd -> RMSNorm -> SwiGLU FFN -> ResidualAdd
//   - Dense FFN: RMSNorm -> SwiGLU FFN -> ResidualAdd
func buildNemotronHGraph(
	tensors map[string]*tensor.TensorNumeric[float32],
	cfg *gguf.ModelConfig,
	engine compute.Engine[float32],
) (*graph.Graph[float32], *tensor.TensorNumeric[float32], error) {
	nc := NemotronHConfigFromGGUF(cfg)
	return BuildNemotronH(nc, tensors, engine, false)
}

// buildNemotronHMoEGraph constructs a computation graph for the Nemotron-H MoE
// architecture. This is identical to the dense variant but also supports MoE
// layers detected via blk.{i}.ffn_gate_inp.weight tensors.
func buildNemotronHMoEGraph(
	tensors map[string]*tensor.TensorNumeric[float32],
	cfg *gguf.ModelConfig,
	engine compute.Engine[float32],
) (*graph.Graph[float32], *tensor.TensorNumeric[float32], error) {
	nc := NemotronHConfigFromGGUF(cfg)
	return BuildNemotronH(nc, tensors, engine, true)
}

// BuildNemotronH constructs a computation graph for the Nemotron-H hybrid
// architecture. When moeEnabled is true, MoE layers are supported via tensor
// name probing.
//
// Expected GGUF tensor names:
//
//	Global:
//	  token_embd.weight, output_norm.weight, output.weight
//
//	Mamba layers (blk.{i}.):
//	  attn_norm.weight, ssm_in.weight, ssm_conv1d.weight, ssm_dt.weight,
//	  ssm_A.weight, ssm_D.weight, ssm_out.weight
//
//	Attention layers (blk.{i}.):
//	  attn_norm.weight, attn_q.weight, attn_k.weight, attn_v.weight,
//	  attn_output.weight, ffn_norm.weight, ffn_gate.weight, ffn_up.weight,
//	  ffn_down.weight
//
//	Dense FFN layers (blk.{i}.):
//	  attn_norm.weight, ffn_gate.weight, ffn_up.weight, ffn_down.weight
//
//	MoE layers (blk.{i}., moeEnabled only):
//	  attn_norm.weight, ffn_gate_inp.weight, ffn_gate_exps.weight,
//	  ffn_up_exps.weight, ffn_down_exps.weight
func BuildNemotronH(
	nc NemotronHConfig,
	tensors map[string]*tensor.TensorNumeric[float32],
	engine compute.Engine[float32],
	moeEnabled bool,
) (*graph.Graph[float32], *tensor.TensorNumeric[float32], error) {
	ops := numeric.Float32Ops{}
	tl := newTensorLookup(tensors)
	pw := newParamWrapper[float32]()

	embedWeight, err := tl.Lookup("token_embd.weight")
	if err != nil {
		return nil, nil, err
	}

	lmHeadWeight, ok := tl.Optional("output.weight")
	if !ok {
		lmHeadWeight = embedWeight
	}

	outputNormWeight, err := tl.Lookup("output_norm.weight")
	if err != nil {
		return nil, nil, err
	}

	proxy := compute.NewEngineProxy[float32](engine)
	builder := graph.NewBuilder[float32](proxy)
	input := builder.Input([]int{1, 1})

	embNode := newEmbeddingNode(proxy, embedWeight, 0)
	hidden := builder.AddNode(embNode, input)

	headDim := nc.HiddenSize / nc.AttnHeads
	if headDim == 0 {
		headDim = 1
	}

	// Derive SSM parameters.
	dInner := nc.IntermediateSize
	if dInner == 0 {
		dInner = nc.HiddenSize * 2
	}
	ssmHeads := nc.SSMNumHeads
	if ssmHeads == 0 {
		ssmHeads = 1
	}
	// Ensure dInner is divisible by ssmHeads; fall back to 1 head if not.
	if dInner%ssmHeads != 0 {
		ssmHeads = 1
	}
	ssmHeadDim := dInner / ssmHeads

	// Derive dt_rank from first SSM layer's ssm_dt.weight shape if available.
	dtRank := int(math.Ceil(float64(nc.HiddenSize) / 16))
	for i := 0; i < nc.NumLayers; i++ {
		prefix := fmt.Sprintf("blk.%d.", i)
		if dtW, exists := tl.Optional(prefix + "ssm_dt.weight"); exists {
			// ssm_dt.weight shape: [d_inner, dt_rank]
			dtRank = dtW.Shape()[1]
			break
		}
	}

	for i := 0; i < nc.NumLayers; i++ {
		prefix := fmt.Sprintf("blk.%d.", i)
		layerType := detectNemotronHLayerType(tensors, i)

		// Use attn_norm.weight as the pre-layer norm for all layer types.
		normName := prefix + "attn_norm.weight"
		normW, lErr := tl.Lookup(normName)
		if lErr != nil {
			return nil, nil, fmt.Errorf("layer %d: %w", i, lErr)
		}
		norm, nErr := normalization.NewRMSNormFromParam[float32](
			proxy, ops, nc.RMSEps, pw.Wrap(normName, normW),
		)
		if nErr != nil {
			return nil, nil, fmt.Errorf("layer %d norm: %w", i, nErr)
		}
		normed := builder.AddNode(norm, hidden)

		switch layerType {
		case nemotronHLayerMamba:
			// --- Mamba-2 SSM Block ---
			block, bErr := ssm.NewMIMOMambaBlock[float32](
				prefix+"mimo_block", proxy, ops,
				nc.HiddenSize, dInner, nc.SSMStateSize, dtRank, nc.SSMConvKernel, ssmHeads,
				ssm.WithMIMODiscretizationMode[float32](ssm.ZOH),
			)
			if bErr != nil {
				return nil, nil, fmt.Errorf("layer %d mimo block: %w", i, bErr)
			}

			if wErr := loadNemotronHSSMWeights(block, tensors, prefix, nc, dtRank, ssmHeadDim); wErr != nil {
				return nil, nil, fmt.Errorf("layer %d load ssm weights: %w", i, wErr)
			}

			mambaOut := builder.AddNode(block, normed)
			resAdd := &mambaResidualAddNode[float32]{engine: proxy}
			hidden = builder.AddNode(resAdd, mambaOut, hidden)

		case nemotronHLayerAttn:
			// --- Transformer Attention Block ---
			attnOut, aErr := buildNemotronHAttention(tensors, proxy, ops, builder, normed, nc, i, prefix, headDim)
			if aErr != nil {
				return nil, nil, fmt.Errorf("layer %d attention: %w", i, aErr)
			}

			resAdd := &mambaResidualAddNode[float32]{engine: proxy}
			hidden = builder.AddNode(resAdd, attnOut, hidden)

			// Pre-FFN RMSNorm.
			ffnNormW, lErr := tl.Lookup(prefix + "ffn_norm.weight")
			if lErr != nil {
				return nil, nil, fmt.Errorf("layer %d: %w", i, lErr)
			}
			ffnNorm, nErr := normalization.NewRMSNormFromParam[float32](
				proxy, ops, nc.RMSEps, pw.Wrap(prefix+"ffn_norm.weight", ffnNormW),
			)
			if nErr != nil {
				return nil, nil, fmt.Errorf("layer %d ffn norm: %w", i, nErr)
			}
			normed2 := builder.AddNode(ffnNorm, hidden)

			// SwiGLU FFN.
			ffnOut, fErr := buildNemotronHFFN(tensors, proxy, ops, builder, normed2, nc, i, prefix)
			if fErr != nil {
				return nil, nil, fmt.Errorf("layer %d ffn: %w", i, fErr)
			}

			resAdd2 := &mambaResidualAddNode[float32]{engine: proxy}
			hidden = builder.AddNode(resAdd2, ffnOut, hidden)

		case nemotronHLayerMoE:
			if !moeEnabled {
				return nil, nil, fmt.Errorf("layer %d has MoE tensors but MoE is not enabled (use nemotron_h_moe architecture)", i)
			}
			// --- MoE Block ---
			moeOut, mErr := buildNemotronHMoE(tensors, nc, proxy, ops, builder, normed, i, prefix)
			if mErr != nil {
				return nil, nil, fmt.Errorf("layer %d moe: %w", i, mErr)
			}

			resAdd := &mambaResidualAddNode[float32]{engine: proxy}
			hidden = builder.AddNode(resAdd, moeOut, hidden)

		case nemotronHLayerFFN:
			// --- Dense FFN Block ---
			ffnOut, fErr := buildNemotronHFFN(tensors, proxy, ops, builder, normed, nc, i, prefix)
			if fErr != nil {
				return nil, nil, fmt.Errorf("layer %d ffn: %w", i, fErr)
			}

			resAdd := &mambaResidualAddNode[float32]{engine: proxy}
			hidden = builder.AddNode(resAdd, ffnOut, hidden)
		}
	}

	// Final RMSNorm.
	finalNorm, err := normalization.NewRMSNormFromParam[float32](
		proxy, ops, nc.RMSEps, pw.Wrap("output_norm.weight", outputNormWeight),
	)
	if err != nil {
		return nil, nil, err
	}
	normedFinal := builder.AddNode(finalNorm, hidden)

	// LM Head.
	lmHead := newLMHeadNode(proxy, lmHeadWeight, 0)
	output := builder.AddNode(lmHead, normedFinal)

	g, err := builder.Build(output)
	if err != nil {
		return nil, nil, fmt.Errorf("build graph: %w", err)
	}

	g.SetEngineProxy(proxy)
	return g, embedWeight, nil
}

// buildNemotronHAttention constructs the GQA attention sub-graph for a single
// Nemotron-H attention layer.
func buildNemotronHAttention(
	tensors map[string]*tensor.TensorNumeric[float32],
	proxy *compute.EngineProxy[float32],
	ops numeric.Float32Ops,
	builder *graph.Builder[float32],
	normed graph.Node[float32],
	nc NemotronHConfig,
	layerIdx int,
	prefix string,
	headDim int,
) (graph.Node[float32], error) {
	tl := newTensorLookup(tensors)
	pw := newParamWrapper[float32]()

	qW, err := tl.Lookup(prefix + "attn_q.weight")
	if err != nil {
		return nil, err
	}
	kW, err := tl.Lookup(prefix + "attn_k.weight")
	if err != nil {
		return nil, err
	}
	vW, err := tl.Lookup(prefix + "attn_v.weight")
	if err != nil {
		return nil, err
	}
	oW, err := tl.Lookup(prefix + "attn_output.weight")
	if err != nil {
		return nil, err
	}

	qWT, err := cpuTranspose2D(qW)
	if err != nil {
		return nil, fmt.Errorf("transpose q: %w", err)
	}
	kWT, err := cpuTranspose2D(kW)
	if err != nil {
		return nil, fmt.Errorf("transpose k: %w", err)
	}
	vWT, err := cpuTranspose2D(vW)
	if err != nil {
		return nil, fmt.Errorf("transpose v: %w", err)
	}
	oWT, err := cpuTranspose2D(oW)
	if err != nil {
		return nil, fmt.Errorf("transpose o: %w", err)
	}

	wq := core.NewDenseFromParams(
		core.NewLinearFromParam(proxy, pw.Wrap(prefix+"attn_q.weight", qWT)), nil,
	)
	wk := core.NewDenseFromParams(
		core.NewLinearFromParam(proxy, pw.Wrap(prefix+"attn_k.weight", kWT)), nil,
	)
	wv := core.NewDenseFromParams(
		core.NewLinearFromParam(proxy, pw.Wrap(prefix+"attn_v.weight", vWT)), nil,
	)
	wo := core.NewDenseFromParams(
		core.NewLinearFromParam(proxy, pw.Wrap(prefix+"attn_output.weight", oWT)), nil,
	)

	maxSeqLen := nc.MaxSeqLen
	if maxSeqLen == 0 {
		maxSeqLen = 2048
	}
	rope, err := embeddings.NewRotaryPositionalEmbedding[float32](
		context.Background(), proxy, headDim, maxSeqLen,
		embeddings.WithRotaryBase(nc.RopeTheta),
	)
	if err != nil {
		return nil, fmt.Errorf("rope: %w", err)
	}

	gqa, err := attention.NewGroupedQueryAttentionFromParams[float32](
		proxy, ops, nc.HiddenSize, nc.AttnHeads, nc.KVHeads,
		wq, wk, wv, wo, rope, headDim,
	)
	if err != nil {
		return nil, fmt.Errorf("gqa: %w", err)
	}
	gqa.LayerIndex = layerIdx

	return builder.AddNode(gqa, normed), nil
}

// buildNemotronHFFN constructs a SwiGLU FFN sub-graph for a Nemotron-H layer.
func buildNemotronHFFN(
	tensors map[string]*tensor.TensorNumeric[float32],
	proxy *compute.EngineProxy[float32],
	ops numeric.Float32Ops,
	builder *graph.Builder[float32],
	normed graph.Node[float32],
	nc NemotronHConfig,
	layerIdx int,
	prefix string,
) (graph.Node[float32], error) {
	tl := newTensorLookup(tensors)

	gateW, err := tl.Lookup(prefix + "ffn_gate.weight")
	if err != nil {
		return nil, err
	}
	upW, err := tl.Lookup(prefix + "ffn_up.weight")
	if err != nil {
		return nil, err
	}
	downW, err := tl.Lookup(prefix + "ffn_down.weight")
	if err != nil {
		return nil, err
	}

	ffn, err := core.NewFFN[float32](
		prefix+"mlp", proxy, ops,
		nc.HiddenSize, nc.IntermediateSize, nc.HiddenSize,
		core.WithSwiGLU[float32](),
		core.WithFFNNoBias[float32](),
	)
	if err != nil {
		return nil, err
	}

	gateWT, err := cpuTranspose2D(gateW)
	if err != nil {
		return nil, fmt.Errorf("transpose gate: %w", err)
	}
	upWT, err := cpuTranspose2D(upW)
	if err != nil {
		return nil, fmt.Errorf("transpose up: %w", err)
	}
	downWT, err := cpuTranspose2D(downW)
	if err != nil {
		return nil, fmt.Errorf("transpose down: %w", err)
	}

	ffnParams := ffn.Parameters()
	ffnParams[0].Value = gateWT
	ffnParams[1].Value = downWT
	ffnParams[2].Value = upWT

	return builder.AddNode(ffn, normed), nil
}

// buildNemotronHMoE constructs the MoE sub-graph for a single Nemotron-H MoE
// layer. This follows the DeepSeek stacked expert tensor pattern with support
// for shared experts.
func buildNemotronHMoE(
	tensors map[string]*tensor.TensorNumeric[float32],
	nc NemotronHConfig,
	proxy *compute.EngineProxy[float32],
	ops numeric.Float32Ops,
	builder *graph.Builder[float32],
	normed graph.Node[float32],
	layerIdx int,
	prefix string,
) (graph.Node[float32], error) {
	numExperts := nc.NumExperts
	if numExperts == 0 {
		numExperts = 128
	}
	topK := nc.NumExpertsPerToken
	if topK == 0 {
		topK = 6
	}

	tl := newTensorLookup(tensors)

	// Load router weight.
	routerW, err := tl.Lookup(prefix + "ffn_gate_inp.weight")
	if err != nil {
		return nil, err
	}

	// Load stacked expert weights.
	gateExpsW, err := tl.Lookup(prefix + "ffn_gate_exps.weight")
	if err != nil {
		return nil, err
	}
	upExpsW, err := tl.Lookup(prefix + "ffn_up_exps.weight")
	if err != nil {
		return nil, err
	}
	downExpsW, err := tl.Lookup(prefix + "ffn_down_exps.weight")
	if err != nil {
		return nil, err
	}

	// Split stacked expert weights into individual expert FFNs.
	experts := make([]graph.Node[float32], numExperts)
	for e := 0; e < numExperts; e++ {
		expertFFN, err := buildExpertFFN(
			proxy, ops, gateExpsW, upExpsW, downExpsW,
			e, numExperts, nc.HiddenSize, nc.IntermediateSize,
			fmt.Sprintf("layer%d_expert%d", layerIdx, e),
		)
		if err != nil {
			return nil, fmt.Errorf("expert %d: %w", e, err)
		}
		experts[e] = expertFFN
	}

	gate := core.NewMoEGate[float32](proxy, ops, topK)
	moe := core.NewMixtureOfExperts[float32](proxy, ops, gate, experts, numExperts, topK)

	// Build shared experts if present.
	if nc.NumSharedExperts > 0 {
		sharedFFN, err := buildNemotronHSharedExpertFFN(tensors, proxy, ops, prefix, nc)
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

// buildNemotronHSharedExpertFFN creates the shared expert FFN from GGUF tensors.
func buildNemotronHSharedExpertFFN(
	tensors map[string]*tensor.TensorNumeric[float32],
	engine *compute.EngineProxy[float32],
	ops numeric.Float32Ops,
	prefix string,
	nc NemotronHConfig,
) (*core.FFN[float32], error) {
	tl := newTensorLookup(tensors)

	gateW, err := tl.Lookup(prefix + "ffn_shared_expert_gate.weight")
	if err != nil {
		return nil, err
	}
	upW, err := tl.Lookup(prefix + "ffn_shared_expert_up.weight")
	if err != nil {
		return nil, err
	}
	downW, err := tl.Lookup(prefix + "ffn_shared_expert_down.weight")
	if err != nil {
		return nil, err
	}

	gateWT, err := cpuTranspose2D(gateW)
	if err != nil {
		return nil, err
	}
	upWT, err := cpuTranspose2D(upW)
	if err != nil {
		return nil, err
	}
	downWT, err := cpuTranspose2D(downW)
	if err != nil {
		return nil, err
	}

	ffn, err := core.NewFFN[float32](
		prefix+"shared_expert", engine, ops,
		nc.HiddenSize, nc.IntermediateSize, nc.HiddenSize,
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

// loadNemotronHSSMWeights overwrites the random-initialized parameters of a
// MIMOMambaBlock with pre-trained GGUF tensors for Nemotron-H Mamba-2 layers.
//
// Nemotron-H uses these GGUF tensor names:
//
//	blk.{i}.ssm_in.weight     -> inProj   (may be fused x+z projection)
//	blk.{i}.ssm_conv1d.weight -> convWeight
//	blk.{i}.ssm_dt.weight     -> dtProj   (may embed x_proj for fused B/C)
//	blk.{i}.ssm_A.weight      -> A        (may NOT be in log-space)
//	blk.{i}.ssm_D.weight      -> D
//	blk.{i}.ssm_out.weight    -> outProj
//
// MIMOMambaBlock.Parameters() returns:
//
//	[0] inProj weight
//	[1] convWeight
//	[2] xProj weight
//	[3] dtProj weight
//	[4..4+2*numHeads-1] per-head A and D (alternating: A_h0, D_h0, A_h1, D_h1, ...)
//	[4+2*numHeads] headMix weight
//	[4+2*numHeads+1] outProj weight
func loadNemotronHSSMWeights(
	block *ssm.MIMOMambaBlock[float32],
	tensors map[string]*tensor.TensorNumeric[float32],
	prefix string,
	nc NemotronHConfig,
	dtRank, ssmHeadDim int,
) error {
	params := block.Parameters()
	tl := newTensorLookup(tensors)

	// Load ssm_in.weight -> inProj.
	inW, err := tl.Lookup(prefix + "ssm_in.weight")
	if err != nil {
		return err
	}
	if len(inW.Shape()) == 2 {
		transposed, err := cpuTranspose2D(inW)
		if err != nil {
			return fmt.Errorf("transpose ssm_in: %w", err)
		}
		params[0].Value = transposed
	} else {
		params[0].Value = inW
	}

	// Load ssm_conv1d.weight -> convWeight.
	convW, err := tl.Lookup(prefix + "ssm_conv1d.weight")
	if err != nil {
		return err
	}
	params[1].Value = convW

	// For Nemotron-H, B/C projection may be fused into ssm_in, so xProj might
	// not exist as a separate tensor. Leave default-initialized if missing.
	if xpW, ok := tl.Optional(prefix + "ssm_x_proj.weight"); ok {
		if len(xpW.Shape()) == 2 {
			transposed, err := cpuTranspose2D(xpW)
			if err != nil {
				return fmt.Errorf("transpose ssm_x_proj: %w", err)
			}
			params[2].Value = transposed
		} else {
			params[2].Value = xpW
		}
	}

	// Load ssm_dt.weight -> dtProj.
	dtW, err := tl.Lookup(prefix + "ssm_dt.weight")
	if err != nil {
		return err
	}
	if len(dtW.Shape()) == 2 {
		transposed, err := cpuTranspose2D(dtW)
		if err != nil {
			return fmt.Errorf("transpose ssm_dt: %w", err)
		}
		params[3].Value = transposed
	} else {
		params[3].Value = dtW
	}

	// Load ssm_A.weight -> per-head A parameters.
	// Nemotron-H ssm_A may not be in log-space; check and apply log if values
	// are all positive (log-space values would be mixed-sign for typical inits).
	aW, err := tl.Lookup(prefix + "ssm_A.weight")
	if err != nil {
		return err
	}

	aData := aW.Data()
	needsLog := true
	for _, v := range aData {
		if v <= 0 {
			needsLog = false
			break
		}
	}
	if needsLog {
		logData := make([]float32, len(aData))
		for i, v := range aData {
			logData[i] = float32(math.Log(float64(v)))
		}
		var err error
		aW, err = tensor.New(aW.Shape(), logData)
		if err != nil {
			return fmt.Errorf("create log-space A: %w", err)
		}
	}

	// Distribute A across heads.
	numHeads := nc.SSMNumHeads
	if numHeads == 0 {
		numHeads = 1
	}
	baseIdx := 4
	for h := 0; h < numHeads; h++ {
		headData := make([]float32, ssmHeadDim*nc.SSMStateSize)
		off := h * ssmHeadDim * nc.SSMStateSize
		src := aW.Data()
		if off+ssmHeadDim*nc.SSMStateSize <= len(src) {
			copy(headData, src[off:off+ssmHeadDim*nc.SSMStateSize])
		} else if len(src) >= ssmHeadDim*nc.SSMStateSize {
			copy(headData, src[:ssmHeadDim*nc.SSMStateSize])
		}
		aHead, err := tensor.New([]int{ssmHeadDim, nc.SSMStateSize}, headData)
		if err != nil {
			return fmt.Errorf("create per-head A for head %d: %w", h, err)
		}
		aIdx := baseIdx + h*2
		if aIdx < len(params) {
			params[aIdx].Value = aHead
		}
	}

	// Load ssm_D.weight -> per-head D parameters.
	dW, err := tl.Lookup(prefix + "ssm_D.weight")
	if err != nil {
		return err
	}
	for h := 0; h < numHeads; h++ {
		headData := make([]float32, ssmHeadDim)
		off := h * ssmHeadDim
		src := dW.Data()
		if off+ssmHeadDim <= len(src) {
			copy(headData, src[off:off+ssmHeadDim])
		} else if len(src) >= ssmHeadDim {
			copy(headData, src[:ssmHeadDim])
		}
		dHead, err := tensor.New([]int{ssmHeadDim}, headData)
		if err != nil {
			return fmt.Errorf("create per-head D for head %d: %w", h, err)
		}
		dIdx := baseIdx + h*2 + 1
		if dIdx < len(params) {
			params[dIdx].Value = dHead
		}
	}

	// headMix: initialize to identity if missing from GGUF.
	hmIdx := baseIdx + numHeads*2
	if hmT, ok := tl.Optional(prefix + "ssm_head_mix.weight"); ok && hmIdx < len(params) {
		if len(hmT.Shape()) == 2 {
			transposed, err := cpuTranspose2D(hmT)
			if err != nil {
				return fmt.Errorf("transpose ssm_head_mix: %w", err)
			}
			params[hmIdx].Value = transposed
		} else {
			params[hmIdx].Value = hmT
		}
	} else if hmIdx < len(params) {
		// Identity initialization for head mixing.
		dInner := nc.IntermediateSize
		if dInner == 0 {
			dInner = nc.HiddenSize * 2
		}
		idData := make([]float32, dInner*dInner)
		for j := 0; j < dInner; j++ {
			idData[j*dInner+j] = 1.0
		}
		idTensor, err := tensor.New([]int{dInner, dInner}, idData)
		if err != nil {
			return fmt.Errorf("create identity head_mix: %w", err)
		}
		params[hmIdx].Value = idTensor
	}

	// Load ssm_out.weight -> outProj.
	opIdx := hmIdx + 1
	outW, err := tl.Lookup(prefix + "ssm_out.weight")
	if err != nil {
		return err
	}
	if opIdx < len(params) {
		if len(outW.Shape()) == 2 {
			transposed, err := cpuTranspose2D(outW)
			if err != nil {
				return fmt.Errorf("transpose ssm_out: %w", err)
			}
			params[opIdx].Value = transposed
		} else {
			params[opIdx].Value = outW
		}
	}

	return nil
}
