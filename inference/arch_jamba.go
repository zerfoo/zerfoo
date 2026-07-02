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

// JambaConfig holds Jamba-specific hybrid model configuration.
type JambaConfig struct {
	NumLayers            int
	HiddenSize           int
	IntermediateSize     int
	AttnHeads            int
	KVHeads              int
	SSMHeads             int // number of SSM heads (maps to DState)
	AttentionLayerOffset int // attention layers at indices that are multiples of this value
	RMSEps               float32
	VocabSize            int
	MaxSeqLen            int
	RopeTheta            float64
	DConv                int // SSM convolution width (default 4)
}

// JambaConfigFromGGUF extracts Jamba configuration from GGUF ModelConfig.
func JambaConfigFromGGUF(cfg *gguf.ModelConfig) JambaConfig {
	eps := cfg.RMSNormEps
	if eps == 0 {
		eps = 1e-5
	}
	dConv := 4
	ssmHeads := cfg.NumKVHeads
	if ssmHeads == 0 {
		ssmHeads = 16
	}
	attnLayerOffset := 8
	ropeTheta := cfg.RopeTheta
	if ropeTheta == 0 {
		ropeTheta = 10000
	}
	return JambaConfig{
		NumLayers:            cfg.NumLayers,
		HiddenSize:           cfg.HiddenSize,
		IntermediateSize:     cfg.IntermediateSize,
		AttnHeads:            cfg.NumHeads,
		KVHeads:              cfg.NumKVHeads,
		SSMHeads:             ssmHeads,
		AttentionLayerOffset: attnLayerOffset,
		RMSEps:               eps,
		VocabSize:            cfg.VocabSize,
		MaxSeqLen:            cfg.MaxSeqLen,
		RopeTheta:            ropeTheta,
		DConv:                dConv,
	}
}

// isAttentionLayer returns true if layer index i should be a Transformer
// attention layer in the Jamba hybrid architecture.
func (jc JambaConfig) isAttentionLayer(i int) bool {
	if jc.AttentionLayerOffset <= 0 {
		return false
	}
	return i%jc.AttentionLayerOffset == 0
}

// parseJambaConfig parses Jamba-family config.json fields.
func parseJambaConfig(raw map[string]interface{}) (*ModelMetadata, error) {
	meta := &ModelMetadata{
		Architecture:          getString(raw, "model_type"),
		VocabSize:             getInt(raw, "vocab_size"),
		HiddenSize:            getInt(raw, "hidden_size"),
		NumLayers:             getInt(raw, "num_hidden_layers"),
		NumQueryHeads:         getInt(raw, "num_attention_heads"),
		NumKeyValueHeads:      getInt(raw, "num_key_value_heads"),
		IntermediateSize:      getInt(raw, "intermediate_size"),
		MaxPositionEmbeddings: getInt(raw, "max_position_embeddings"),
		EOSTokenID:            getInt(raw, "eos_token_id"),
		BOSTokenID:            getInt(raw, "bos_token_id"),
		RopeTheta:             getFloat(raw, "rope_theta"),
	}
	if meta.HiddenSize == 0 {
		meta.HiddenSize = getInt(raw, "d_model")
	}
	if meta.NumLayers == 0 {
		meta.NumLayers = getInt(raw, "num_layers")
	}
	if meta.RopeTheta == 0 {
		meta.RopeTheta = 10000
	}
	return meta, nil
}

// buildJambaGraph constructs a computation graph for the Jamba architecture
// from pre-loaded GGUF tensors.
//
// Architecture:
//
//	Embed -> [RMSNorm -> (Attention+FFN | MambaBlock) -> Add] x N -> RMSNorm -> LMHead
func buildJambaGraph(
	tensors map[string]*tensor.TensorNumeric[float32],
	cfg *gguf.ModelConfig,
	engine compute.Engine[float32],
) (*graph.Graph[float32], *tensor.TensorNumeric[float32], error) {
	jc := JambaConfigFromGGUF(cfg)
	return BuildJamba(jc, tensors, engine)
}

// BuildJamba constructs a computation graph for the Jamba hybrid architecture.
//
// Attention layers use tensor names:
//
//	blk.{i}.attn_norm.weight
//	blk.{i}.attn_q.weight, blk.{i}.attn_k.weight, blk.{i}.attn_v.weight, blk.{i}.attn_output.weight
//	blk.{i}.ffn_norm.weight
//	blk.{i}.ffn_gate.weight, blk.{i}.ffn_up.weight, blk.{i}.ffn_down.weight
//
// SSM layers use tensor names:
//
//	blk.{i}.ssm_norm.weight
//	blk.{i}.ssm_in_proj.weight, blk.{i}.ssm_conv1d.weight, blk.{i}.ssm_x_proj.weight
//	blk.{i}.ssm_dt_proj.weight, blk.{i}.ssm_A_log, blk.{i}.ssm_D, blk.{i}.ssm_out_proj.weight
func BuildJamba(
	jc JambaConfig,
	tensors map[string]*tensor.TensorNumeric[float32],
	engine compute.Engine[float32],
) (*graph.Graph[float32], *tensor.TensorNumeric[float32], error) {
	ops := numeric.Float32Ops{}

	tl := newTensorLookup(tensors)

	pw := newParamWrapper[float32]()

	embedWeight, err := tl.Lookup("token_embd.weight")
	if err != nil {
		return nil, nil, err
	}

	lmHeadWeight, ok := tensors["output.weight"]
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

	// Derive SSM parameters.
	dInner := jc.IntermediateSize
	if dInner == 0 {
		dInner = jc.HiddenSize * 2
	}
	dState := jc.SSMHeads
	if dState == 0 {
		dState = 16
	}
	dConv := jc.DConv
	if dConv == 0 {
		dConv = 4
	}

	dtRank := int(math.Ceil(float64(jc.HiddenSize) / 16))
	if xpW, ok := tensors["blk.0.ssm_x_proj.weight"]; ok {
		dtRank = xpW.Shape()[0] - 2*dState
		if dtRank <= 0 {
			dtRank = int(math.Ceil(float64(jc.HiddenSize) / 16))
		}
	}

	headDim := jc.HiddenSize / jc.AttnHeads
	if headDim == 0 {
		headDim = 1
	}

	for i := 0; i < jc.NumLayers; i++ {
		prefix := fmt.Sprintf("blk.%d.", i)

		if jc.isAttentionLayer(i) {
			// --- Transformer Attention Block ---

			// Pre-attention RMSNorm.
			attnNormW, lErr := tl.Lookup(prefix + "attn_norm.weight")
			if lErr != nil {
				return nil, nil, fmt.Errorf("layer %d attention: %w", i, lErr)
			}
			attnNorm, nErr := normalization.NewRMSNormFromParam[float32](
				proxy, ops, jc.RMSEps, pw.Wrap(prefix+"attn_norm.weight", attnNormW),
			)
			if nErr != nil {
				return nil, nil, fmt.Errorf("layer %d attention norm: %w", i, nErr)
			}
			normed := builder.AddNode(attnNorm, hidden)

			// Q/K/V/O projections.
			qW, lErr := tl.Lookup(prefix + "attn_q.weight")
			if lErr != nil {
				return nil, nil, fmt.Errorf("layer %d: %w", i, lErr)
			}
			kW, lErr := tl.Lookup(prefix + "attn_k.weight")
			if lErr != nil {
				return nil, nil, fmt.Errorf("layer %d: %w", i, lErr)
			}
			vW, lErr := tl.Lookup(prefix + "attn_v.weight")
			if lErr != nil {
				return nil, nil, fmt.Errorf("layer %d: %w", i, lErr)
			}
			oW, lErr := tl.Lookup(prefix + "attn_output.weight")
			if lErr != nil {
				return nil, nil, fmt.Errorf("layer %d: %w", i, lErr)
			}

			qWT, tErr := cpuTranspose2D(qW)
			if tErr != nil {
				return nil, nil, fmt.Errorf("layer %d transpose q: %w", i, tErr)
			}
			kWT, tErr := cpuTranspose2D(kW)
			if tErr != nil {
				return nil, nil, fmt.Errorf("layer %d transpose k: %w", i, tErr)
			}
			vWT, tErr := cpuTranspose2D(vW)
			if tErr != nil {
				return nil, nil, fmt.Errorf("layer %d transpose v: %w", i, tErr)
			}
			oWT, tErr := cpuTranspose2D(oW)
			if tErr != nil {
				return nil, nil, fmt.Errorf("layer %d transpose o: %w", i, tErr)
			}

			wq := core.NewDenseFromParams(
				core.NewLinearFromParam(proxy, pw.Wrap(prefix+"attn_q.weight", qWT)),
				nil,
			)
			wk := core.NewDenseFromParams(
				core.NewLinearFromParam(proxy, pw.Wrap(prefix+"attn_k.weight", kWT)),
				nil,
			)
			wv := core.NewDenseFromParams(
				core.NewLinearFromParam(proxy, pw.Wrap(prefix+"attn_v.weight", vWT)),
				nil,
			)
			wo := core.NewDenseFromParams(
				core.NewLinearFromParam(proxy, pw.Wrap(prefix+"attn_output.weight", oWT)),
				nil,
			)

			maxSeqLen := jc.MaxSeqLen
			if maxSeqLen == 0 {
				maxSeqLen = 2048
			}
			rope, rErr := embeddings.NewRotaryPositionalEmbedding[float32](
				context.Background(), proxy, headDim, maxSeqLen,
				embeddings.WithRotaryBase(jc.RopeTheta),
			)
			if rErr != nil {
				return nil, nil, fmt.Errorf("layer %d rope: %w", i, rErr)
			}

			gqa, gErr := attention.NewGroupedQueryAttentionFromParams[float32](
				proxy, ops, jc.HiddenSize, jc.AttnHeads, jc.KVHeads,
				wq, wk, wv, wo, rope, headDim,
			)
			if gErr != nil {
				return nil, nil, fmt.Errorf("layer %d gqa: %w", i, gErr)
			}
			gqa.LayerIndex = i

			attnOut := builder.AddNode(gqa, normed)

			// Residual add.
			resAdd := &mambaResidualAddNode[float32]{engine: proxy}
			hidden = builder.AddNode(resAdd, attnOut, hidden)

			// Pre-FFN RMSNorm.
			ffnNormW, lErr := tl.Lookup(prefix + "ffn_norm.weight")
			if lErr != nil {
				return nil, nil, fmt.Errorf("layer %d: %w", i, lErr)
			}
			ffnNorm, nErr := normalization.NewRMSNormFromParam[float32](
				proxy, ops, jc.RMSEps, pw.Wrap(prefix+"ffn_norm.weight", ffnNormW),
			)
			if nErr != nil {
				return nil, nil, fmt.Errorf("layer %d ffn norm: %w", i, nErr)
			}
			normed2 := builder.AddNode(ffnNorm, hidden)

			// FFN (SwiGLU).
			gateW, lErr := tl.Lookup(prefix + "ffn_gate.weight")
			if lErr != nil {
				return nil, nil, fmt.Errorf("layer %d: %w", i, lErr)
			}
			upW, lErr := tl.Lookup(prefix + "ffn_up.weight")
			if lErr != nil {
				return nil, nil, fmt.Errorf("layer %d: %w", i, lErr)
			}
			downW, lErr := tl.Lookup(prefix + "ffn_down.weight")
			if lErr != nil {
				return nil, nil, fmt.Errorf("layer %d: %w", i, lErr)
			}

			ffn, fErr := core.NewFFN[float32](
				prefix+"mlp", proxy, ops,
				jc.HiddenSize, jc.IntermediateSize, jc.HiddenSize,
				core.WithSwiGLU[float32](),
				core.WithFFNNoBias[float32](),
			)
			if fErr != nil {
				return nil, nil, fmt.Errorf("layer %d ffn: %w", i, fErr)
			}

			gateWT, tErr := cpuTranspose2D(gateW)
			if tErr != nil {
				return nil, nil, fmt.Errorf("layer %d transpose gate: %w", i, tErr)
			}
			upWT, tErr := cpuTranspose2D(upW)
			if tErr != nil {
				return nil, nil, fmt.Errorf("layer %d transpose up: %w", i, tErr)
			}
			downWT, tErr := cpuTranspose2D(downW)
			if tErr != nil {
				return nil, nil, fmt.Errorf("layer %d transpose down: %w", i, tErr)
			}

			ffnParams := ffn.Parameters()
			ffnParams[0].Value = gateWT
			ffnParams[1].Value = downWT
			ffnParams[2].Value = upWT

			ffnOut := builder.AddNode(ffn, normed2)

			resAdd2 := &mambaResidualAddNode[float32]{engine: proxy}
			hidden = builder.AddNode(resAdd2, ffnOut, hidden)

		} else {
			// --- Mamba SSM Block ---

			normW, lErr := tl.Lookup(prefix + "ssm_norm.weight")
			if lErr != nil {
				return nil, nil, fmt.Errorf("layer %d ssm: %w", i, lErr)
			}
			norm, nErr := normalization.NewRMSNormFromParam[float32](
				proxy, ops, jc.RMSEps, pw.Wrap(prefix+"ssm_norm.weight", normW),
			)
			if nErr != nil {
				return nil, nil, fmt.Errorf("layer %d ssm norm: %w", i, nErr)
			}
			normed := builder.AddNode(norm, hidden)

			block, bErr := ssm.NewMambaBlock[float32](
				prefix+"ssm_block", proxy, ops,
				jc.HiddenSize, dInner, dState, dtRank, dConv,
			)
			if bErr != nil {
				return nil, nil, fmt.Errorf("layer %d mamba block: %w", i, bErr)
			}

			if wErr := loadJambaSSMWeights(block, tensors, prefix, dState, dtRank); wErr != nil {
				return nil, nil, fmt.Errorf("layer %d load ssm weights: %w", i, wErr)
			}

			mambaOut := builder.AddNode(block, normed)

			resAdd := &mambaResidualAddNode[float32]{engine: proxy}
			hidden = builder.AddNode(resAdd, mambaOut, hidden)
		}
	}

	// Final RMSNorm.
	finalNorm, err := normalization.NewRMSNormFromParam[float32](
		proxy, ops, jc.RMSEps, pw.Wrap("output_norm.weight", outputNormWeight),
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

// loadJambaSSMWeights loads GGUF tensors into a MambaBlock for Jamba SSM layers.
func loadJambaSSMWeights(
	block *ssm.MambaBlock[float32],
	tensors map[string]*tensor.TensorNumeric[float32],
	prefix string,
	dState, dtRank int,
) error {
	params := block.Parameters()

	weightNames := []struct {
		idx       int
		name      string
		transpose bool
	}{
		{0, prefix + "ssm_in_proj.weight", true},
		{1, prefix + "ssm_conv1d.weight", false},
		{2, prefix + "ssm_conv1d.bias", false},
		{3, prefix + "ssm_x_proj.weight", true},
		{4, prefix + "ssm_dt_proj.weight", true},
		{5, prefix + "ssm_A_log", false},
		{6, prefix + "ssm_D", false},
		{7, prefix + "ssm_out_proj.weight", true},
	}

	for _, wn := range weightNames {
		t, ok := tensors[wn.name]
		if !ok {
			// conv1d.bias may not be present in all models; skip gracefully
			if wn.idx == 2 {
				continue
			}
			return fmt.Errorf("missing tensor %q", wn.name)
		}
		if wn.idx < len(params) {
			if wn.transpose && len(t.Shape()) == 2 {
				transposed, err := cpuTranspose2D(t)
				if err != nil {
					return fmt.Errorf("transpose %s: %w", wn.name, err)
				}
				params[wn.idx].Value = transposed
			} else {
				params[wn.idx].Value = t
			}
		}
	}

	return nil
}
