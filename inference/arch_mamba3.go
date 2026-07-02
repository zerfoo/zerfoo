package inference

import (
	"fmt"
	"math"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/zerfoo/layers/normalization"
	"github.com/zerfoo/zerfoo/layers/ssm"
	"github.com/zerfoo/zerfoo/model/gguf"
)

// Mamba3Config holds Mamba 3-specific model configuration.
// Mamba 3 extends Mamba with multi-head MIMO SSM, exponential-trapezoidal
// discretization, and cross-head mixing.
type Mamba3Config struct {
	NumLayers  int
	DModel     int
	DState     int
	DConv      int
	DInner     int
	NumHeads   int
	VocabSize  int
	EOSTokenID int
	RMSNormEps float32
}

// Mamba3ConfigFromGGUF extracts Mamba 3 configuration from GGUF ModelConfig.
// Fields are mapped as: HiddenSize -> DModel, NumKVHeads -> DState,
// IntermediateSize -> DInner, NumHeads -> NumHeads. DConv defaults to 4.
func Mamba3ConfigFromGGUF(cfg *gguf.ModelConfig) Mamba3Config {
	dInner := cfg.IntermediateSize
	if dInner == 0 {
		dInner = cfg.HiddenSize * 2
	}
	dState := cfg.NumKVHeads
	if dState == 0 {
		dState = 16
	}
	dConv := 4
	eps := cfg.RMSNormEps
	if eps == 0 {
		eps = 1e-5
	}
	numHeads := cfg.NumHeads
	if numHeads == 0 {
		numHeads = 1
	}
	// Ensure dInner is divisible by numHeads; fall back to 1 head if not.
	if dInner%numHeads != 0 {
		numHeads = 1
	}
	return Mamba3Config{
		NumLayers:  cfg.NumLayers,
		DModel:     cfg.HiddenSize,
		DState:     dState,
		DConv:      dConv,
		DInner:     dInner,
		NumHeads:   numHeads,
		VocabSize:  cfg.VocabSize,
		EOSTokenID: 0,
		RMSNormEps: eps,
	}
}

// Mamba3ConfigFromMetadata extracts Mamba 3 configuration from a raw metadata map.
func Mamba3ConfigFromMetadata(meta map[string]interface{}) Mamba3Config {
	dModel := getInt(meta, "d_model")
	if dModel == 0 {
		dModel = getInt(meta, "hidden_size")
	}
	dInner := getInt(meta, "d_inner")
	if dInner == 0 {
		dInner = getInt(meta, "intermediate_size")
	}
	if dInner == 0 {
		dInner = dModel * 2
	}
	dState := getInt(meta, "d_state")
	if dState == 0 {
		dState = 16
	}
	dConv := getInt(meta, "d_conv")
	if dConv == 0 {
		dConv = 4
	}
	numLayers := getInt(meta, "num_layers")
	if numLayers == 0 {
		numLayers = getInt(meta, "num_hidden_layers")
	}
	numHeads := getInt(meta, "num_heads")
	if numHeads == 0 {
		numHeads = getInt(meta, "num_attention_heads")
	}
	if numHeads == 0 {
		numHeads = 1
	}
	if dInner%numHeads != 0 {
		numHeads = 1
	}
	vocabSize := getInt(meta, "vocab_size")
	eosID := getInt(meta, "eos_token_id")

	return Mamba3Config{
		NumLayers:  numLayers,
		DModel:     dModel,
		DState:     dState,
		DConv:      dConv,
		DInner:     dInner,
		NumHeads:   numHeads,
		VocabSize:  vocabSize,
		EOSTokenID: eosID,
		RMSNormEps: 1e-5,
	}
}

// buildMamba3Graph constructs a computation graph for the Mamba 3 architecture
// from pre-loaded GGUF tensors. Mamba 3 uses multi-head MIMO SSM blocks with
// exponential-trapezoidal discretization and cross-head mixing.
//
// Architecture:
//
//	Embed -> [RMSNorm -> MIMOMambaBlock(ExpTrap) -> Add] x N -> RMSNorm -> LMHead
func buildMamba3Graph(
	tensors map[string]*tensor.TensorNumeric[float32],
	cfg *gguf.ModelConfig,
	engine compute.Engine[float32],
) (*graph.Graph[float32], *tensor.TensorNumeric[float32], error) {
	mc := Mamba3ConfigFromGGUF(cfg)
	return BuildMamba3MIMO(mc, tensors, engine)
}

// BuildMamba3MIMO constructs a computation graph for Mamba 3 using MIMO SSM
// blocks with exponential-trapezoidal discretization.
//
// Expected tensor names:
//
//	token_embd.weight                      — [vocab_size, d_model]
//	output.weight                          — [vocab_size, d_model]
//	output_norm.weight                     — [d_model]
//	mamba3.{i}.norm.weight                 — [d_model]
//	mamba3.{i}.in_proj.weight              — [2*d_inner, d_model]
//	mamba3.{i}.conv1d.weight               — [d_inner, 1, d_conv]
//	mamba3.{i}.x_proj.weight               — [dt_rank + 2*d_state*num_heads, d_inner]
//	mamba3.{i}.dt_proj.weight              — [d_inner, dt_rank]
//	mamba3.{i}.A_log.{h}                   — [head_dim, d_state] per head
//	mamba3.{i}.D.{h}                       — [head_dim] per head
//	mamba3.{i}.head_mix.weight             — [d_inner, d_inner]
//	mamba3.{i}.out_proj.weight             — [d_model, d_inner]
func BuildMamba3MIMO(
	mc Mamba3Config,
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
		lmHeadWeight = embedWeight // tie embeddings
	}

	outputNormWeight, err := tl.Lookup("output_norm.weight")
	if err != nil {
		return nil, nil, err
	}

	proxy := compute.NewEngineProxy[float32](engine)
	builder := graph.NewBuilder[float32](proxy)
	input := builder.Input([]int{1, 1})

	// Embedding lookup.
	embNode := newEmbeddingNode(proxy, embedWeight, 0)
	hidden := builder.AddNode(embNode, input)

	headDim := mc.DInner / mc.NumHeads

	// Derive dt_rank from first layer's x_proj weight shape if possible,
	// otherwise default to ceil(d_model / 16).
	dtRank := int(math.Ceil(float64(mc.DModel) / 16))
	if xpW, ok := tensors["mamba3.0.x_proj.weight"]; ok {
		// x_proj shape: [dt_rank + 2*d_state*num_heads, d_inner]
		dtRank = xpW.Shape()[0] - 2*mc.DState*mc.NumHeads
		if dtRank <= 0 {
			dtRank = int(math.Ceil(float64(mc.DModel) / 16))
		}
	}

	for i := 0; i < mc.NumLayers; i++ {
		prefix := fmt.Sprintf("mamba3.%d.", i)

		// Pre-layer RMSNorm.
		normW, lErr := tl.Lookup(prefix + "norm.weight")
		if lErr != nil {
			return nil, nil, lErr
		}
		norm, nErr := normalization.NewRMSNormFromParam[float32](
			proxy, ops, mc.RMSNormEps, pw.Wrap(prefix+"norm.weight", normW),
		)
		if nErr != nil {
			return nil, nil, nErr
		}
		normed := builder.AddNode(norm, hidden)

		// Create MIMOMambaBlock with exponential-trapezoidal discretization.
		block, bErr := ssm.NewMIMOMambaBlock[float32](
			prefix+"mimo_block", proxy, ops,
			mc.DModel, mc.DInner, mc.DState, dtRank, mc.DConv, mc.NumHeads,
			ssm.WithMIMODiscretizationMode[float32](ssm.ExpTrap),
		)
		if bErr != nil {
			return nil, nil, fmt.Errorf("layer %d mimo block: %w", i, bErr)
		}

		// Load GGUF weights into the MIMOMambaBlock parameters.
		if wErr := loadMamba3MIMOWeights(block, tensors, prefix, mc, dtRank, headDim); wErr != nil {
			return nil, nil, fmt.Errorf("layer %d load weights: %w", i, wErr)
		}

		mimoOut := builder.AddNode(block, normed)

		// Residual connection.
		resAdd := &mambaResidualAddNode[float32]{engine: proxy}
		hidden = builder.AddNode(resAdd, mimoOut, hidden)
	}

	// Final RMSNorm.
	finalNorm, err := normalization.NewRMSNormFromParam[float32](
		proxy, ops, mc.RMSNormEps, pw.Wrap("output_norm.weight", outputNormWeight),
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

// loadMamba3MIMOWeights overwrites the random-initialized parameters of a
// MIMOMambaBlock with pre-trained GGUF tensors.
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
func loadMamba3MIMOWeights(
	block *ssm.MIMOMambaBlock[float32],
	tensors map[string]*tensor.TensorNumeric[float32],
	prefix string,
	mc Mamba3Config,
	dtRank, headDim int,
) error {
	params := block.Parameters()

	// Load projection weights (need transposing for 2D).
	projWeights := []struct {
		idx       int
		name      string
		transpose bool
	}{
		{0, prefix + "in_proj.weight", true},
		{1, prefix + "conv1d.weight", false},
		{2, prefix + "x_proj.weight", true},
		{3, prefix + "dt_proj.weight", true},
	}

	for _, wn := range projWeights {
		t, ok := tensors[wn.name]
		if !ok {
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

	// Load per-head A and D parameters.
	// Parameters are ordered: A_h0, D_h0, A_h1, D_h1, ...
	baseIdx := 4
	for h := 0; h < mc.NumHeads; h++ {
		aName := fmt.Sprintf("%sA_log.%d", prefix, h)
		aT, ok := tensors[aName]
		if !ok {
			// Fall back to single shared A_log for all heads.
			aName = prefix + "A_log"
			aT, ok = tensors[aName]
			if !ok {
				return fmt.Errorf("missing tensor %q", fmt.Sprintf("%sA_log.%d", prefix, h))
			}
			// Slice to per-head dimensions if shared.
			aData := aT.Data()
			headData := make([]float32, headDim*mc.DState)
			off := h * headDim * mc.DState
			if off+headDim*mc.DState <= len(aData) {
				copy(headData, aData[off:off+headDim*mc.DState])
			} else {
				copy(headData, aData[:headDim*mc.DState])
			}
			var err error
			aT, err = tensor.New([]int{headDim, mc.DState}, headData)
			if err != nil {
				return fmt.Errorf("create per-head A for head %d: %w", h, err)
			}
		}
		aIdx := baseIdx + h*2
		if aIdx < len(params) {
			params[aIdx].Value = aT
		}

		dName := fmt.Sprintf("%sD.%d", prefix, h)
		dT, ok := tensors[dName]
		if !ok {
			// Fall back to single shared D for all heads.
			dName = prefix + "D"
			dT, ok = tensors[dName]
			if !ok {
				return fmt.Errorf("missing tensor %q", fmt.Sprintf("%sD.%d", prefix, h))
			}
			// Slice to per-head dimensions if shared.
			dData := dT.Data()
			headData := make([]float32, headDim)
			off := h * headDim
			if off+headDim <= len(dData) {
				copy(headData, dData[off:off+headDim])
			} else {
				copy(headData, dData[:headDim])
			}
			var err error
			dT, err = tensor.New([]int{headDim}, headData)
			if err != nil {
				return fmt.Errorf("create per-head D for head %d: %w", h, err)
			}
		}
		dIdx := baseIdx + h*2 + 1
		if dIdx < len(params) {
			params[dIdx].Value = dT
		}
	}

	// Load head_mix weight.
	hmIdx := baseIdx + mc.NumHeads*2
	hmName := prefix + "head_mix.weight"
	if hmT, ok := tensors[hmName]; ok && hmIdx < len(params) {
		if len(hmT.Shape()) == 2 {
			transposed, err := cpuTranspose2D(hmT)
			if err != nil {
				return fmt.Errorf("transpose %s: %w", hmName, err)
			}
			params[hmIdx].Value = transposed
		} else {
			params[hmIdx].Value = hmT
		}
	}

	// Load out_proj weight.
	opIdx := hmIdx + 1
	opName := prefix + "out_proj.weight"
	opT, ok := tensors[opName]
	if !ok {
		return fmt.Errorf("missing tensor %q", opName)
	}
	if opIdx < len(params) {
		if len(opT.Shape()) == 2 {
			transposed, err := cpuTranspose2D(opT)
			if err != nil {
				return fmt.Errorf("transpose %s: %w", opName, err)
			}
			params[opIdx].Value = transposed
		} else {
			params[opIdx].Value = opT
		}
	}

	return nil
}

// parseMamba3Config parses Mamba 3-family config.json fields.
func parseMamba3Config(raw map[string]interface{}) (*ModelMetadata, error) {
	meta := &ModelMetadata{
		Architecture:  getString(raw, "model_type"),
		VocabSize:     getInt(raw, "vocab_size"),
		HiddenSize:    getInt(raw, "d_model"),
		NumLayers:     getInt(raw, "num_hidden_layers"),
		NumQueryHeads: getInt(raw, "num_heads"),
		EOSTokenID:    getInt(raw, "eos_token_id"),
		BOSTokenID:    getInt(raw, "bos_token_id"),
	}
	if meta.HiddenSize == 0 {
		meta.HiddenSize = getInt(raw, "hidden_size")
	}
	if meta.NumLayers == 0 {
		meta.NumLayers = getInt(raw, "num_layers")
	}
	if meta.NumQueryHeads == 0 {
		meta.NumQueryHeads = getInt(raw, "num_attention_heads")
	}
	return meta, nil
}
