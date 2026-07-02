package inference

import (
	"context"
	"fmt"
	"math"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/ztensor/types"
	"github.com/zerfoo/zerfoo/layers/normalization"
	"github.com/zerfoo/zerfoo/layers/ssm"
	"github.com/zerfoo/zerfoo/model/gguf"
)

// MambaConfig holds Mamba-specific model configuration.
type MambaConfig struct {
	NumLayers  int
	DModel     int
	DState     int
	DConv      int
	DInner     int
	VocabSize  int
	EOSTokenID int
	RMSNormEps float32
}

// MambaConfigFromGGUF extracts Mamba configuration from GGUF ModelConfig.
// Fields are mapped as: HiddenSize -> DModel, NumKVHeads -> DState,
// IntermediateSize -> DInner. DConv defaults to 4 if not specified.
func MambaConfigFromGGUF(cfg *gguf.ModelConfig) MambaConfig {
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
	return MambaConfig{
		NumLayers:  cfg.NumLayers,
		DModel:     cfg.HiddenSize,
		DState:     dState,
		DConv:      dConv,
		DInner:     dInner,
		VocabSize:  cfg.VocabSize,
		EOSTokenID: 0,
		RMSNormEps: eps,
	}
}

// MambaConfigFromMetadata extracts Mamba configuration from a raw metadata map.
func MambaConfigFromMetadata(meta map[string]interface{}) MambaConfig {
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
	vocabSize := getInt(meta, "vocab_size")
	eosID := getInt(meta, "eos_token_id")

	return MambaConfig{
		NumLayers:  numLayers,
		DModel:     dModel,
		DState:     dState,
		DConv:      dConv,
		DInner:     dInner,
		VocabSize:  vocabSize,
		EOSTokenID: eosID,
		RMSNormEps: 1e-5,
	}
}

// buildMambaGraph constructs a computation graph for the Mamba architecture
// from pre-loaded GGUF tensors. Unlike transformer architectures, Mamba uses
// selective state space model blocks instead of attention + FFN.
//
// Architecture:
//
//	Embed -> [RMSNorm -> MambaBlock -> Add] x N -> RMSNorm -> LMHead
func buildMambaGraph(
	tensors map[string]*tensor.TensorNumeric[float32],
	cfg *gguf.ModelConfig,
	engine compute.Engine[float32],
) (*graph.Graph[float32], *tensor.TensorNumeric[float32], error) {
	mc := MambaConfigFromGGUF(cfg)
	return BuildMamba3(mc, tensors, engine)
}

// BuildMamba3 constructs a computation graph for Mamba-3 from a weight map.
//
// Expected tensor names:
//
//	token_embd.weight                 — [vocab_size, d_model]
//	output.weight                     — [vocab_size, d_model]
//	output_norm.weight                — [d_model]
//	mamba.{i}.norm.weight             — [d_model]
//	mamba.{i}.in_proj.weight          — [2*d_inner, d_model]
//	mamba.{i}.conv1d.weight           — [d_inner, 1, d_conv]
//	mamba.{i}.conv1d.bias             — [d_inner] (optional)
//	mamba.{i}.x_proj.weight           — [dt_rank + 2*d_state, d_inner]
//	mamba.{i}.dt_proj.weight          — [d_inner, dt_rank]
//	mamba.{i}.dt_proj.bias            — [d_inner] (optional)
//	mamba.{i}.A_log                   — [d_inner, d_state]
//	mamba.{i}.D                       — [d_inner]
//	mamba.{i}.out_proj.weight         — [d_model, d_inner]
func BuildMamba3(
	mc MambaConfig,
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

	// Derive dt_rank from first layer's x_proj weight shape if possible,
	// otherwise default to ceil(d_model / 16).
	dtRank := int(math.Ceil(float64(mc.DModel) / 16))
	if xpW, ok := tensors["mamba.0.x_proj.weight"]; ok {
		// x_proj shape: [dt_rank + 2*d_state, d_inner]
		dtRank = xpW.Shape()[0] - 2*mc.DState
		if dtRank <= 0 {
			dtRank = int(math.Ceil(float64(mc.DModel) / 16))
		}
	}

	for i := 0; i < mc.NumLayers; i++ {
		prefix := fmt.Sprintf("mamba.%d.", i)

		// Pre-layer RMSNorm.
		normW, err := tl.Lookup(prefix + "norm.weight")
		if err != nil {
			return nil, nil, err
		}
		norm, err := normalization.NewRMSNormFromParam[float32](
			proxy, ops, mc.RMSNormEps, pw.Wrap(prefix+"norm.weight", normW),
		)
		if err != nil {
			return nil, nil, err
		}
		normed := builder.AddNode(norm, hidden)

		// Create MambaBlock with matching dimensions.
		block, err := ssm.NewMambaBlock[float32](
			prefix+"block", proxy, ops,
			mc.DModel, mc.DInner, mc.DState, dtRank, mc.DConv,
		)
		if err != nil {
			return nil, nil, fmt.Errorf("layer %d mamba block: %w", i, err)
		}

		// Load GGUF weights into the MambaBlock parameters.
		if err := loadMambaBlockWeights(block, tensors, prefix, mc, dtRank); err != nil {
			return nil, nil, fmt.Errorf("layer %d load weights: %w", i, err)
		}

		mambaOut := builder.AddNode(block, normed)

		// Residual connection.
		resAdd := &mambaResidualAddNode[float32]{engine: proxy}
		hidden = builder.AddNode(resAdd, mambaOut, hidden)
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

// loadMambaBlockWeights overwrites the random-initialized parameters of a
// MambaBlock with pre-trained GGUF tensors.
// GGUF stores projection weights as [out_features, in_features], but
// Linear.Forward does input * weight, so weights need transposing to
// [in_features, out_features].
func loadMambaBlockWeights(
	block *ssm.MambaBlock[float32],
	tensors map[string]*tensor.TensorNumeric[float32],
	prefix string,
	mc MambaConfig,
	dtRank int,
) error {
	params := block.Parameters()
	// MambaBlock.Parameters() returns:
	//   [0] inProj weight
	//   [1] conv1d weight
	//   [2] conv1d bias
	//   [3] xProj weight
	//   [4] dtProj weight
	//   [5] A_log
	//   [6] D
	//   [7] outProj weight

	weightNames := []struct {
		idx       int
		name      string
		transpose bool // true for 2D projection weights that need transposing
	}{
		{0, prefix + "in_proj.weight", true},
		{1, prefix + "conv1d.weight", false},
		{2, prefix + "conv1d.bias", false},
		{3, prefix + "x_proj.weight", true},
		{4, prefix + "dt_proj.weight", true},
		{5, prefix + "A_log", false},
		{6, prefix + "D", false},
		{7, prefix + "out_proj.weight", true},
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

// cpuTranspose2D transposes a 2D tensor [rows, cols] -> [cols, rows].
func cpuTranspose2D(t *tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
	shape := t.Shape()
	rows, cols := shape[0], shape[1]
	data := t.Data()
	transposed := make([]float32, len(data))
	for r := 0; r < rows; r++ {
		for c := 0; c < cols; c++ {
			transposed[c*rows+r] = data[r*cols+c]
		}
	}
	return tensor.New([]int{cols, rows}, transposed)
}

// mambaResidualAddNode computes element-wise addition for residual connections.
// Input[0] = block output, Input[1] = residual.
type mambaResidualAddNode[T tensor.Numeric] struct {
	engine compute.Engine[T]
}

func (n *mambaResidualAddNode[T]) OpType() string                  { return "MambaResidualAdd" }
func (n *mambaResidualAddNode[T]) Attributes() map[string]any       { return nil }
func (n *mambaResidualAddNode[T]) OutputShape() []int               { return nil }
func (n *mambaResidualAddNode[T]) Parameters() []*graph.Parameter[T] { return nil }

func (n *mambaResidualAddNode[T]) Forward(ctx context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if len(inputs) != 2 {
		return nil, fmt.Errorf("MambaResidualAdd requires 2 inputs, got %d", len(inputs))
	}
	return n.engine.Add(ctx, inputs[0], inputs[1])
}

func (n *mambaResidualAddNode[T]) Backward(_ context.Context, _ types.BackwardMode, _ *tensor.TensorNumeric[T], _ ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	return nil, nil
}
