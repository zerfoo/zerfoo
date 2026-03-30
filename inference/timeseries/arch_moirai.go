// Package timeseries implements time-series model builders.
package timeseries

import (
	"context"
	"fmt"
	"math"

	"github.com/zerfoo/zerfoo/layers/core"
	"github.com/zerfoo/zerfoo/layers/normalization"
	"github.com/zerfoo/zerfoo/layers/timeseries"
	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/ztensor/types"
)

// MoiraiConfig holds configuration for building a Moirai-2 graph.
type MoiraiConfig struct {
	// NumLayers is the number of transformer encoder layers.
	NumLayers int
	// HiddenDim is the hidden/embedding dimension of the model.
	HiddenDim int
	// NumHeads is the number of attention heads.
	NumHeads int
	// InputDim is the time-series patch length (input dimension per variate).
	InputDim int
	// NumFreqEmbeddings is the maximum number of variate frequency embeddings.
	NumFreqEmbeddings int
	// Horizon is the prediction horizon.
	Horizon int
	// NumVars is the number of output variates.
	NumVars int
	// Training enables random masking on input patches during forward pass.
	Training bool
}

func validateMoiraiConfig(cfg *MoiraiConfig) error {
	if cfg.NumLayers <= 0 {
		return fmt.Errorf("NumLayers must be positive, got %d", cfg.NumLayers)
	}
	if cfg.HiddenDim <= 0 {
		return fmt.Errorf("HiddenDim must be positive, got %d", cfg.HiddenDim)
	}
	if cfg.NumHeads <= 0 {
		return fmt.Errorf("NumHeads must be positive, got %d", cfg.NumHeads)
	}
	if cfg.HiddenDim%cfg.NumHeads != 0 {
		return fmt.Errorf("HiddenDim (%d) must be divisible by NumHeads (%d)", cfg.HiddenDim, cfg.NumHeads)
	}
	if cfg.InputDim <= 0 {
		return fmt.Errorf("InputDim must be positive, got %d", cfg.InputDim)
	}
	if cfg.NumFreqEmbeddings <= 0 {
		return fmt.Errorf("NumFreqEmbeddings must be positive, got %d", cfg.NumFreqEmbeddings)
	}
	if cfg.Horizon <= 0 {
		return fmt.Errorf("Horizon must be positive, got %d", cfg.Horizon)
	}
	if cfg.NumVars <= 0 {
		return fmt.Errorf("NumVars must be positive, got %d", cfg.NumVars)
	}
	return nil
}

// BuildMoirai constructs a Moirai-2 masked encoder transformer computation graph.
//
// The Moirai-2 architecture:
//  1. Variate projection with frequency-aware position embeddings:
//     [batch, numVars, inputDim] -> [batch, numVars, hiddenDim]
//  2. Transformer encoder with pre-norm self-attention + FFN
//  3. Output head: [batch, numVars, hiddenDim] -> [batch, horizon, numVars]
//
// tensors is a map of GGUF tensor name -> tensor data for loading pre-trained weights.
func BuildMoirai[T tensor.Float](
	tensors map[string]*tensor.TensorNumeric[T],
	cfg *MoiraiConfig,
	engine compute.Engine[T],
) (*graph.Graph[T], error) {
	if err := validateMoiraiConfig(cfg); err != nil {
		return nil, fmt.Errorf("invalid Moirai config: %w", err)
	}

	ops := engine.Ops()
	node, err := newMoiraiNode[T](tensors, cfg, engine, ops)
	if err != nil {
		return nil, fmt.Errorf("create Moirai node: %w", err)
	}

	builder := graph.NewBuilder[T](engine)
	input := builder.Input([]int{-1, -1, cfg.InputDim})
	builder.AddNode(node, input)

	return builder.Build(node)
}

// moiraiEncoderLayer holds the components of a single Moirai encoder layer.
type moiraiEncoderLayer[T tensor.Float] struct {
	norm1 *normalization.RMSNorm[T]
	qProj *graph.Parameter[T] // [hiddenDim, hiddenDim]
	kProj *graph.Parameter[T] // [hiddenDim, hiddenDim]
	vProj *graph.Parameter[T] // [hiddenDim, hiddenDim]
	oProj *graph.Parameter[T] // [hiddenDim, hiddenDim]
	norm2 *normalization.RMSNorm[T]
	ffn1  *graph.Parameter[T] // [hiddenDim, 4*hiddenDim]
	ffn2  *graph.Parameter[T] // [4*hiddenDim, hiddenDim]
}

// moiraiNode implements the full Moirai-2 forward pass as a single graph node.
type moiraiNode[T tensor.Float] struct {
	cfg    *MoiraiConfig
	engine compute.Engine[T]
	ops    numeric.Arithmetic[T]

	// Input variate projection with frequency embeddings.
	varProj *timeseries.VariateProjection[T]

	// Transformer encoder layers.
	encoderLayers []moiraiEncoderLayer[T]

	// Final layer norm on encoder output.
	finalNorm *normalization.RMSNorm[T]

	// Output head: [hiddenDim, horizon]
	outputHead *core.Linear[T]
}

func newMoiraiNode[T tensor.Float](
	tensors map[string]*tensor.TensorNumeric[T],
	cfg *MoiraiConfig,
	engine compute.Engine[T],
	ops numeric.Arithmetic[T],
) (*moiraiNode[T], error) {
	// Variate projection: [batch, numVars, inputDim] -> [batch, numVars, hiddenDim]
	varProj, err := timeseries.NewVariateProjection[T](
		"moirai_var_proj", engine, ops,
		cfg.InputDim, cfg.HiddenDim, cfg.NumFreqEmbeddings,
	)
	if err != nil {
		return nil, fmt.Errorf("create variate projection: %w", err)
	}
	loadVariateProjectionWeights(tensors, varProj)

	// Transformer encoder layers.
	layers := make([]moiraiEncoderLayer[T], cfg.NumLayers)
	for i := range cfg.NumLayers {
		prefix := fmt.Sprintf("moirai_enc_%d", i)
		layer, lErr := newMoiraiEncoderLayer[T](prefix, engine, ops, cfg.HiddenDim)
		if lErr != nil {
			return nil, fmt.Errorf("create encoder layer %d: %w", i, lErr)
		}
		loadMoiraiEncoderLayerWeights(tensors, &layer, i)
		layers[i] = layer
	}

	// Final norm.
	finalNorm, err := normalization.NewRMSNorm[T]("moirai_final_norm", engine, ops, cfg.HiddenDim)
	if err != nil {
		return nil, fmt.Errorf("create final norm: %w", err)
	}

	// Output head: projects each variate from hiddenDim to horizon.
	outputHead, err := core.NewLinear[T]("moirai_output_head", engine, ops, cfg.HiddenDim, cfg.Horizon)
	if err != nil {
		return nil, fmt.Errorf("create output head: %w", err)
	}
	loadLinearWeights(tensors, outputHead, "moirai.output_head.weight")

	return &moiraiNode[T]{
		cfg:           cfg,
		engine:        engine,
		ops:           ops,
		varProj:       varProj,
		encoderLayers: layers,
		finalNorm:     finalNorm,
		outputHead:    outputHead,
	}, nil
}

func newMoiraiEncoderLayer[T tensor.Float](
	prefix string,
	engine compute.Engine[T],
	ops numeric.Arithmetic[T],
	hiddenDim int,
) (moiraiEncoderLayer[T], error) {
	var layer moiraiEncoderLayer[T]
	var err error

	layer.norm1, err = normalization.NewRMSNorm[T](prefix+"_norm1", engine, ops, hiddenDim)
	if err != nil {
		return layer, err
	}

	layer.norm2, err = normalization.NewRMSNorm[T](prefix+"_norm2", engine, ops, hiddenDim)
	if err != nil {
		return layer, err
	}

	makeParam := func(name string, rows, cols int) (*graph.Parameter[T], error) {
		data := make([]T, rows*cols)
		scale := T(math.Sqrt(2.0 / float64(rows+cols)))
		for i := range data {
			data[i] = ops.Mul(ops.FromFloat64(float64(i%11-5)*0.1), scale)
		}
		t, tErr := tensor.New[T]([]int{rows, cols}, data)
		if tErr != nil {
			return nil, tErr
		}
		return graph.NewParameter[T](name, t, tensor.New[T])
	}

	layer.qProj, err = makeParam(prefix+"_q_proj", hiddenDim, hiddenDim)
	if err != nil {
		return layer, err
	}
	layer.kProj, err = makeParam(prefix+"_k_proj", hiddenDim, hiddenDim)
	if err != nil {
		return layer, err
	}
	layer.vProj, err = makeParam(prefix+"_v_proj", hiddenDim, hiddenDim)
	if err != nil {
		return layer, err
	}
	layer.oProj, err = makeParam(prefix+"_o_proj", hiddenDim, hiddenDim)
	if err != nil {
		return layer, err
	}

	ffnHidden := 4 * hiddenDim
	layer.ffn1, err = makeParam(prefix+"_ffn1", hiddenDim, ffnHidden)
	if err != nil {
		return layer, err
	}
	layer.ffn2, err = makeParam(prefix+"_ffn2", ffnHidden, hiddenDim)
	if err != nil {
		return layer, err
	}

	return layer, nil
}

// loadVariateProjectionWeights loads variate projection weights from GGUF tensors.
func loadVariateProjectionWeights[T tensor.Float](
	tensors map[string]*tensor.TensorNumeric[T],
	vp *timeseries.VariateProjection[T],
) {
	params := vp.Parameters()
	ggufNames := []string{
		"moirai.var_proj.weight",
		"moirai.var_proj.bias",
		"moirai.var_proj.freq_emb",
	}
	for i, name := range ggufNames {
		if w, ok := tensors[name]; ok && i < len(params) {
			params[i].Value = w
		}
	}
}

// loadMoiraiEncoderLayerWeights loads encoder layer weights from GGUF tensors.
func loadMoiraiEncoderLayerWeights[T tensor.Float](
	tensors map[string]*tensor.TensorNumeric[T],
	layer *moiraiEncoderLayer[T],
	layerIdx int,
) {
	prefix := fmt.Sprintf("moirai.enc.layer.%d.", layerIdx)

	paramMap := map[string]*graph.Parameter[T]{
		"self_attn.q_proj.weight": layer.qProj,
		"self_attn.k_proj.weight": layer.kProj,
		"self_attn.v_proj.weight": layer.vProj,
		"self_attn.o_proj.weight": layer.oProj,
		"mlp.fc1.weight":         layer.ffn1,
		"mlp.fc2.weight":         layer.ffn2,
	}
	for name, param := range paramMap {
		if w, ok := tensors[prefix+name]; ok {
			param.Value = w
		}
	}

	// Load norm weights.
	norm1Params := layer.norm1.Parameters()
	if w, ok := tensors[prefix+"norm1.weight"]; ok && len(norm1Params) > 0 {
		norm1Params[0].Value = w
	}
	norm2Params := layer.norm2.Parameters()
	if w, ok := tensors[prefix+"norm2.weight"]; ok && len(norm2Params) > 0 {
		norm2Params[0].Value = w
	}
}

func (n *moiraiNode[T]) OpType() string { return "Moirai" }

func (n *moiraiNode[T]) Attributes() map[string]interface{} {
	return map[string]interface{}{
		"num_layers":         n.cfg.NumLayers,
		"hidden_dim":         n.cfg.HiddenDim,
		"num_heads":          n.cfg.NumHeads,
		"input_dim":          n.cfg.InputDim,
		"num_freq_embeddings": n.cfg.NumFreqEmbeddings,
		"horizon":            n.cfg.Horizon,
		"num_vars":           n.cfg.NumVars,
		"training":           n.cfg.Training,
	}
}

func (n *moiraiNode[T]) OutputShape() []int {
	return []int{-1, n.cfg.Horizon, n.cfg.NumVars}
}

func (n *moiraiNode[T]) Parameters() []*graph.Parameter[T] {
	var params []*graph.Parameter[T]
	params = append(params, n.varProj.Parameters()...)
	for _, layer := range n.encoderLayers {
		params = append(params, layer.norm1.Parameters()...)
		params = append(params, layer.qProj, layer.kProj, layer.vProj, layer.oProj)
		params = append(params, layer.norm2.Parameters()...)
		params = append(params, layer.ffn1, layer.ffn2)
	}
	params = append(params, n.finalNorm.Parameters()...)
	params = append(params, n.outputHead.Parameters()...)
	return params
}

// Forward processes [batch, numVars, inputDim] input and produces [batch, horizon, numVars].
//
// The forward pass:
//  1. Variate projection with frequency embeddings: [batch, numVars, inputDim] -> [batch, numVars, hiddenDim]
//  2. Transformer encoder (pre-norm self-attention + FFN) over variate tokens
//  3. Output head per variate: [batch, numVars, hiddenDim] -> [batch, numVars, horizon]
//  4. Transpose to [batch, horizon, numVars]
func (n *moiraiNode[T]) Forward(ctx context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if len(inputs) != 1 {
		return nil, fmt.Errorf("Moirai expects 1 input, got %d", len(inputs))
	}
	x := inputs[0]
	shape := x.Shape()
	if len(shape) != 3 {
		return nil, fmt.Errorf("Moirai input must be 3D [batch, numVars, inputDim], got shape %v", shape)
	}

	batch, numVars, inputDim := shape[0], shape[1], shape[2]
	if inputDim != n.cfg.InputDim {
		return nil, fmt.Errorf("Moirai input dim mismatch: got %d, want %d", inputDim, n.cfg.InputDim)
	}

	d := n.cfg.HiddenDim

	// 1. Variate projection: [batch, numVars, inputDim] -> [batch, numVars, hiddenDim]
	hidden, err := n.varProj.Forward(ctx, x)
	if err != nil {
		return nil, fmt.Errorf("variate projection: %w", err)
	}

	// 2. Transformer encoder layers over variate tokens.
	for i, layer := range n.encoderLayers {
		hidden, err = n.encoderForward(ctx, layer, hidden, batch, numVars, d)
		if err != nil {
			return nil, fmt.Errorf("encoder layer %d: %w", i, err)
		}
	}

	// Final norm: [batch, numVars, hiddenDim]
	hidden, err = n.finalNorm.Forward(ctx, hidden)
	if err != nil {
		return nil, fmt.Errorf("final norm: %w", err)
	}

	// 3. Output head: project each variate independently.
	// Reshape to [batch*numVars, hiddenDim] for shared linear projection.
	flat, err := n.engine.Reshape(ctx, hidden, []int{batch * numVars, d})
	if err != nil {
		return nil, fmt.Errorf("reshape for output head: %w", err)
	}
	projected, err := n.outputHead.Forward(ctx, flat)
	if err != nil {
		return nil, fmt.Errorf("output head: %w", err)
	}

	// Reshape to [batch, numVars, horizon].
	projected, err = n.engine.Reshape(ctx, projected, []int{batch, numVars, n.cfg.Horizon})
	if err != nil {
		return nil, fmt.Errorf("reshape projected: %w", err)
	}

	// 4. Transpose to [batch, horizon, numVars].
	output, err := n.engine.Transpose(ctx, projected, []int{0, 2, 1})
	if err != nil {
		return nil, fmt.Errorf("transpose to [batch, horizon, numVars]: %w", err)
	}

	return output, nil
}

// encoderForward runs a single Moirai encoder layer (pre-norm self-attention + FFN).
func (n *moiraiNode[T]) encoderForward(
	ctx context.Context,
	layer moiraiEncoderLayer[T],
	x *tensor.TensorNumeric[T],
	batch, seqLen, dModel int,
) (*tensor.TensorNumeric[T], error) {
	headDim := dModel / n.cfg.NumHeads

	// Pre-norm self-attention.
	normed, err := layer.norm1.Forward(ctx, x)
	if err != nil {
		return nil, fmt.Errorf("norm1: %w", err)
	}

	// Project Q, K, V.
	flat, err := n.engine.Reshape(ctx, normed, []int{batch * seqLen, dModel})
	if err != nil {
		return nil, err
	}

	q, err := n.engine.MatMul(ctx, flat, layer.qProj.Value)
	if err != nil {
		return nil, fmt.Errorf("q projection: %w", err)
	}
	k, err := n.engine.MatMul(ctx, flat, layer.kProj.Value)
	if err != nil {
		return nil, fmt.Errorf("k projection: %w", err)
	}
	v, err := n.engine.MatMul(ctx, flat, layer.vProj.Value)
	if err != nil {
		return nil, fmt.Errorf("v projection: %w", err)
	}

	// Reshape to multi-head: [batch, numHeads, seqLen, headDim]
	q, err = n.reshapeToHeads(ctx, q, batch, seqLen, headDim)
	if err != nil {
		return nil, err
	}
	k, err = n.reshapeToHeads(ctx, k, batch, seqLen, headDim)
	if err != nil {
		return nil, err
	}
	v, err = n.reshapeToHeads(ctx, v, batch, seqLen, headDim)
	if err != nil {
		return nil, err
	}

	// Scaled dot-product attention (no causal mask — encoder).
	kT, err := n.engine.Transpose(ctx, k, []int{0, 1, 3, 2})
	if err != nil {
		return nil, fmt.Errorf("transpose K: %w", err)
	}
	scores, err := n.engine.MatMul(ctx, q, kT)
	if err != nil {
		return nil, fmt.Errorf("attention scores: %w", err)
	}
	scale := n.ops.FromFloat64(1.0 / math.Sqrt(float64(headDim)))
	scores, err = n.engine.MulScalar(ctx, scores, scale)
	if err != nil {
		return nil, fmt.Errorf("scale scores: %w", err)
	}
	attnWeights, err := n.engine.Softmax(ctx, scores, -1)
	if err != nil {
		return nil, fmt.Errorf("softmax: %w", err)
	}

	// attn_output = weights @ V
	attnOut, err := n.engine.MatMul(ctx, attnWeights, v)
	if err != nil {
		return nil, fmt.Errorf("attention output: %w", err)
	}

	// Reshape back: [batch, numHeads, seqLen, headDim] -> [batch*seqLen, dModel]
	attnOut, err = n.engine.Transpose(ctx, attnOut, []int{0, 2, 1, 3})
	if err != nil {
		return nil, err
	}
	attnOut, err = n.engine.Reshape(ctx, attnOut, []int{batch * seqLen, dModel})
	if err != nil {
		return nil, err
	}

	// Output projection.
	attnOut, err = n.engine.MatMul(ctx, attnOut, layer.oProj.Value)
	if err != nil {
		return nil, fmt.Errorf("output projection: %w", err)
	}

	// Reshape to [batch, seqLen, dModel] and add residual.
	attnOut, err = n.engine.Reshape(ctx, attnOut, []int{batch, seqLen, dModel})
	if err != nil {
		return nil, err
	}
	x, err = n.engine.Add(ctx, x, attnOut)
	if err != nil {
		return nil, fmt.Errorf("residual add 1: %w", err)
	}

	// Pre-norm FFN.
	normed, err = layer.norm2.Forward(ctx, x)
	if err != nil {
		return nil, fmt.Errorf("norm2: %w", err)
	}

	ffnFlat, err := n.engine.Reshape(ctx, normed, []int{batch * seqLen, dModel})
	if err != nil {
		return nil, err
	}
	ffnOut, err := n.engine.MatMul(ctx, ffnFlat, layer.ffn1.Value)
	if err != nil {
		return nil, fmt.Errorf("ffn1: %w", err)
	}

	// GELU activation.
	ffnOut, err = n.geluForward(ctx, ffnOut)
	if err != nil {
		return nil, fmt.Errorf("gelu: %w", err)
	}

	ffnOut, err = n.engine.MatMul(ctx, ffnOut, layer.ffn2.Value)
	if err != nil {
		return nil, fmt.Errorf("ffn2: %w", err)
	}

	// Reshape and residual.
	ffnOut, err = n.engine.Reshape(ctx, ffnOut, []int{batch, seqLen, dModel})
	if err != nil {
		return nil, err
	}
	result, err := n.engine.Add(ctx, x, ffnOut)
	if err != nil {
		return nil, fmt.Errorf("residual add 2: %w", err)
	}

	return result, nil
}

// reshapeToHeads reshapes [batch*seqLen, dModel] to [batch, numHeads, seqLen, headDim].
func (n *moiraiNode[T]) reshapeToHeads(ctx context.Context, x *tensor.TensorNumeric[T], batch, seqLen, headDim int) (*tensor.TensorNumeric[T], error) {
	r, err := n.engine.Reshape(ctx, x, []int{batch, seqLen, n.cfg.NumHeads, headDim})
	if err != nil {
		return nil, err
	}
	return n.engine.Transpose(ctx, r, []int{0, 2, 1, 3})
}

// geluForward computes GELU activation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))).
func (n *moiraiNode[T]) geluForward(ctx context.Context, x *tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	x2, err := n.engine.Mul(ctx, x, x)
	if err != nil {
		return nil, err
	}
	x3, err := n.engine.Mul(ctx, x2, x)
	if err != nil {
		return nil, err
	}
	coeff := n.ops.FromFloat64(0.044715)
	term, err := n.engine.MulScalar(ctx, x3, coeff)
	if err != nil {
		return nil, err
	}
	inner, err := n.engine.Add(ctx, x, term)
	if err != nil {
		return nil, err
	}
	sqrtTwoPi := n.ops.FromFloat64(math.Sqrt(2.0 / math.Pi))
	inner, err = n.engine.MulScalar(ctx, inner, sqrtTwoPi)
	if err != nil {
		return nil, err
	}
	tanhOut, err := n.engine.Tanh(ctx, inner)
	if err != nil {
		return nil, err
	}
	one := n.ops.FromFloat64(1.0)
	onePlusTanh, err := n.engine.AddScalar(ctx, tanhOut, one)
	if err != nil {
		return nil, err
	}
	half := n.ops.FromFloat64(0.5)
	halfX, err := n.engine.MulScalar(ctx, x, half)
	if err != nil {
		return nil, err
	}
	return n.engine.Mul(ctx, halfX, onePlusTanh)
}

func (n *moiraiNode[T]) Backward(_ context.Context, _ types.BackwardMode, _ *tensor.TensorNumeric[T], _ ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	return nil, nil
}
