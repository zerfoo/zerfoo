// Package timeseries implements time-series model builders.
package timeseries

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
	"github.com/zerfoo/zerfoo/layers/timeseries"
)

// PatchTSTConfig holds configuration for the PatchTST model.
type PatchTSTConfig struct {
	PatchLen  int // patch size (e.g., 16)
	Stride    int // patch stride (e.g., 8)
	NumLayers int // transformer encoder depth (e.g., 3)
	NumHeads  int // attention heads (e.g., 8)
	DModel    int // model dim (e.g., 128)
	Horizon   int // prediction horizon H
	NumVars   int // number of input features D
}

// BuildPatchTST constructs a PatchTST computation graph.
// Input: [batch, seq_len, num_vars] time series.
// Output: [batch, horizon, num_vars] predictions.
func BuildPatchTST[T tensor.Numeric](cfg PatchTSTConfig, engine compute.Engine[T]) (*graph.Graph[T], error) {
	if cfg.PatchLen <= 0 {
		return nil, fmt.Errorf("PatchLen must be positive, got %d", cfg.PatchLen)
	}
	if cfg.Stride <= 0 {
		return nil, fmt.Errorf("Stride must be positive, got %d", cfg.Stride)
	}
	if cfg.NumLayers <= 0 {
		return nil, fmt.Errorf("NumLayers must be positive, got %d", cfg.NumLayers)
	}
	if cfg.NumHeads <= 0 {
		return nil, fmt.Errorf("NumHeads must be positive, got %d", cfg.NumHeads)
	}
	if cfg.DModel <= 0 {
		return nil, fmt.Errorf("DModel must be positive, got %d", cfg.DModel)
	}
	if cfg.DModel%cfg.NumHeads != 0 {
		return nil, fmt.Errorf("DModel (%d) must be divisible by NumHeads (%d)", cfg.DModel, cfg.NumHeads)
	}
	if cfg.Horizon <= 0 {
		return nil, fmt.Errorf("Horizon must be positive, got %d", cfg.Horizon)
	}
	if cfg.NumVars <= 0 {
		return nil, fmt.Errorf("NumVars must be positive, got %d", cfg.NumVars)
	}

	ops := engine.Ops()

	node, err := newPatchTSTNode[T](cfg, engine, ops)
	if err != nil {
		return nil, fmt.Errorf("create PatchTST node: %w", err)
	}

	builder := graph.NewBuilder[T](engine)
	input := builder.Input([]int{-1, -1, cfg.NumVars})
	builder.AddNode(node, input)

	return builder.Build(node)
}

// patchTSTNode implements the full PatchTST forward pass as a single graph node.
// PatchTST is channel-independent: each variable is patched and encoded separately,
// then a shared projection head maps encoder output to predictions.
type patchTSTNode[T tensor.Numeric] struct {
	cfg    PatchTSTConfig
	engine compute.Engine[T]
	ops    numeric.Arithmetic[T]

	// Per-variable patch embedding (shared weights across variables).
	patchEmbed *timeseries.PatchEmbed[T]

	// Transformer encoder layers.
	encoderLayers []encoderLayer[T]

	// Final layer norm.
	finalNorm *normalization.RMSNorm[T]

	// Projection head: [dModel, horizon] (shared across variables).
	projWeight *graph.Parameter[T]
}

// encoderLayer holds the components of a single transformer encoder layer.
type encoderLayer[T tensor.Numeric] struct {
	norm1 *normalization.RMSNorm[T]
	qProj *graph.Parameter[T] // [dModel, dModel]
	kProj *graph.Parameter[T] // [dModel, dModel]
	vProj *graph.Parameter[T] // [dModel, dModel]
	oProj *graph.Parameter[T] // [dModel, dModel]
	norm2 *normalization.RMSNorm[T]
	ffn1  *graph.Parameter[T] // [dModel, 4*dModel]
	ffn2  *graph.Parameter[T] // [4*dModel, dModel]
}

func newPatchTSTNode[T tensor.Numeric](cfg PatchTSTConfig, engine compute.Engine[T], ops numeric.Arithmetic[T]) (*patchTSTNode[T], error) {
	pe, err := timeseries.NewPatchEmbed[T]("patchtst_patch_embed", engine, ops, cfg.PatchLen, cfg.DModel)
	if err != nil {
		return nil, fmt.Errorf("create patch embed: %w", err)
	}

	layers := make([]encoderLayer[T], cfg.NumLayers)
	for i := range cfg.NumLayers {
		prefix := fmt.Sprintf("patchtst_encoder_%d", i)
		layer, lErr := newEncoderLayer[T](prefix, engine, ops, cfg.DModel)
		if lErr != nil {
			return nil, fmt.Errorf("create encoder layer %d: %w", i, lErr)
		}
		layers[i] = layer
	}

	finalNorm, err := normalization.NewRMSNorm[T]("patchtst_final_norm", engine, ops, cfg.DModel)
	if err != nil {
		return nil, fmt.Errorf("create final norm: %w", err)
	}

	projData := make([]T, cfg.DModel*cfg.Horizon)
	scale := T(math.Sqrt(2.0 / float64(cfg.DModel)))
	for i := range projData {
		projData[i] = ops.Mul(ops.FromFloat64(float64(i%7-3) * 0.1), scale)
	}
	projTensor, err := tensor.New[T]([]int{cfg.DModel, cfg.Horizon}, projData)
	if err != nil {
		return nil, err
	}
	projWeight, err := graph.NewParameter[T]("patchtst_proj_weight", projTensor, tensor.New[T])
	if err != nil {
		return nil, err
	}

	return &patchTSTNode[T]{
		cfg:           cfg,
		engine:        engine,
		ops:           ops,
		patchEmbed:    pe,
		encoderLayers: layers,
		finalNorm:     finalNorm,
		projWeight:    projWeight,
	}, nil
}

func newEncoderLayer[T tensor.Numeric](prefix string, engine compute.Engine[T], ops numeric.Arithmetic[T], dModel int) (encoderLayer[T], error) {
	var layer encoderLayer[T]
	var err error

	layer.norm1, err = normalization.NewRMSNorm[T](prefix+"_norm1", engine, ops, dModel)
	if err != nil {
		return layer, err
	}

	layer.norm2, err = normalization.NewRMSNorm[T](prefix+"_norm2", engine, ops, dModel)
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

	layer.qProj, err = makeParam(prefix+"_q_proj", dModel, dModel)
	if err != nil {
		return layer, err
	}
	layer.kProj, err = makeParam(prefix+"_k_proj", dModel, dModel)
	if err != nil {
		return layer, err
	}
	layer.vProj, err = makeParam(prefix+"_v_proj", dModel, dModel)
	if err != nil {
		return layer, err
	}
	layer.oProj, err = makeParam(prefix+"_o_proj", dModel, dModel)
	if err != nil {
		return layer, err
	}

	ffnHidden := 4 * dModel
	layer.ffn1, err = makeParam(prefix+"_ffn1", dModel, ffnHidden)
	if err != nil {
		return layer, err
	}
	layer.ffn2, err = makeParam(prefix+"_ffn2", ffnHidden, dModel)
	if err != nil {
		return layer, err
	}

	return layer, nil
}

func (n *patchTSTNode[T]) OpType() string { return "PatchTST" }

func (n *patchTSTNode[T]) Attributes() map[string]interface{} {
	return map[string]interface{}{
		"patch_len":  n.cfg.PatchLen,
		"stride":     n.cfg.Stride,
		"num_layers": n.cfg.NumLayers,
		"num_heads":  n.cfg.NumHeads,
		"d_model":    n.cfg.DModel,
		"horizon":    n.cfg.Horizon,
		"num_vars":   n.cfg.NumVars,
	}
}

func (n *patchTSTNode[T]) OutputShape() []int {
	return []int{-1, n.cfg.Horizon, n.cfg.NumVars}
}

func (n *patchTSTNode[T]) Parameters() []*graph.Parameter[T] {
	var params []*graph.Parameter[T]
	params = append(params, n.patchEmbed.Parameters()...)
	for _, layer := range n.encoderLayers {
		params = append(params, layer.norm1.Parameters()...)
		params = append(params, layer.qProj, layer.kProj, layer.vProj, layer.oProj)
		params = append(params, layer.norm2.Parameters()...)
		params = append(params, layer.ffn1, layer.ffn2)
	}
	params = append(params, n.finalNorm.Parameters()...)
	params = append(params, n.projWeight)
	return params
}

// Forward processes [batch, seq_len, num_vars] input and produces [batch, horizon, num_vars].
//
// PatchTST is channel-independent: each variable is processed through the same
// patch embedding and transformer encoder independently, then the outputs are
// combined via a projection head.
func (n *patchTSTNode[T]) Forward(ctx context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if len(inputs) != 1 {
		return nil, fmt.Errorf("PatchTST expects 1 input, got %d", len(inputs))
	}
	x := inputs[0]
	shape := x.Shape()
	if len(shape) != 3 {
		return nil, fmt.Errorf("PatchTST input must be 3D [batch, seq_len, num_vars], got shape %v", shape)
	}

	batch, seqLen, numVars := shape[0], shape[1], shape[2]

	// Process each variable independently through patch embed + encoder.
	// PatchEmbed expects [batch, seq_len] per variable.
	varOutputs := make([]*tensor.TensorNumeric[T], numVars)
	for v := range numVars {
		// Extract variable v: [batch, seq_len]
		varSlice, err := n.extractVariable(ctx, x, batch, seqLen, numVars, v)
		if err != nil {
			return nil, fmt.Errorf("extract variable %d: %w", v, err)
		}

		// Patch embed: [batch, seq_len] -> [batch, num_patches, d_model]
		embedded, err := n.patchEmbed.Forward(ctx, varSlice)
		if err != nil {
			return nil, fmt.Errorf("patch embed variable %d: %w", v, err)
		}

		// Run through transformer encoder layers.
		hidden := embedded
		for i, layer := range n.encoderLayers {
			hidden, err = n.encoderForward(ctx, layer, hidden)
			if err != nil {
				return nil, fmt.Errorf("encoder layer %d variable %d: %w", i, v, err)
			}
		}

		// Final norm.
		hidden, err = n.finalNorm.Forward(ctx, hidden)
		if err != nil {
			return nil, fmt.Errorf("final norm variable %d: %w", v, err)
		}

		// Mean pool over patches: [batch, num_patches, d_model] -> [batch, d_model]
		hidden, err = n.engine.ReduceMean(ctx, hidden, 1, false)
		if err != nil {
			return nil, fmt.Errorf("mean pool variable %d: %w", v, err)
		}

		varOutputs[v] = hidden
	}

	// Stack variable outputs: each is [batch, d_model] -> concatenate to [batch, numVars * d_model]
	stacked, err := n.engine.Concat(ctx, varOutputs, 1)
	if err != nil {
		return nil, fmt.Errorf("concat variable outputs: %w", err)
	}

	// Channel-independent projection: reshape to [batch*numVars, d_model],
	// project each variable independently to horizon, then reshape and transpose.
	projected, err := n.engine.Reshape(ctx, stacked, []int{batch * numVars, n.cfg.DModel})
	if err != nil {
		return nil, fmt.Errorf("reshape for projection: %w", err)
	}

	// Project: [batch*numVars, d_model] @ [d_model, horizon] -> [batch*numVars, horizon]
	projected, err = n.engine.MatMul(ctx, projected, n.projWeight.Value)
	if err != nil {
		return nil, fmt.Errorf("projection matmul: %w", err)
	}

	// Reshape to [batch, numVars, horizon].
	projected, err = n.engine.Reshape(ctx, projected, []int{batch, numVars, n.cfg.Horizon})
	if err != nil {
		return nil, fmt.Errorf("reshape projected: %w", err)
	}

	// Transpose to [batch, horizon, numVars].
	projected, err = n.engine.Transpose(ctx, projected, []int{0, 2, 1})
	if err != nil {
		return nil, fmt.Errorf("transpose to [batch, horizon, numVars]: %w", err)
	}

	// Result: [batch, horizon, numVars]
	return projected, nil
}

// extractVariable extracts variable v from [batch, seq_len, num_vars] as [batch, seq_len].
func (n *patchTSTNode[T]) extractVariable(ctx context.Context, x *tensor.TensorNumeric[T], batch, seqLen, numVars, v int) (*tensor.TensorNumeric[T], error) {
	data := x.Data()
	out := make([]T, batch*seqLen)
	for b := range batch {
		for s := range seqLen {
			out[b*seqLen+s] = data[b*seqLen*numVars+s*numVars+v]
		}
	}
	return tensor.New[T]([]int{batch, seqLen}, out)
}

// encoderForward runs a single transformer encoder layer.
// Input: [batch, num_patches, d_model]
// Output: [batch, num_patches, d_model]
func (n *patchTSTNode[T]) encoderForward(ctx context.Context, layer encoderLayer[T], x *tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	shape := x.Shape()
	batch, numPatches, dModel := shape[0], shape[1], shape[2]
	headDim := dModel / n.cfg.NumHeads

	// Pre-norm self-attention.
	normed, err := layer.norm1.Forward(ctx, x)
	if err != nil {
		return nil, fmt.Errorf("norm1: %w", err)
	}

	// Project Q, K, V: [batch * num_patches, d_model] @ [d_model, d_model]
	flat, err := n.engine.Reshape(ctx, normed, []int{batch * numPatches, dModel})
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

	// Reshape to multi-head: [batch, numHeads, num_patches, headDim]
	q, err = n.reshapeToHeads(ctx, q, batch, numPatches, headDim)
	if err != nil {
		return nil, err
	}
	k, err = n.reshapeToHeads(ctx, k, batch, numPatches, headDim)
	if err != nil {
		return nil, err
	}
	v, err = n.reshapeToHeads(ctx, v, batch, numPatches, headDim)
	if err != nil {
		return nil, err
	}

	// Scaled dot-product attention (no causal mask for encoder).
	// scores = Q @ K^T / sqrt(headDim)
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

	// attn_output = weights @ V: [batch, numHeads, num_patches, headDim]
	attnOut, err := n.engine.MatMul(ctx, attnWeights, v)
	if err != nil {
		return nil, fmt.Errorf("attention output: %w", err)
	}

	// Reshape back: [batch, numHeads, num_patches, headDim] -> [batch * num_patches, d_model]
	attnOut, err = n.engine.Transpose(ctx, attnOut, []int{0, 2, 1, 3})
	if err != nil {
		return nil, err
	}
	attnOut, err = n.engine.Reshape(ctx, attnOut, []int{batch * numPatches, dModel})
	if err != nil {
		return nil, err
	}

	// Output projection.
	attnOut, err = n.engine.MatMul(ctx, attnOut, layer.oProj.Value)
	if err != nil {
		return nil, fmt.Errorf("output projection: %w", err)
	}

	// Reshape to [batch, num_patches, d_model] and add residual.
	attnOut, err = n.engine.Reshape(ctx, attnOut, []int{batch, numPatches, dModel})
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

	// FFN: Linear -> GELU approx -> Linear.
	ffnFlat, err := n.engine.Reshape(ctx, normed, []int{batch * numPatches, dModel})
	if err != nil {
		return nil, err
	}
	ffnOut, err := n.engine.MatMul(ctx, ffnFlat, layer.ffn1.Value)
	if err != nil {
		return nil, fmt.Errorf("ffn1: %w", err)
	}

	// GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
	ffnOut, err = n.geluForward(ctx, ffnOut)
	if err != nil {
		return nil, fmt.Errorf("gelu: %w", err)
	}

	ffnOut, err = n.engine.MatMul(ctx, ffnOut, layer.ffn2.Value)
	if err != nil {
		return nil, fmt.Errorf("ffn2: %w", err)
	}

	// Reshape and residual.
	ffnOut, err = n.engine.Reshape(ctx, ffnOut, []int{batch, numPatches, dModel})
	if err != nil {
		return nil, err
	}
	result, err := n.engine.Add(ctx, x, ffnOut)
	if err != nil {
		return nil, fmt.Errorf("residual add 2: %w", err)
	}

	return result, nil
}

// reshapeToHeads reshapes [batch * seqLen, d_model] to [batch, numHeads, seqLen, headDim].
func (n *patchTSTNode[T]) reshapeToHeads(ctx context.Context, x *tensor.TensorNumeric[T], batch, seqLen, headDim int) (*tensor.TensorNumeric[T], error) {
	// [batch * seqLen, d_model] -> [batch, seqLen, numHeads, headDim]
	r, err := n.engine.Reshape(ctx, x, []int{batch, seqLen, n.cfg.NumHeads, headDim})
	if err != nil {
		return nil, err
	}
	// [batch, seqLen, numHeads, headDim] -> [batch, numHeads, seqLen, headDim]
	return n.engine.Transpose(ctx, r, []int{0, 2, 1, 3})
}

// geluForward computes GELU activation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))).
func (n *patchTSTNode[T]) geluForward(ctx context.Context, x *tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	// x^2
	x2, err := n.engine.Mul(ctx, x, x)
	if err != nil {
		return nil, err
	}
	// x^3
	x3, err := n.engine.Mul(ctx, x2, x)
	if err != nil {
		return nil, err
	}
	// 0.044715 * x^3
	coeff := n.ops.FromFloat64(0.044715)
	term, err := n.engine.MulScalar(ctx, x3, coeff)
	if err != nil {
		return nil, err
	}
	// x + 0.044715 * x^3
	inner, err := n.engine.Add(ctx, x, term)
	if err != nil {
		return nil, err
	}
	// sqrt(2/pi) * (...)
	sqrtTwoPi := n.ops.FromFloat64(math.Sqrt(2.0 / math.Pi))
	inner, err = n.engine.MulScalar(ctx, inner, sqrtTwoPi)
	if err != nil {
		return nil, err
	}
	// tanh(...)
	tanhOut, err := n.engine.Tanh(ctx, inner)
	if err != nil {
		return nil, err
	}
	// 1 + tanh(...)
	one := n.ops.FromFloat64(1.0)
	onePlusTanh, err := n.engine.AddScalar(ctx, tanhOut, one)
	if err != nil {
		return nil, err
	}
	// 0.5 * x
	half := n.ops.FromFloat64(0.5)
	halfX, err := n.engine.MulScalar(ctx, x, half)
	if err != nil {
		return nil, err
	}
	// 0.5 * x * (1 + tanh(...))
	return n.engine.Mul(ctx, halfX, onePlusTanh)
}

func (n *patchTSTNode[T]) Backward(_ context.Context, _ types.BackwardMode, _ *tensor.TensorNumeric[T], _ ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	return nil, nil
}
