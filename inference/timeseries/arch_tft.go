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
	tslayers "github.com/zerfoo/zerfoo/layers/timeseries"
)

// TFTConfig holds configuration for the Temporal Fusion Transformer.
type TFTConfig struct {
	NumStaticFeatures   int       // static covariates (e.g. asset metadata)
	NumTemporalFeatures int       // time-varying inputs
	HiddenDim           int       // d_model
	NumHeads            int       // attention heads
	NumLSTMLayers       int       // LSTM encoder layers
	HorizonLen          int       // forecast steps H
	Quantiles           []float32 // e.g. [0.1, 0.5, 0.9]
}

// BuildTFT constructs a TFT computation graph.
// Input 0: [batch, seq_len, num_temporal_features] temporal features.
// Input 1: [batch, num_static_features] static covariates.
// Output: [batch, H, len(quantiles)] quantile predictions.
func BuildTFT[T tensor.Numeric](cfg TFTConfig, engine compute.Engine[T]) (*graph.Graph[T], error) {
	if cfg.NumStaticFeatures <= 0 {
		return nil, fmt.Errorf("NumStaticFeatures must be positive, got %d", cfg.NumStaticFeatures)
	}
	if cfg.NumTemporalFeatures <= 0 {
		return nil, fmt.Errorf("NumTemporalFeatures must be positive, got %d", cfg.NumTemporalFeatures)
	}
	if cfg.HiddenDim <= 0 {
		return nil, fmt.Errorf("HiddenDim must be positive, got %d", cfg.HiddenDim)
	}
	if cfg.NumHeads <= 0 {
		return nil, fmt.Errorf("NumHeads must be positive, got %d", cfg.NumHeads)
	}
	if cfg.HiddenDim%cfg.NumHeads != 0 {
		return nil, fmt.Errorf("HiddenDim (%d) must be divisible by NumHeads (%d)", cfg.HiddenDim, cfg.NumHeads)
	}
	if cfg.NumLSTMLayers <= 0 {
		return nil, fmt.Errorf("NumLSTMLayers must be positive, got %d", cfg.NumLSTMLayers)
	}
	if cfg.HorizonLen <= 0 {
		return nil, fmt.Errorf("HorizonLen must be positive, got %d", cfg.HorizonLen)
	}
	if len(cfg.Quantiles) == 0 {
		return nil, fmt.Errorf("Quantiles must be non-empty")
	}

	ops := engine.Ops()

	node, err := newTFTNode[T](cfg, engine, ops)
	if err != nil {
		return nil, fmt.Errorf("create TFT node: %w", err)
	}

	builder := graph.NewBuilder[T](engine)
	temporalInput := builder.Input([]int{-1, -1, cfg.NumTemporalFeatures})
	staticInput := builder.Input([]int{-1, cfg.NumStaticFeatures})
	builder.AddNode(node, temporalInput, staticInput)

	return builder.Build(node)
}

// lstmLayer holds parameters for a single LSTM cell applied across a sequence.
type lstmLayer[T tensor.Numeric] struct {
	// Input gate: i = sigmoid(xt @ Wi + h @ Ui + bi)
	wi *graph.Parameter[T] // [inputDim, hiddenDim]
	ui *graph.Parameter[T] // [hiddenDim, hiddenDim]
	bi *graph.Parameter[T] // [1, hiddenDim]

	// Forget gate: f = sigmoid(xt @ Wf + h @ Uf + bf)
	wf *graph.Parameter[T]
	uf *graph.Parameter[T]
	bf *graph.Parameter[T]

	// Cell gate: g = tanh(xt @ Wg + h @ Ug + bg)
	wg *graph.Parameter[T]
	ug *graph.Parameter[T]
	bg *graph.Parameter[T]

	// Output gate: o = sigmoid(xt @ Wo + h @ Uo + bo)
	wo *graph.Parameter[T]
	uo *graph.Parameter[T]
	bo *graph.Parameter[T]

	inputDim  int
	hiddenDim int
}

func newLSTMLayer[T tensor.Numeric](prefix string, inputDim, hiddenDim int) (*lstmLayer[T], error) {
	makeGateParams := func(suffix string, inDim int) (*graph.Parameter[T], *graph.Parameter[T], *graph.Parameter[T], error) {
		w, err := newParam[T](prefix+"_w"+suffix, inDim, hiddenDim)
		if err != nil {
			return nil, nil, nil, err
		}
		u, err := newParam[T](prefix+"_u"+suffix, hiddenDim, hiddenDim)
		if err != nil {
			return nil, nil, nil, err
		}
		bData := make([]T, hiddenDim)
		bTensor, err := tensor.New[T]([]int{1, hiddenDim}, bData)
		if err != nil {
			return nil, nil, nil, err
		}
		b, err := graph.NewParameter[T](prefix+"_b"+suffix, bTensor, tensor.New[T])
		if err != nil {
			return nil, nil, nil, err
		}
		return w, u, b, nil
	}

	wi, ui, bi, err := makeGateParams("i", inputDim)
	if err != nil {
		return nil, err
	}
	wf, uf, bf, err := makeGateParams("f", inputDim)
	if err != nil {
		return nil, err
	}
	wg, ug, bg, err := makeGateParams("g", inputDim)
	if err != nil {
		return nil, err
	}
	wo, uo, bo, err := makeGateParams("o", inputDim)
	if err != nil {
		return nil, err
	}

	// Initialize forget gate bias to 1 for better gradient flow.
	bfData := bf.Value.Data()
	for i := range bfData {
		bfData[i] = 1
	}

	return &lstmLayer[T]{
		wi: wi, ui: ui, bi: bi,
		wf: wf, uf: uf, bf: bf,
		wg: wg, ug: ug, bg: bg,
		wo: wo, uo: uo, bo: bo,
		inputDim:  inputDim,
		hiddenDim: hiddenDim,
	}, nil
}

func (l *lstmLayer[T]) parameters() []*graph.Parameter[T] {
	return []*graph.Parameter[T]{
		l.wi, l.ui, l.bi,
		l.wf, l.uf, l.bf,
		l.wg, l.ug, l.bg,
		l.wo, l.uo, l.bo,
	}
}

// lstmForward processes a sequence through the LSTM layer and returns all hidden states.
// input: [batch, seqLen, inputDim], output: [batch, seqLen, hiddenDim]
func lstmForward[T tensor.Numeric](
	ctx context.Context,
	engine compute.Engine[T],
	ops numeric.Arithmetic[T],
	l *lstmLayer[T],
	input *tensor.TensorNumeric[T],
) (*tensor.TensorNumeric[T], error) {
	shape := input.Shape()
	batch := shape[0]
	seqLen := shape[1]

	// Initialize hidden state and cell state to zeros.
	h, err := tensor.New[T]([]int{batch, l.hiddenDim}, make([]T, batch*l.hiddenDim))
	if err != nil {
		return nil, err
	}
	c, err := tensor.New[T]([]int{batch, l.hiddenDim}, make([]T, batch*l.hiddenDim))
	if err != nil {
		return nil, err
	}

	// Collect all hidden states for output.
	hiddenStates := make([]*tensor.TensorNumeric[T], seqLen)

	for t := 0; t < seqLen; t++ {
		// Extract timestep t: [batch, inputDim]
		xt, err := extractTimestep(input, t, batch, l.inputDim)
		if err != nil {
			return nil, fmt.Errorf("extract timestep %d: %w", t, err)
		}

		// Input gate: i = sigmoid(xt @ Wi + h @ Ui + bi)
		xWi, err := engine.MatMul(ctx, xt, l.wi.Value)
		if err != nil {
			return nil, err
		}
		hUi, err := engine.MatMul(ctx, h, l.ui.Value)
		if err != nil {
			return nil, err
		}
		iPre, err := engine.Add(ctx, xWi, hUi)
		if err != nil {
			return nil, err
		}
		iPre, err = engine.Add(ctx, iPre, l.bi.Value)
		if err != nil {
			return nil, err
		}
		iGate, err := sigmoidFn(ctx, engine, ops, iPre)
		if err != nil {
			return nil, err
		}

		// Forget gate: f = sigmoid(xt @ Wf + h @ Uf + bf)
		xWf, err := engine.MatMul(ctx, xt, l.wf.Value)
		if err != nil {
			return nil, err
		}
		hUf, err := engine.MatMul(ctx, h, l.uf.Value)
		if err != nil {
			return nil, err
		}
		fPre, err := engine.Add(ctx, xWf, hUf)
		if err != nil {
			return nil, err
		}
		fPre, err = engine.Add(ctx, fPre, l.bf.Value)
		if err != nil {
			return nil, err
		}
		fGate, err := sigmoidFn(ctx, engine, ops, fPre)
		if err != nil {
			return nil, err
		}

		// Cell gate: g = tanh(xt @ Wg + h @ Ug + bg)
		xWg, err := engine.MatMul(ctx, xt, l.wg.Value)
		if err != nil {
			return nil, err
		}
		hUg, err := engine.MatMul(ctx, h, l.ug.Value)
		if err != nil {
			return nil, err
		}
		gPre, err := engine.Add(ctx, xWg, hUg)
		if err != nil {
			return nil, err
		}
		gPre, err = engine.Add(ctx, gPre, l.bg.Value)
		if err != nil {
			return nil, err
		}
		gGate, err := engine.UnaryOp(ctx, gPre, ops.Tanh)
		if err != nil {
			return nil, err
		}

		// Output gate: o = sigmoid(xt @ Wo + h @ Uo + bo)
		xWo, err := engine.MatMul(ctx, xt, l.wo.Value)
		if err != nil {
			return nil, err
		}
		hUo, err := engine.MatMul(ctx, h, l.uo.Value)
		if err != nil {
			return nil, err
		}
		oPre, err := engine.Add(ctx, xWo, hUo)
		if err != nil {
			return nil, err
		}
		oPre, err = engine.Add(ctx, oPre, l.bo.Value)
		if err != nil {
			return nil, err
		}
		oGate, err := sigmoidFn(ctx, engine, ops, oPre)
		if err != nil {
			return nil, err
		}

		// Cell state: c = f * c_prev + i * g
		fc, err := engine.Mul(ctx, fGate, c)
		if err != nil {
			return nil, err
		}
		ig, err := engine.Mul(ctx, iGate, gGate)
		if err != nil {
			return nil, err
		}
		c, err = engine.Add(ctx, fc, ig)
		if err != nil {
			return nil, err
		}

		// Hidden state: h = o * tanh(c)
		tanhC, err := engine.UnaryOp(ctx, c, ops.Tanh)
		if err != nil {
			return nil, err
		}
		h, err = engine.Mul(ctx, oGate, tanhC)
		if err != nil {
			return nil, err
		}

		hiddenStates[t] = h
	}

	// Stack hidden states: [batch, seqLen, hiddenDim]
	// Reshape each [batch, hiddenDim] to [batch, 1, hiddenDim], then concat along axis 1.
	for t := 0; t < seqLen; t++ {
		hiddenStates[t], err = engine.Reshape(ctx, hiddenStates[t], []int{batch, 1, l.hiddenDim})
		if err != nil {
			return nil, fmt.Errorf("reshape hidden state %d: %w", t, err)
		}
	}
	return engine.Concat(ctx, hiddenStates, 1)
}

// tftNode implements the full TFT forward pass as a single graph node.
type tftNode[T tensor.Numeric] struct {
	cfg    TFTConfig
	engine compute.Engine[T]
	ops    numeric.Arithmetic[T]

	// Static covariate encoder.
	staticProj *graph.Parameter[T] // [numStatic, hiddenDim]
	staticGRN  *tslayers.GRN[T]

	// Temporal VSN.
	temporalVSN *tslayers.VSN[T]

	// Stacked LSTM encoder layers.
	lstmLayers []*lstmLayer[T]

	// Multi-head attention over LSTM outputs.
	attnQProj *graph.Parameter[T] // [hiddenDim, hiddenDim]
	attnKProj *graph.Parameter[T] // [hiddenDim, hiddenDim]
	attnVProj *graph.Parameter[T] // [hiddenDim, hiddenDim]
	attnOProj *graph.Parameter[T] // [hiddenDim, hiddenDim]

	// Quantile projection head.
	quantileProj *graph.Parameter[T] // [hiddenDim, horizonLen * numQuantiles]
}

func newTFTNode[T tensor.Numeric](cfg TFTConfig, engine compute.Engine[T], ops numeric.Arithmetic[T]) (*tftNode[T], error) {
	numQuantiles := len(cfg.Quantiles)

	// Static covariate encoder: Linear(numStatic -> hiddenDim) + GRN.
	staticProj, err := newParam[T]("tft_static_proj", cfg.NumStaticFeatures, cfg.HiddenDim)
	if err != nil {
		return nil, err
	}

	staticGRN, err := tslayers.NewGRN[T]("tft_static_grn", engine, ops, cfg.HiddenDim, cfg.HiddenDim, cfg.HiddenDim)
	if err != nil {
		return nil, err
	}

	// Temporal VSN: each temporal feature is treated as a 1-dim variable.
	temporalVSN, err := tslayers.NewVSN[T]("tft_temporal_vsn", engine, ops, cfg.NumTemporalFeatures, 1, cfg.HiddenDim)
	if err != nil {
		return nil, err
	}

	// LSTM encoder layers.
	lstmLayers := make([]*lstmLayer[T], cfg.NumLSTMLayers)
	for i := range lstmLayers {
		inDim := cfg.HiddenDim
		lstmLayers[i], err = newLSTMLayer[T](fmt.Sprintf("tft_lstm_%d", i), inDim, cfg.HiddenDim)
		if err != nil {
			return nil, fmt.Errorf("create LSTM layer %d: %w", i, err)
		}
	}

	// Multi-head attention parameters.
	attnQProj, err := newParam[T]("tft_attn_q", cfg.HiddenDim, cfg.HiddenDim)
	if err != nil {
		return nil, err
	}
	attnKProj, err := newParam[T]("tft_attn_k", cfg.HiddenDim, cfg.HiddenDim)
	if err != nil {
		return nil, err
	}
	attnVProj, err := newParam[T]("tft_attn_v", cfg.HiddenDim, cfg.HiddenDim)
	if err != nil {
		return nil, err
	}
	attnOProj, err := newParam[T]("tft_attn_o", cfg.HiddenDim, cfg.HiddenDim)
	if err != nil {
		return nil, err
	}

	// Quantile projection: [hiddenDim, horizonLen * numQuantiles]
	quantileProj, err := newParam[T]("tft_quantile_proj", cfg.HiddenDim, cfg.HorizonLen*numQuantiles)
	if err != nil {
		return nil, err
	}

	return &tftNode[T]{
		cfg:          cfg,
		engine:       engine,
		ops:          ops,
		staticProj:   staticProj,
		staticGRN:    staticGRN,
		temporalVSN:  temporalVSN,
		lstmLayers:   lstmLayers,
		attnQProj:    attnQProj,
		attnKProj:    attnKProj,
		attnVProj:    attnVProj,
		attnOProj:    attnOProj,
		quantileProj: quantileProj,
	}, nil
}

func (n *tftNode[T]) OpType() string { return "TFT" }

func (n *tftNode[T]) Attributes() map[string]interface{} {
	return map[string]interface{}{
		"num_static_features":  n.cfg.NumStaticFeatures,
		"num_temporal_features": n.cfg.NumTemporalFeatures,
		"hidden_dim":           n.cfg.HiddenDim,
		"num_heads":            n.cfg.NumHeads,
		"num_lstm_layers":      n.cfg.NumLSTMLayers,
		"horizon_len":          n.cfg.HorizonLen,
		"num_quantiles":        len(n.cfg.Quantiles),
	}
}

func (n *tftNode[T]) OutputShape() []int {
	return []int{-1, n.cfg.HorizonLen, len(n.cfg.Quantiles)}
}

func (n *tftNode[T]) Parameters() []*graph.Parameter[T] {
	var params []*graph.Parameter[T]
	params = append(params, n.staticProj)
	params = append(params, n.staticGRN.Parameters()...)
	params = append(params, n.temporalVSN.Parameters()...)
	for _, l := range n.lstmLayers {
		params = append(params, l.parameters()...)
	}
	params = append(params, n.attnQProj, n.attnKProj, n.attnVProj, n.attnOProj)
	params = append(params, n.quantileProj)
	return params
}

// Forward processes temporal [batch, seq_len, num_temporal_features] and
// static [batch, num_static_features] inputs, producing [batch, H, num_quantiles].
func (n *tftNode[T]) Forward(ctx context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if len(inputs) != 2 {
		return nil, fmt.Errorf("TFT expects 2 inputs (temporal, static), got %d", len(inputs))
	}
	temporal := inputs[0] // [batch, seqLen, numTemporalFeatures]
	static := inputs[1]   // [batch, numStaticFeatures]

	tShape := temporal.Shape()
	if len(tShape) != 3 {
		return nil, fmt.Errorf("temporal input must be 3D [batch, seq_len, features], got %v", tShape)
	}
	sShape := static.Shape()
	if len(sShape) != 2 {
		return nil, fmt.Errorf("static input must be 2D [batch, features], got %v", sShape)
	}

	batch, seqLen := tShape[0], tShape[1]

	// 1. Static covariate encoder: Linear(numStatic -> hiddenDim) + GRN.
	// [batch, numStatic] @ [numStatic, hiddenDim] -> [batch, hiddenDim]
	staticEmb, err := n.engine.MatMul(ctx, static, n.staticProj.Value)
	if err != nil {
		return nil, fmt.Errorf("static projection: %w", err)
	}
	staticEmb, err = n.staticGRN.Forward(ctx, staticEmb)
	if err != nil {
		return nil, fmt.Errorf("static GRN: %w", err)
	}

	// 2. Temporal VSN: apply per-timestep variable selection.
	// Process each timestep through the VSN to get [batch, seqLen, hiddenDim].
	//
	// Build one-hot column selectors for per-variable extraction.
	// selector[v] is [numTemporalFeatures, 1] with a 1 at row v.
	selectors := make([]*tensor.TensorNumeric[T], n.cfg.NumTemporalFeatures)
	for v := 0; v < n.cfg.NumTemporalFeatures; v++ {
		col := make([]T, n.cfg.NumTemporalFeatures)
		col[v] = n.ops.One()
		selectors[v], err = tensor.New[T]([]int{n.cfg.NumTemporalFeatures, 1}, col)
		if err != nil {
			return nil, fmt.Errorf("create selector %d: %w", v, err)
		}
	}

	timestepOutputs := make([]*tensor.TensorNumeric[T], seqLen)
	for t := 0; t < seqLen; t++ {
		// Extract timestep t: [batch, numTemporalFeatures]
		xt, err := extractTimestep(temporal, t, batch, n.cfg.NumTemporalFeatures)
		if err != nil {
			return nil, fmt.Errorf("extract temporal timestep %d: %w", t, err)
		}

		// Split into per-variable inputs via MatMul with one-hot selectors: each [batch, 1]
		varInputs := make([]*tensor.TensorNumeric[T], n.cfg.NumTemporalFeatures)
		for v := 0; v < n.cfg.NumTemporalFeatures; v++ {
			varInputs[v], err = n.engine.MatMul(ctx, xt, selectors[v])
			if err != nil {
				return nil, fmt.Errorf("extract variable %d timestep %d: %w", v, t, err)
			}
		}

		// VSN forward: [batch, hiddenDim]
		vsnOut, _, err := n.temporalVSN.Forward(ctx, varInputs)
		if err != nil {
			return nil, fmt.Errorf("temporal VSN timestep %d: %w", t, err)
		}

		// Add static context to each timestep.
		vsnOut, err = n.engine.Add(ctx, vsnOut, staticEmb)
		if err != nil {
			return nil, fmt.Errorf("add static context timestep %d: %w", t, err)
		}

		// Reshape [batch, hiddenDim] -> [batch, 1, hiddenDim] for stacking.
		timestepOutputs[t], err = n.engine.Reshape(ctx, vsnOut, []int{batch, 1, n.cfg.HiddenDim})
		if err != nil {
			return nil, fmt.Errorf("reshape VSN output timestep %d: %w", t, err)
		}
	}

	// Concat along axis 1: [batch, seqLen, hiddenDim]
	enriched, err := n.engine.Concat(ctx, timestepOutputs, 1)
	if err != nil {
		return nil, fmt.Errorf("concat temporal outputs: %w", err)
	}

	// 3. LSTM encoder over enriched temporal features.
	lstmOut := enriched
	for i, lstm := range n.lstmLayers {
		lstmOut, err = lstmForward(ctx, n.engine, n.ops, lstm, lstmOut)
		if err != nil {
			return nil, fmt.Errorf("LSTM layer %d: %w", i, err)
		}
	}

	// 4. Multi-head self-attention over LSTM outputs.
	// lstmOut: [batch, seqLen, hiddenDim]
	attnOut, err := n.selfAttention(ctx, lstmOut, batch, seqLen)
	if err != nil {
		return nil, fmt.Errorf("self-attention: %w", err)
	}

	// Residual connection.
	attnOut, err = n.engine.Add(ctx, lstmOut, attnOut)
	if err != nil {
		return nil, fmt.Errorf("attention residual: %w", err)
	}

	// 5. Mean pool over sequence: [batch, seqLen, hiddenDim] -> [batch, hiddenDim]
	pooled, err := n.engine.ReduceMean(ctx, attnOut, 1, false)
	if err != nil {
		return nil, fmt.Errorf("mean pool: %w", err)
	}

	// 6. Quantile projection: [batch, hiddenDim] @ [hiddenDim, H*Q] -> [batch, H*Q]
	numQuantiles := len(n.cfg.Quantiles)
	projected, err := n.engine.MatMul(ctx, pooled, n.quantileProj.Value)
	if err != nil {
		return nil, fmt.Errorf("quantile projection: %w", err)
	}

	// Reshape to [batch, H, Q].
	result, err := n.engine.Reshape(ctx, projected, []int{batch, n.cfg.HorizonLen, numQuantiles})
	if err != nil {
		return nil, fmt.Errorf("reshape quantile output: %w", err)
	}

	return result, nil
}

// selfAttention computes multi-head self-attention over [batch, seqLen, hiddenDim].
func (n *tftNode[T]) selfAttention(ctx context.Context, x *tensor.TensorNumeric[T], batch, seqLen int) (*tensor.TensorNumeric[T], error) {
	dModel := n.cfg.HiddenDim
	headDim := dModel / n.cfg.NumHeads

	// Flatten to [batch*seqLen, hiddenDim] for projection.
	flat, err := n.engine.Reshape(ctx, x, []int{batch * seqLen, dModel})
	if err != nil {
		return nil, err
	}

	q, err := n.engine.MatMul(ctx, flat, n.attnQProj.Value)
	if err != nil {
		return nil, fmt.Errorf("q projection: %w", err)
	}
	k, err := n.engine.MatMul(ctx, flat, n.attnKProj.Value)
	if err != nil {
		return nil, fmt.Errorf("k projection: %w", err)
	}
	v, err := n.engine.MatMul(ctx, flat, n.attnVProj.Value)
	if err != nil {
		return nil, fmt.Errorf("v projection: %w", err)
	}

	// Reshape to multi-head: [batch, numHeads, seqLen, headDim]
	reshapeToHeads := func(t *tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
		r, err := n.engine.Reshape(ctx, t, []int{batch, seqLen, n.cfg.NumHeads, headDim})
		if err != nil {
			return nil, err
		}
		return n.engine.Transpose(ctx, r, []int{0, 2, 1, 3})
	}

	q, err = reshapeToHeads(q)
	if err != nil {
		return nil, err
	}
	k, err = reshapeToHeads(k)
	if err != nil {
		return nil, err
	}
	v, err = reshapeToHeads(v)
	if err != nil {
		return nil, err
	}

	// Scaled dot-product attention.
	kT, err := n.engine.Transpose(ctx, k, []int{0, 1, 3, 2})
	if err != nil {
		return nil, err
	}
	scores, err := n.engine.MatMul(ctx, q, kT)
	if err != nil {
		return nil, err
	}
	scale := n.ops.FromFloat64(1.0 / math.Sqrt(float64(headDim)))
	scores, err = n.engine.MulScalar(ctx, scores, scale)
	if err != nil {
		return nil, err
	}
	attnWeights, err := n.engine.Softmax(ctx, scores, -1)
	if err != nil {
		return nil, err
	}

	// Attention output: [batch, numHeads, seqLen, headDim]
	attnOut, err := n.engine.MatMul(ctx, attnWeights, v)
	if err != nil {
		return nil, err
	}

	// Reshape back: [batch, seqLen, hiddenDim]
	attnOut, err = n.engine.Transpose(ctx, attnOut, []int{0, 2, 1, 3})
	if err != nil {
		return nil, err
	}
	attnOut, err = n.engine.Reshape(ctx, attnOut, []int{batch * seqLen, dModel})
	if err != nil {
		return nil, err
	}

	// Output projection.
	attnOut, err = n.engine.MatMul(ctx, attnOut, n.attnOProj.Value)
	if err != nil {
		return nil, err
	}

	// Reshape to [batch, seqLen, hiddenDim].
	return n.engine.Reshape(ctx, attnOut, []int{batch, seqLen, dModel})
}

func (n *tftNode[T]) Backward(_ context.Context, _ types.BackwardMode, _ *tensor.TensorNumeric[T], _ ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	return nil, nil
}

// Statically assert that tftNode implements graph.Node.
var _ graph.Node[float32] = (*tftNode[float32])(nil)


// sigmoidFn computes sigmoid(x) = exp(x) / (1 + exp(x)) using composed engine primitives.
func sigmoidFn[T tensor.Numeric](ctx context.Context, engine compute.Engine[T], ops numeric.Arithmetic[T], x *tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	expX, err := engine.Exp(ctx, x)
	if err != nil {
		return nil, err
	}
	onePlusExpX, err := engine.AddScalar(ctx, expX, ops.One())
	if err != nil {
		return nil, err
	}
	return engine.Div(ctx, expX, onePlusExpX)
}
