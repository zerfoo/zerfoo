package timeseries

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"math/rand/v2"
	"os"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/zerfoo/layers/functional"
)

// ITransformerConfig holds the configuration for an iTransformer model.
type ITransformerConfig struct {
	Channels  int // number of input channels/variates
	InputLen  int // lookback window length
	OutputLen int // forecast horizon
	DModel    int // model dimension
	DFF       int // feed-forward dimension
	NHeads    int // number of attention heads
	NLayers   int // number of encoder layers
}

// iTransformerLayer holds weights for one encoder layer.
type iTransformerLayer struct {
	// Attention projections: [dModel][dModel]
	qW, kW, vW, oW [][]float64
	qB, kB, vB, oB []float64

	// Layer norm 1 (post-attention): scale and bias [dModel]
	ln1Scale, ln1Bias []float64

	// Feed-forward: fc1 [dModel][dFF], fc2 [dFF][dModel]
	fc1W [][]float64
	fc1B []float64
	fc2W [][]float64
	fc2B []float64

	// Layer norm 2 (post-FFN): scale and bias [dModel]
	ln2Scale, ln2Bias []float64
}

// ITransformer implements the iTransformer model (ICLR 2024).
// It inverts the standard transformer by treating each variate (channel) as a
// token and computing attention across variates rather than across time steps.
type ITransformer struct {
	config ITransformerConfig

	// Variate embedding: linear inputLen -> dModel, per channel is shared.
	embedW [][]float64 // [inputLen][dModel]
	embedB []float64   // [dModel]

	// Encoder layers.
	layers []iTransformerLayer

	// Output projection: dModel -> outputLen.
	projW [][]float64 // [dModel][outputLen]
	projB []float64   // [outputLen]

	// Optional GPU engine for accelerated training. When non-nil,
	// TrainWindowed uses float32 tensor operations instead of the
	// pure-Go float64 CPU path.
	engine compute.Engine[float32]
	ops    numeric.Arithmetic[float32]

	// Normalization stats from training.
	normMeans [][]float64
	normStds  [][]float64

	// Gradient accumulator for TrainableBackend.
	grads []float64
}

// NewITransformer creates a new iTransformer model. The engine and ops
// parameters are optional — pass nil to use the pure-Go CPU training path.
func NewITransformer(config ITransformerConfig, engine compute.Engine[float32], ops numeric.Arithmetic[float32]) (*ITransformer, error) {
	if config.Channels <= 0 {
		return nil, fmt.Errorf("itransformer: Channels must be positive, got %d", config.Channels)
	}
	if config.InputLen <= 0 {
		return nil, fmt.Errorf("itransformer: InputLen must be positive, got %d", config.InputLen)
	}
	if config.OutputLen <= 0 {
		return nil, fmt.Errorf("itransformer: OutputLen must be positive, got %d", config.OutputLen)
	}
	if config.DModel <= 0 {
		return nil, fmt.Errorf("itransformer: DModel must be positive, got %d", config.DModel)
	}
	if config.DFF <= 0 {
		return nil, fmt.Errorf("itransformer: DFF must be positive, got %d", config.DFF)
	}
	if config.NHeads <= 0 {
		return nil, fmt.Errorf("itransformer: NHeads must be positive, got %d", config.NHeads)
	}
	if config.DModel%config.NHeads != 0 {
		return nil, fmt.Errorf("itransformer: DModel (%d) must be divisible by NHeads (%d)", config.DModel, config.NHeads)
	}
	if config.NLayers <= 0 {
		return nil, fmt.Errorf("itransformer: NLayers must be positive, got %d", config.NLayers)
	}

	m := &ITransformer{config: config, engine: engine, ops: ops}

	// Variate embedding: inputLen -> dModel.
	m.embedW = xavierMatrix(config.InputLen, config.DModel)
	m.embedB = make([]float64, config.DModel)

	// Encoder layers.
	m.layers = make([]iTransformerLayer, config.NLayers)
	for i := range m.layers {
		m.layers[i] = newITransformerLayer(config.DModel, config.DFF)
	}

	// Output projection: dModel -> outputLen.
	m.projW = xavierMatrix(config.DModel, config.OutputLen)
	m.projB = make([]float64, config.OutputLen)

	return m, nil
}

// xavierMatrix creates a [rows][cols] matrix with Xavier initialization.
func xavierMatrix(rows, cols int) [][]float64 {
	scale := math.Sqrt(2.0 / float64(rows+cols))
	m := make([][]float64, rows)
	for i := range m {
		m[i] = make([]float64, cols)
		for j := range m[i] {
			m[i][j] = rand.NormFloat64() * scale
		}
	}
	return m
}

func newITransformerLayer(dModel, dFF int) iTransformerLayer {
	l := iTransformerLayer{
		qW: xavierMatrix(dModel, dModel),
		kW: xavierMatrix(dModel, dModel),
		vW: xavierMatrix(dModel, dModel),
		oW: xavierMatrix(dModel, dModel),
		qB: make([]float64, dModel),
		kB: make([]float64, dModel),
		vB: make([]float64, dModel),
		oB: make([]float64, dModel),

		ln1Scale: make([]float64, dModel),
		ln1Bias:  make([]float64, dModel),

		fc1W: xavierMatrix(dModel, dFF),
		fc1B: make([]float64, dFF),
		fc2W: xavierMatrix(dFF, dModel),
		fc2B: make([]float64, dModel),

		ln2Scale: make([]float64, dModel),
		ln2Bias:  make([]float64, dModel),
	}
	// Initialize layer norm scales to 1.
	for i := range l.ln1Scale {
		l.ln1Scale[i] = 1.0
	}
	for i := range l.ln2Scale {
		l.ln2Scale[i] = 1.0
	}
	return l
}

// forward runs the iTransformer forward pass on a single sample.
// Input: [channels][inputLen], returns: [channels][outputLen].
func (m *ITransformer) forward(input [][]float64) [][]float64 {
	ctx := context.Background()

	// Step 1: Variate embedding via engine MatMul.
	// [channels, inputLen] @ [inputLen, dModel] + embedB -> [channels, dModel].
	tokens := linearBatchF64(ctx, input, m.embedW, m.embedB)

	// Step 2: Encoder layers.
	for _, layer := range m.layers {
		tokens = m.encoderLayerForward(tokens, layer)
	}

	// Step 3: Output projection via engine MatMul.
	return linearBatchF64(ctx, tokens, m.projW, m.projB)
}

// linearForwardVec computes y = x @ W + b.
// x: [inDim], W: [inDim][outDim], b: [outDim] -> y: [outDim].
func linearForwardVec(x []float64, w [][]float64, b []float64) []float64 {
	outDim := len(b)
	y := make([]float64, outDim)
	copy(y, b)
	for i, xi := range x {
		for j := 0; j < outDim; j++ {
			y[j] += xi * w[i][j]
		}
	}
	return y
}

// linearBatchF64 computes Y = X @ W + b using cpuEngine64.MatMul and Add.
// X: [rows][inDim], W: [inDim][outDim], b: [outDim] -> Y: [rows][outDim].
func linearBatchF64(ctx context.Context, xRows [][]float64, w [][]float64, b []float64) [][]float64 {
	rows := len(xRows)
	inDim := len(w)
	outDim := len(b)

	xFlat := make([]float64, rows*inDim)
	for r := 0; r < rows; r++ {
		copy(xFlat[r*inDim:], xRows[r])
	}
	wFlat := make([]float64, inDim*outDim)
	for i := 0; i < inDim; i++ {
		copy(wFlat[i*outDim:], w[i])
	}

	xT, _ := tensor.New[float64]([]int{rows, inDim}, xFlat)
	wT, _ := tensor.New[float64]([]int{inDim, outDim}, wFlat)
	bT, _ := tensor.New[float64]([]int{1, outDim}, b)

	yT, err := cpuEngine64.MatMul(ctx, xT, wT)
	if err != nil {
		result := make([][]float64, rows)
		for r := range xRows {
			result[r] = linearForwardVec(xRows[r], w, b)
		}
		return result
	}
	yT, err = cpuEngine64.Add(ctx, yT, bT)
	if err != nil {
		result := make([][]float64, rows)
		for r := range xRows {
			result[r] = linearForwardVec(xRows[r], w, b)
		}
		return result
	}

	yData := yT.Data()
	result := make([][]float64, rows)
	for r := 0; r < rows; r++ {
		result[r] = make([]float64, outDim)
		copy(result[r], yData[r*outDim:(r+1)*outDim])
	}
	return result
}

// addMatricesF64 computes A + B element-wise for [rows][cols] matrices
// using cpuEngine64.Add.
func addMatricesF64(ctx context.Context, a, b [][]float64, rows, cols int) [][]float64 {
	aFlat := make([]float64, rows*cols)
	bFlat := make([]float64, rows*cols)
	for r := 0; r < rows; r++ {
		copy(aFlat[r*cols:], a[r])
		copy(bFlat[r*cols:], b[r])
	}
	aT, _ := tensor.New[float64]([]int{rows, cols}, aFlat)
	bT, _ := tensor.New[float64]([]int{rows, cols}, bFlat)
	cT, err := cpuEngine64.Add(ctx, aT, bT)
	if err != nil {
		for r := 0; r < rows; r++ {
			for c := 0; c < cols; c++ {
				a[r][c] += b[r][c]
			}
		}
		return a
	}
	cData := cT.Data()
	result := make([][]float64, rows)
	for r := 0; r < rows; r++ {
		result[r] = make([]float64, cols)
		copy(result[r], cData[r*cols:(r+1)*cols])
	}
	return result
}

// encoderLayerForward runs one transformer encoder layer.
// tokens: [channels][dModel] -> [channels][dModel].
func (m *ITransformer) encoderLayerForward(tokens [][]float64, layer iTransformerLayer) [][]float64 {
	channels := len(tokens)
	dModel := m.config.DModel
	ctx := context.Background()

	// Multi-head self-attention over variates.
	attnOut := m.multiHeadAttention(tokens, layer)

	// Residual addition via engine.
	tokens = addMatricesF64(ctx, tokens, attnOut, channels, dModel)

	// LayerNorm per channel.
	for c := 0; c < channels; c++ {
		tokens[c] = layerNorm1D(tokens[c], layer.ln1Scale, layer.ln1Bias)
	}

	// Feed-forward: fc1 + GELU + fc2, all via engine MatMul.
	fc1Out := linearBatchF64(ctx, tokens, layer.fc1W, layer.fc1B)
	for c := 0; c < channels; c++ {
		for i, v := range fc1Out[c] {
			inner := math.Sqrt(2/math.Pi) * (v + 0.044715*v*v*v)
			fc1Out[c][i] = 0.5 * v * (1 + math.Tanh(inner))
		}
	}
	fc2Out := linearBatchF64(ctx, fc1Out, layer.fc2W, layer.fc2B)

	// Residual addition via engine.
	tokens = addMatricesF64(ctx, tokens, fc2Out, channels, dModel)

	// LayerNorm per channel.
	for c := 0; c < channels; c++ {
		tokens[c] = layerNorm1D(tokens[c], layer.ln2Scale, layer.ln2Bias)
	}

	return tokens
}

// multiHeadAttention computes multi-head self-attention over variates
// using functional.MultiHeadAttention via the package-level cpuEngine64.
// tokens: [channels][dModel] -> [channels][dModel].
func (m *ITransformer) multiHeadAttention(tokens [][]float64, layer iTransformerLayer) [][]float64 {
	channels := len(tokens)
	dModel := m.config.DModel
	nHeads := m.config.NHeads
	ctx := context.Background()

	// Project Q, K, V for all channels via engine MatMul.
	Q := linearBatchF64(ctx, tokens, layer.qW, layer.qB)
	K := linearBatchF64(ctx, tokens, layer.kW, layer.kB)
	V := linearBatchF64(ctx, tokens, layer.vW, layer.vB)

	// Use functional.MultiHeadAttention for scaled dot-product attention.
	attnConcat := mhaF64(Q, K, V, channels, dModel, nHeads)

	// Output projection via engine MatMul.
	return linearBatchF64(ctx, attnConcat, layer.oW, layer.oB)
}

// mhaF64 wraps functional.MultiHeadAttention for float64 slice data.
// q, k, v: [seq][dModel]. Returns [seq][dModel].
func mhaF64(q, k, v [][]float64, seq, dModel, nHeads int) [][]float64 {
	// Flatten to 1D for tensor creation.
	qFlat := make([]float64, seq*dModel)
	kFlat := make([]float64, seq*dModel)
	vFlat := make([]float64, seq*dModel)
	for s := 0; s < seq; s++ {
		copy(qFlat[s*dModel:], q[s])
		copy(kFlat[s*dModel:], k[s])
		copy(vFlat[s*dModel:], v[s])
	}

	qT, _ := tensor.New[float64]([]int{seq, dModel}, qFlat)
	kT, _ := tensor.New[float64]([]int{seq, dModel}, kFlat)
	vT, _ := tensor.New[float64]([]int{seq, dModel}, vFlat)

	ctx := context.Background()
	out, err := functional.MultiHeadAttention(ctx, cpuEngine64, qT, kT, vT, nHeads)
	if err != nil {
		panic("mhaF64: " + err.Error())
	}

	// Extract back to [][]float64.
	result := make([][]float64, seq)
	data := out.Data()
	for s := 0; s < seq; s++ {
		result[s] = make([]float64, dModel)
		copy(result[s], data[s*dModel:(s+1)*dModel])
	}
	return result
}


// TrainWindowed trains the iTransformer model on windowed data using AdamW.
// windows: [nSamples][channels][inputLen].
// labels: flat slice of length nSamples * channels * outputLen.
func (m *ITransformer) TrainWindowed(windows [][][]float64, labels []float64, config TrainConfig) (*TrainResult, error) {
	nSamples := len(windows)
	if nSamples == 0 {
		return nil, fmt.Errorf("itransformer: empty training set")
	}

	expectedLabels := nSamples * m.config.Channels * m.config.OutputLen
	if len(labels) != expectedLabels {
		return nil, fmt.Errorf("itransformer: expected %d labels, got %d", expectedLabels, len(labels))
	}

	for i, w := range windows {
		if len(w) != m.config.Channels {
			return nil, fmt.Errorf("itransformer: window %d has %d channels, expected %d", i, len(w), m.config.Channels)
		}
		for c, ch := range w {
			if len(ch) != m.config.InputLen {
				return nil, fmt.Errorf("itransformer: window %d channel %d has length %d, expected %d", i, c, len(ch), m.config.InputLen)
			}
		}
	}

	if config.Epochs <= 0 {
		config.Epochs = 100
	}
	if config.LR <= 0 {
		config.LR = 1e-3
	}
	if config.Beta1 <= 0 {
		config.Beta1 = 0.9
	}
	if config.Beta2 <= 0 {
		config.Beta2 = 0.999
	}
	if config.Epsilon <= 0 {
		config.Epsilon = 1e-8
	}

	// Z-score normalize inputs.
	windows, m.normMeans, m.normStds = normalizeWindows(windows)

	if m.engine != nil {
		return m.trainWindowedEngine(windows, labels, config)
	}

	return TrainLoop(m, windows, labels, config)
}

// PredictWindowed runs inference on windowed data.
// windows: [nSamples][channels][inputLen].
// Returns flat predictions of length nSamples * channels * outputLen.
func (m *ITransformer) PredictWindowed(modelPath string, windows [][][]float64) ([]float64, error) {
	if modelPath != "" {
		if err := m.Load(modelPath); err != nil {
			return nil, fmt.Errorf("itransformer: load weights: %w", err)
		}
	}

	nSamples := len(windows)
	if nSamples == 0 {
		return nil, fmt.Errorf("itransformer: empty input")
	}

	if m.normMeans != nil {
		windows = applyNormalization(windows, m.normMeans, m.normStds)
	}

	out := make([]float64, 0, nSamples*m.config.Channels*m.config.OutputLen)
	for _, w := range windows {
		if len(w) != m.config.Channels {
			return nil, fmt.Errorf("itransformer: expected %d channels, got %d", m.config.Channels, len(w))
		}
		pred := m.forward(w)
		for c := 0; c < m.config.Channels; c++ {
			out = append(out, pred[c]...)
		}
	}
	return out, nil
}

// FlatParams returns pointers to all trainable parameters (exported for TrainableBackend).
func (m *ITransformer) FlatParams() []*float64 {
	var params []*float64

	// Embedding weights and bias.
	for i := range m.embedW {
		for j := range m.embedW[i] {
			params = append(params, &m.embedW[i][j])
		}
	}
	for i := range m.embedB {
		params = append(params, &m.embedB[i])
	}

	// Encoder layers.
	for li := range m.layers {
		l := &m.layers[li]
		// Q, K, V, O projections.
		for _, wPtr := range []*[][]float64{&l.qW, &l.kW, &l.vW, &l.oW} {
			for i := range *wPtr {
				for j := range (*wPtr)[i] {
					params = append(params, &(*wPtr)[i][j])
				}
			}
		}
		for _, bPtr := range []*[]float64{&l.qB, &l.kB, &l.vB, &l.oB} {
			for i := range *bPtr {
				params = append(params, &(*bPtr)[i])
			}
		}

		// LN1.
		for i := range l.ln1Scale {
			params = append(params, &l.ln1Scale[i])
		}
		for i := range l.ln1Bias {
			params = append(params, &l.ln1Bias[i])
		}

		// FFN.
		for i := range l.fc1W {
			for j := range l.fc1W[i] {
				params = append(params, &l.fc1W[i][j])
			}
		}
		for i := range l.fc1B {
			params = append(params, &l.fc1B[i])
		}
		for i := range l.fc2W {
			for j := range l.fc2W[i] {
				params = append(params, &l.fc2W[i][j])
			}
		}
		for i := range l.fc2B {
			params = append(params, &l.fc2B[i])
		}

		// LN2.
		for i := range l.ln2Scale {
			params = append(params, &l.ln2Scale[i])
		}
		for i := range l.ln2Bias {
			params = append(params, &l.ln2Bias[i])
		}
	}

	// Output projection.
	for i := range m.projW {
		for j := range m.projW[i] {
			params = append(params, &m.projW[i][j])
		}
	}
	for i := range m.projB {
		params = append(params, &m.projB[i])
	}

	return params
}

// ParamCount returns the total number of trainable parameters (exported for TrainableBackend).
func (m *ITransformer) ParamCount() int {
	return len(m.FlatParams())
}

// ForwardSample runs the iTransformer forward pass on a single sample and returns
// a flat output with cached activations for BackwardSample.
func (m *ITransformer) ForwardSample(input [][]float64) ([]float64, interface{}, error) {
	output, cache := m.forwardWithCache(input)

	flat := make([]float64, 0, m.config.Channels*m.config.OutputLen)
	for c := 0; c < m.config.Channels; c++ {
		flat = append(flat, output[c]...)
	}
	return flat, cache, nil
}

// BackwardSample accumulates parameter gradients for a single sample.
func (m *ITransformer) BackwardSample(dOutput []float64, cacheIface interface{}) error {
	cache, ok := cacheIface.(iTransformerCache)
	if !ok {
		return fmt.Errorf("itransformer: invalid cache type")
	}

	if m.grads == nil {
		m.grads = make([]float64, m.ParamCount())
	}

	// Reshape flat dOutput [channels*outputLen] -> [channels][outputLen].
	dOut2D := make([][]float64, m.config.Channels)
	for c := 0; c < m.config.Channels; c++ {
		dOut2D[c] = dOutput[c*m.config.OutputLen : (c+1)*m.config.OutputLen]
	}

	// Use the existing structured gradient accumulation.
	accGrads := newITransformerGrads(m.config)
	m.backward(dOut2D, cache, &accGrads)

	// Add collected gradients into the flat buffer.
	collected := accGrads.collectGrads(m.config)
	for i, g := range collected {
		m.grads[i] += g
	}
	return nil
}

// FlatGrads returns the internal gradient accumulator.
func (m *ITransformer) FlatGrads() []float64 {
	if m.grads == nil {
		m.grads = make([]float64, m.ParamCount())
	}
	return m.grads
}

// ZeroGrads resets all accumulated gradients to zero.
func (m *ITransformer) ZeroGrads() {
	if m.grads == nil {
		m.grads = make([]float64, m.ParamCount())
		return
	}
	for i := range m.grads {
		m.grads[i] = 0
	}
}

// Parameters returns all trainable parameters as float64 tensors.
// Order: embedding (W, b), per-layer (Q/K/V/O W+b, LN1, FFN, LN2), output projection (W, b).
func (m *ITransformer) Parameters() []*tensor.TensorNumeric[float64] {
	var params []*tensor.TensorNumeric[float64]

	params = append(params, matrixToTensor(m.embedW))
	embedBT, _ := tensor.New[float64]([]int{len(m.embedB)}, m.embedB)
	params = append(params, embedBT)

	for li := range m.layers {
		l := &m.layers[li]
		for _, w := range [][][]float64{l.qW, l.kW, l.vW, l.oW} {
			params = append(params, matrixToTensor(w))
		}
		for _, b := range [][]float64{l.qB, l.kB, l.vB, l.oB} {
			t, _ := tensor.New[float64]([]int{len(b)}, b)
			params = append(params, t)
		}
		t, _ := tensor.New[float64]([]int{len(l.ln1Scale)}, l.ln1Scale)
		params = append(params, t)
		t, _ = tensor.New[float64]([]int{len(l.ln1Bias)}, l.ln1Bias)
		params = append(params, t)

		params = append(params, matrixToTensor(l.fc1W))
		t, _ = tensor.New[float64]([]int{len(l.fc1B)}, l.fc1B)
		params = append(params, t)
		params = append(params, matrixToTensor(l.fc2W))
		t, _ = tensor.New[float64]([]int{len(l.fc2B)}, l.fc2B)
		params = append(params, t)

		t, _ = tensor.New[float64]([]int{len(l.ln2Scale)}, l.ln2Scale)
		params = append(params, t)
		t, _ = tensor.New[float64]([]int{len(l.ln2Bias)}, l.ln2Bias)
		params = append(params, t)
	}

	params = append(params, matrixToTensor(m.projW))
	projBT, _ := tensor.New[float64]([]int{len(m.projB)}, m.projB)
	params = append(params, projBT)

	return params
}

// matrixToTensor converts a [][]float64 matrix to a 2D tensor.
func matrixToTensor(m [][]float64) *tensor.TensorNumeric[float64] {
	rows := len(m)
	if rows == 0 {
		t, _ := tensor.New[float64]([]int{0, 0}, nil)
		return t
	}
	cols := len(m[0])
	flat := make([]float64, rows*cols)
	for i := range m {
		copy(flat[i*cols:], m[i])
	}
	t, _ := tensor.New[float64]([]int{rows, cols}, flat)
	return t
}

// Compile-time check that ITransformer implements TrainableBackend.
var _ TrainableBackend = (*ITransformer)(nil)

// iTransformerWeights is the JSON-serializable form of iTransformer parameters.
type iTransformerWeights struct {
	Config    ITransformerConfig       `json:"config"`
	EmbedW    [][]float64              `json:"embed_w"`
	EmbedB    []float64                `json:"embed_b"`
	Layers    []iTransformerLayerJSON  `json:"layers"`
	ProjW     [][]float64              `json:"proj_w"`
	ProjB     []float64                `json:"proj_b"`
	NormMeans [][]float64              `json:"norm_means,omitempty"`
	NormStds  [][]float64              `json:"norm_stds,omitempty"`
}

type iTransformerLayerJSON struct {
	QW [][]float64 `json:"q_w"`
	KW [][]float64 `json:"k_w"`
	VW [][]float64 `json:"v_w"`
	OW [][]float64 `json:"o_w"`
	QB []float64   `json:"q_b"`
	KB []float64   `json:"k_b"`
	VB []float64   `json:"v_b"`
	OB []float64   `json:"o_b"`

	LN1Scale []float64 `json:"ln1_scale"`
	LN1Bias  []float64 `json:"ln1_bias"`

	FC1W [][]float64 `json:"fc1_w"`
	FC1B []float64   `json:"fc1_b"`
	FC2W [][]float64 `json:"fc2_w"`
	FC2B []float64   `json:"fc2_b"`

	LN2Scale []float64 `json:"ln2_scale"`
	LN2Bias  []float64 `json:"ln2_bias"`
}

// Save writes the model weights to a JSON file.
func (m *ITransformer) Save(path string) error {
	w := iTransformerWeights{
		Config:    m.config,
		EmbedW:    m.embedW,
		EmbedB:    m.embedB,
		ProjW:     m.projW,
		ProjB:     m.projB,
		NormMeans: m.normMeans,
		NormStds:  m.normStds,
	}
	for _, l := range m.layers {
		w.Layers = append(w.Layers, iTransformerLayerJSON{
			QW: l.qW, KW: l.kW, VW: l.vW, OW: l.oW,
			QB: l.qB, KB: l.kB, VB: l.vB, OB: l.oB,
			LN1Scale: l.ln1Scale, LN1Bias: l.ln1Bias,
			FC1W: l.fc1W, FC1B: l.fc1B,
			FC2W: l.fc2W, FC2B: l.fc2B,
			LN2Scale: l.ln2Scale, LN2Bias: l.ln2Bias,
		})
	}
	data, err := json.Marshal(w)
	if err != nil {
		return fmt.Errorf("itransformer: marshal weights: %w", err)
	}
	return os.WriteFile(path, data, 0o644)
}

// Load reads model weights from a JSON file.
func (m *ITransformer) Load(path string) error {
	data, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	var w iTransformerWeights
	if err := json.Unmarshal(data, &w); err != nil {
		return err
	}
	if w.Config != m.config {
		return fmt.Errorf("itransformer: config mismatch: file has %+v, model has %+v", w.Config, m.config)
	}
	if len(w.Layers) != len(m.layers) {
		return fmt.Errorf("itransformer: layer count mismatch: file=%d, model=%d", len(w.Layers), len(m.layers))
	}

	m.embedW = w.EmbedW
	m.embedB = w.EmbedB
	m.projW = w.ProjW
	m.projB = w.ProjB
	m.normMeans = w.NormMeans
	m.normStds = w.NormStds

	for i, lj := range w.Layers {
		m.layers[i] = iTransformerLayer{
			qW: lj.QW, kW: lj.KW, vW: lj.VW, oW: lj.OW,
			qB: lj.QB, kB: lj.KB, vB: lj.VB, oB: lj.OB,
			ln1Scale: lj.LN1Scale, ln1Bias: lj.LN1Bias,
			fc1W: lj.FC1W, fc1B: lj.FC1B,
			fc2W: lj.FC2W, fc2B: lj.FC2B,
			ln2Scale: lj.LN2Scale, ln2Bias: lj.LN2Bias,
		}
	}
	return nil
}
