package timeseries

import (
	"encoding/json"
	"fmt"
	"math"
	"math/rand/v2"
	"os"
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

	// Normalization stats from training.
	normMeans [][]float64
	normStds  [][]float64
}

// NewITransformer creates a new iTransformer model.
func NewITransformer(config ITransformerConfig) (*ITransformer, error) {
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

	m := &ITransformer{config: config}

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
	channels := m.config.Channels

	// Step 1: Variate embedding. Each channel's full time series -> dModel vector.
	// tokens[c] is the embedding for channel c: [dModel].
	tokens := make([][]float64, channels)
	for c := 0; c < channels; c++ {
		tokens[c] = linearForwardVec(input[c], m.embedW, m.embedB)
	}

	// Step 2: Encoder layers.
	for _, layer := range m.layers {
		tokens = m.encoderLayerForward(tokens, layer)
	}

	// Step 3: Output projection per variate.
	output := make([][]float64, channels)
	for c := 0; c < channels; c++ {
		output[c] = linearForwardVec(tokens[c], m.projW, m.projB)
	}
	return output
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

// encoderLayerForward runs one transformer encoder layer.
// tokens: [channels][dModel] -> [channels][dModel].
func (m *ITransformer) encoderLayerForward(tokens [][]float64, layer iTransformerLayer) [][]float64 {
	channels := len(tokens)
	dModel := m.config.DModel

	// Multi-head self-attention over variates.
	attnOut := m.multiHeadAttention(tokens, layer)

	// Residual + LayerNorm.
	for c := 0; c < channels; c++ {
		for d := 0; d < dModel; d++ {
			tokens[c][d] += attnOut[c][d]
		}
		tokens[c] = layerNorm(tokens[c], layer.ln1Scale, layer.ln1Bias)
	}

	// Feed-forward network per variate.
	for c := 0; c < channels; c++ {
		ffnOut := linearForwardVec(tokens[c], layer.fc1W, layer.fc1B)
		// GELU activation.
		for i := range ffnOut {
			ffnOut[i] = gelu(ffnOut[i])
		}
		ffnOut = linearForwardVec(ffnOut, layer.fc2W, layer.fc2B)

		// Residual + LayerNorm.
		for d := 0; d < dModel; d++ {
			tokens[c][d] += ffnOut[d]
		}
		tokens[c] = layerNorm(tokens[c], layer.ln2Scale, layer.ln2Bias)
	}

	return tokens
}

// multiHeadAttention computes multi-head self-attention over variates.
// tokens: [channels][dModel] -> [channels][dModel].
func (m *ITransformer) multiHeadAttention(tokens [][]float64, layer iTransformerLayer) [][]float64 {
	channels := len(tokens)
	dModel := m.config.DModel
	nHeads := m.config.NHeads
	headDim := dModel / nHeads

	// Project Q, K, V for all channels.
	Q := make([][]float64, channels)
	K := make([][]float64, channels)
	V := make([][]float64, channels)
	for c := 0; c < channels; c++ {
		Q[c] = linearForwardVec(tokens[c], layer.qW, layer.qB)
		K[c] = linearForwardVec(tokens[c], layer.kW, layer.kB)
		V[c] = linearForwardVec(tokens[c], layer.vW, layer.vB)
	}

	// Per-head scaled dot-product attention.
	scale := 1.0 / math.Sqrt(float64(headDim))
	attnConcat := make([][]float64, channels)
	for c := range attnConcat {
		attnConcat[c] = make([]float64, dModel)
	}

	for h := 0; h < nHeads; h++ {
		off := h * headDim

		// Compute attention scores: [channels][channels].
		scores := make([][]float64, channels)
		for i := 0; i < channels; i++ {
			scores[i] = make([]float64, channels)
			for j := 0; j < channels; j++ {
				dot := 0.0
				for d := 0; d < headDim; d++ {
					dot += Q[i][off+d] * K[j][off+d]
				}
				scores[i][j] = dot * scale
			}
		}

		// Softmax over variate dimension.
		for i := 0; i < channels; i++ {
			scores[i] = softmax(scores[i])
		}

		// Weighted sum of values.
		for i := 0; i < channels; i++ {
			for d := 0; d < headDim; d++ {
				val := 0.0
				for j := 0; j < channels; j++ {
					val += scores[i][j] * V[j][off+d]
				}
				attnConcat[i][off+d] = val
			}
		}
	}

	// Output projection.
	out := make([][]float64, channels)
	for c := 0; c < channels; c++ {
		out[c] = linearForwardVec(attnConcat[c], layer.oW, layer.oB)
	}
	return out
}

// softmax computes softmax in-place with numerical stability.
func softmax(x []float64) []float64 {
	max := x[0]
	for _, v := range x[1:] {
		if v > max {
			max = v
		}
	}
	sum := 0.0
	out := make([]float64, len(x))
	for i, v := range x {
		out[i] = math.Exp(v - max)
		sum += out[i]
	}
	for i := range out {
		out[i] /= sum
	}
	return out
}

// layerNorm applies layer normalization: y = scale * (x - mean) / (std + eps) + bias.
func layerNorm(x, scale, bias []float64) []float64 {
	n := len(x)
	mean := 0.0
	for _, v := range x {
		mean += v
	}
	mean /= float64(n)

	variance := 0.0
	for _, v := range x {
		d := v - mean
		variance += d * d
	}
	variance /= float64(n)
	std := math.Sqrt(variance + 1e-5)

	out := make([]float64, n)
	for i := range x {
		out[i] = scale[i]*(x[i]-mean)/std + bias[i]
	}
	return out
}

// gelu approximates the GELU activation function.
func gelu(x float64) float64 {
	return 0.5 * x * (1.0 + math.Tanh(math.Sqrt(2.0/math.Pi)*(x+0.044715*x*x*x)))
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

	// Collect all parameters and initialize AdamW state.
	params := m.flatParams()
	nParams := len(params)
	mState := make([]float64, nParams) // first moment
	vState := make([]float64, nParams) // second moment

	result := &TrainResult{
		LossHistory: make([]float64, config.Epochs),
	}

	batchSize := nSamples
	if config.BatchSize > 0 && config.BatchSize < nSamples {
		batchSize = config.BatchSize
	}

	for epoch := 0; epoch < config.Epochs; epoch++ {
		epochLoss := 0.0
		nBatches := 0

		for start := 0; start < nSamples; start += batchSize {
			end := start + batchSize
			if end > nSamples {
				end = nSamples
			}
			bs := end - start
			batchWindows := windows[start:end]
			batchLabels := labels[start*m.config.Channels*m.config.OutputLen : end*m.config.Channels*m.config.OutputLen]

			// Numerical gradient computation.
			grads := make([]float64, nParams)
			eps := 1e-5

			// Compute base loss for this batch.
			baseLoss := m.batchLoss(batchWindows, batchLabels, bs)

			// Compute gradient for each parameter.
			for p := 0; p < nParams; p++ {
				orig := *params[p]
				*params[p] = orig + eps
				lossPlus := m.batchLoss(batchWindows, batchLabels, bs)
				*params[p] = orig
				grads[p] = (lossPlus - baseLoss) / eps
			}

			epochLoss += baseLoss
			nBatches++

			// Gradient clipping.
			if config.GradClip > 0 {
				norm := 0.0
				for _, g := range grads {
					norm += g * g
				}
				norm = math.Sqrt(norm)
				if norm > config.GradClip {
					s := config.GradClip / norm
					for i := range grads {
						grads[i] *= s
					}
				}
			}

			// AdamW update with LR warmup.
			lr := warmupLR(config.LR, epoch, config.WarmupEpochs)
			t := float64(epoch*((nSamples+batchSize-1)/batchSize) + nBatches)
			for i := range params {
				mState[i] = config.Beta1*mState[i] + (1-config.Beta1)*grads[i]
				vState[i] = config.Beta2*vState[i] + (1-config.Beta2)*grads[i]*grads[i]
				mHat := mState[i] / (1 - math.Pow(config.Beta1, t))
				vHat := vState[i] / (1 - math.Pow(config.Beta2, t))
				*params[i] -= lr * (mHat/(math.Sqrt(vHat)+config.Epsilon) + config.WeightDecay*(*params[i]))
			}
		}

		result.LossHistory[epoch] = epochLoss / float64(nBatches)
		result.FinalLoss = result.LossHistory[epoch]

		if !isFinite(result.FinalLoss) {
			return nil, fmt.Errorf("itransformer: training diverged at epoch %d: loss=%v", epoch, result.FinalLoss)
		}
	}

	result.Metrics = map[string]float64{"mse": result.FinalLoss}
	return result, nil
}

// batchLoss computes the MSE loss for a batch.
func (m *ITransformer) batchLoss(windows [][][]float64, labels []float64, bs int) float64 {
	loss := 0.0
	for s := 0; s < bs; s++ {
		pred := m.forward(windows[s])
		for c := 0; c < m.config.Channels; c++ {
			for o := 0; o < m.config.OutputLen; o++ {
				labelIdx := s*m.config.Channels*m.config.OutputLen + c*m.config.OutputLen + o
				diff := pred[c][o] - labels[labelIdx]
				loss += diff * diff
			}
		}
	}
	return loss / float64(bs*m.config.Channels*m.config.OutputLen)
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

// flatParams returns pointers to all trainable parameters.
func (m *ITransformer) flatParams() []*float64 {
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
