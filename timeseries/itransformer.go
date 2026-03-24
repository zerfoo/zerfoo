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

// geluGrad computes the derivative of the GELU activation.
func geluGrad(x float64) float64 {
	c := math.Sqrt(2.0 / math.Pi)
	inner := c * (x + 0.044715*x*x*x)
	tanhVal := math.Tanh(inner)
	sech2 := 1.0 - tanhVal*tanhVal
	dinnerDx := c * (1.0 + 3.0*0.044715*x*x)
	return 0.5*(1.0+tanhVal) + 0.5*x*sech2*dinnerDx
}

// iTransformerCache stores intermediate activations needed for backward pass.
type iTransformerCache struct {
	// Input to the model: [channels][inputLen].
	input [][]float64

	// Tokens after embedding, before encoder layers: [channels][dModel].
	embedOut [][]float64

	// Per-layer caches.
	layerCaches []iTransformerLayerCache

	// Tokens before output projection (after all encoder layers): [channels][dModel].
	preProj [][]float64
}

// iTransformerLayerCache stores per-layer intermediate values.
type iTransformerLayerCache struct {
	// Input tokens to this layer: [channels][dModel].
	inputTokens [][]float64

	// Attention sub-layer.
	preAttnTokens [][]float64   // input to attention (same as inputTokens)
	Q, K, V       [][]float64   // [channels][dModel] after linear projections
	attnScores    [][][]float64 // [nHeads][channels][channels] post-softmax
	attnConcat    [][]float64   // [channels][dModel] attention weighted values before O proj
	attnOut       [][]float64   // [channels][dModel] after O projection

	// Post-attention residual + LN1.
	preLN1 [][]float64 // [channels][dModel] residual sum before LN1
	ln1Out [][]float64 // [channels][dModel] after LN1
	ln1Mu  []float64   // [channels] per-channel mean
	ln1Std []float64   // [channels] per-channel std

	// FFN sub-layer.
	fc1Out    [][]float64 // [channels][dFF] before activation
	geluOut   [][]float64 // [channels][dFF] after GELU
	fc2Out    [][]float64 // [channels][dModel] after fc2

	// Post-FFN residual + LN2.
	preLN2 [][]float64 // [channels][dModel] residual sum before LN2
	ln2Out [][]float64 // [channels][dModel] after LN2
	ln2Mu  []float64   // [channels]
	ln2Std []float64   // [channels]
}

// iTransformerGrads accumulates gradients for all parameters.
type iTransformerGrads struct {
	dEmbedW [][]float64
	dEmbedB []float64

	dLayers []iTransformerLayerGrads

	dProjW [][]float64
	dProjB []float64
}

type iTransformerLayerGrads struct {
	dQW, dKW, dVW, dOW [][]float64
	dQB, dKB, dVB, dOB []float64

	dLN1Scale, dLN1Bias []float64

	dFC1W [][]float64
	dFC1B []float64
	dFC2W [][]float64
	dFC2B []float64

	dLN2Scale, dLN2Bias []float64
}

func newITransformerGrads(cfg ITransformerConfig) iTransformerGrads {
	g := iTransformerGrads{
		dEmbedW: zeroMatrix(cfg.InputLen, cfg.DModel),
		dEmbedB: make([]float64, cfg.DModel),
		dProjW:  zeroMatrix(cfg.DModel, cfg.OutputLen),
		dProjB:  make([]float64, cfg.OutputLen),
	}
	g.dLayers = make([]iTransformerLayerGrads, cfg.NLayers)
	for i := range g.dLayers {
		g.dLayers[i] = iTransformerLayerGrads{
			dQW: zeroMatrix(cfg.DModel, cfg.DModel),
			dKW: zeroMatrix(cfg.DModel, cfg.DModel),
			dVW: zeroMatrix(cfg.DModel, cfg.DModel),
			dOW: zeroMatrix(cfg.DModel, cfg.DModel),
			dQB: make([]float64, cfg.DModel),
			dKB: make([]float64, cfg.DModel),
			dVB: make([]float64, cfg.DModel),
			dOB: make([]float64, cfg.DModel),

			dLN1Scale: make([]float64, cfg.DModel),
			dLN1Bias:  make([]float64, cfg.DModel),

			dFC1W: zeroMatrix(cfg.DModel, cfg.DFF),
			dFC1B: make([]float64, cfg.DFF),
			dFC2W: zeroMatrix(cfg.DFF, cfg.DModel),
			dFC2B: make([]float64, cfg.DModel),

			dLN2Scale: make([]float64, cfg.DModel),
			dLN2Bias:  make([]float64, cfg.DModel),
		}
	}
	return g
}

func zeroMatrix(rows, cols int) [][]float64 {
	m := make([][]float64, rows)
	for i := range m {
		m[i] = make([]float64, cols)
	}
	return m
}

// forwardWithCache runs forward pass storing all intermediate activations.
func (m *ITransformer) forwardWithCache(input [][]float64) ([][]float64, iTransformerCache) {
	channels := m.config.Channels
	dModel := m.config.DModel
	cache := iTransformerCache{
		input: input,
	}

	// Step 1: Variate embedding.
	tokens := make([][]float64, channels)
	for c := 0; c < channels; c++ {
		tokens[c] = linearForwardVec(input[c], m.embedW, m.embedB)
	}
	cache.embedOut = deepCopy2D(tokens)

	// Step 2: Encoder layers.
	cache.layerCaches = make([]iTransformerLayerCache, len(m.layers))
	for li, layer := range m.layers {
		var lc iTransformerLayerCache
		tokens, lc = m.encoderLayerForwardCached(tokens, layer)
		cache.layerCaches[li] = lc
	}

	// Store pre-projection tokens.
	cache.preProj = deepCopy2D(tokens)

	// Step 3: Output projection.
	output := make([][]float64, channels)
	for c := 0; c < channels; c++ {
		output[c] = linearForwardVec(tokens[c], m.projW, m.projB)
	}

	_ = dModel
	return output, cache
}

func deepCopy2D(src [][]float64) [][]float64 {
	dst := make([][]float64, len(src))
	for i := range src {
		dst[i] = make([]float64, len(src[i]))
		copy(dst[i], src[i])
	}
	return dst
}

// encoderLayerForwardCached runs one encoder layer, returning cached activations.
func (m *ITransformer) encoderLayerForwardCached(tokens [][]float64, layer iTransformerLayer) ([][]float64, iTransformerLayerCache) {
	channels := len(tokens)
	dModel := m.config.DModel
	nHeads := m.config.NHeads
	headDim := dModel / nHeads

	lc := iTransformerLayerCache{
		inputTokens:   deepCopy2D(tokens),
		preAttnTokens: deepCopy2D(tokens),
	}

	// --- Multi-head self-attention ---
	Q := make([][]float64, channels)
	K := make([][]float64, channels)
	V := make([][]float64, channels)
	for c := 0; c < channels; c++ {
		Q[c] = linearForwardVec(tokens[c], layer.qW, layer.qB)
		K[c] = linearForwardVec(tokens[c], layer.kW, layer.kB)
		V[c] = linearForwardVec(tokens[c], layer.vW, layer.vB)
	}
	lc.Q = deepCopy2D(Q)
	lc.K = deepCopy2D(K)
	lc.V = deepCopy2D(V)

	scale := 1.0 / math.Sqrt(float64(headDim))
	attnConcat := make([][]float64, channels)
	for c := range attnConcat {
		attnConcat[c] = make([]float64, dModel)
	}

	lc.attnScores = make([][][]float64, nHeads)
	for h := 0; h < nHeads; h++ {
		off := h * headDim

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
		for i := 0; i < channels; i++ {
			scores[i] = softmax(scores[i])
		}
		lc.attnScores[h] = scores

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
	lc.attnConcat = deepCopy2D(attnConcat)

	// Output projection.
	attnOut := make([][]float64, channels)
	for c := 0; c < channels; c++ {
		attnOut[c] = linearForwardVec(attnConcat[c], layer.oW, layer.oB)
	}
	lc.attnOut = deepCopy2D(attnOut)

	// Residual + LN1.
	preLN1 := make([][]float64, channels)
	ln1Out := make([][]float64, channels)
	lc.ln1Mu = make([]float64, channels)
	lc.ln1Std = make([]float64, channels)
	for c := 0; c < channels; c++ {
		preLN1[c] = make([]float64, dModel)
		for d := 0; d < dModel; d++ {
			preLN1[c][d] = tokens[c][d] + attnOut[c][d]
		}
		ln1Out[c], lc.ln1Mu[c], lc.ln1Std[c] = layerNormCached(preLN1[c], layer.ln1Scale, layer.ln1Bias)
	}
	lc.preLN1 = deepCopy2D(preLN1)
	lc.ln1Out = deepCopy2D(ln1Out)

	// --- FFN ---
	fc1Out := make([][]float64, channels)
	geluOut := make([][]float64, channels)
	fc2Out := make([][]float64, channels)
	preLN2 := make([][]float64, channels)
	ln2Out := make([][]float64, channels)
	lc.ln2Mu = make([]float64, channels)
	lc.ln2Std = make([]float64, channels)

	for c := 0; c < channels; c++ {
		fc1Out[c] = linearForwardVec(ln1Out[c], layer.fc1W, layer.fc1B)
		geluOut[c] = make([]float64, len(fc1Out[c]))
		for i := range fc1Out[c] {
			geluOut[c][i] = gelu(fc1Out[c][i])
		}
		fc2Out[c] = linearForwardVec(geluOut[c], layer.fc2W, layer.fc2B)

		preLN2[c] = make([]float64, dModel)
		for d := 0; d < dModel; d++ {
			preLN2[c][d] = ln1Out[c][d] + fc2Out[c][d]
		}
		ln2Out[c], lc.ln2Mu[c], lc.ln2Std[c] = layerNormCached(preLN2[c], layer.ln2Scale, layer.ln2Bias)
	}
	lc.fc1Out = deepCopy2D(fc1Out)
	lc.geluOut = deepCopy2D(geluOut)
	lc.fc2Out = deepCopy2D(fc2Out)
	lc.preLN2 = deepCopy2D(preLN2)
	lc.ln2Out = deepCopy2D(ln2Out)

	return ln2Out, lc
}

// layerNormCached computes layer norm and returns output, mean, std.
func layerNormCached(x, scale, bias []float64) ([]float64, float64, float64) {
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
	return out, mean, std
}

// backward computes analytical gradients given dLoss/dOutput.
// dOutput: [channels][outputLen], returns accumulated gradients.
func (m *ITransformer) backward(dOutput [][]float64, cache iTransformerCache, grads *iTransformerGrads) {
	channels := m.config.Channels
	dModel := m.config.DModel

	// Backward through output projection: y = x @ projW + projB
	// dProjW += x^T @ dOutput, dProjB += sum(dOutput), dTokens = dOutput @ projW^T
	dTokens := make([][]float64, channels)
	for c := 0; c < channels; c++ {
		dTokens[c] = make([]float64, dModel)
		// dProjB
		for j := range dOutput[c] {
			grads.dProjB[j] += dOutput[c][j]
		}
		// dProjW and dTokens
		for i := 0; i < dModel; i++ {
			for j := range dOutput[c] {
				grads.dProjW[i][j] += cache.preProj[c][i] * dOutput[c][j]
				dTokens[c][i] += dOutput[c][j] * m.projW[i][j]
			}
		}
	}

	// Backward through encoder layers in reverse order.
	for li := len(m.layers) - 1; li >= 0; li-- {
		dTokens = m.encoderLayerBackward(dTokens, m.layers[li], cache.layerCaches[li], &grads.dLayers[li])
	}

	// Backward through embedding: y = x @ embedW + embedB
	for c := 0; c < channels; c++ {
		for j := 0; j < dModel; j++ {
			grads.dEmbedB[j] += dTokens[c][j]
		}
		for i := range cache.input[c] {
			for j := 0; j < dModel; j++ {
				grads.dEmbedW[i][j] += cache.input[c][i] * dTokens[c][j]
			}
		}
	}
}

// encoderLayerBackward propagates gradients through one encoder layer.
// Returns dTokens (gradient w.r.t. layer input).
func (m *ITransformer) encoderLayerBackward(
	dOut [][]float64,
	layer iTransformerLayer,
	lc iTransformerLayerCache,
	lg *iTransformerLayerGrads,
) [][]float64 {
	channels := len(dOut)
	dModel := m.config.DModel
	dFF := m.config.DFF
	nHeads := m.config.NHeads
	headDim := dModel / nHeads

	// === Backward through LN2 ===
	// Output of layer = LN2(ln1Out + fc2Out)
	// dPreLN2 = layerNormBackward(dOut, ...)
	dPreLN2 := make([][]float64, channels)
	for c := 0; c < channels; c++ {
		dPreLN2[c] = layerNormBackward(dOut[c], lc.preLN2[c], layer.ln2Scale, lc.ln2Mu[c], lc.ln2Std[c], lg.dLN2Scale, lg.dLN2Bias)
	}

	// Residual split: preLN2 = ln1Out + fc2Out
	// dLN1Out = dPreLN2 (residual path)
	// dFC2Out = dPreLN2 (FFN path)
	dLN1Out := make([][]float64, channels)
	for c := 0; c < channels; c++ {
		dLN1Out[c] = make([]float64, dModel)
		copy(dLN1Out[c], dPreLN2[c])
	}

	// === Backward through FFN ===
	for c := 0; c < channels; c++ {
		dFC2Out := dPreLN2[c] // same as dPreLN2[c] for this path

		// fc2: geluOut @ fc2W + fc2B -> fc2Out
		dGeluOut := make([]float64, dFF)
		for j := 0; j < dModel; j++ {
			lg.dFC2B[j] += dFC2Out[j]
		}
		for i := 0; i < dFF; i++ {
			for j := 0; j < dModel; j++ {
				lg.dFC2W[i][j] += lc.geluOut[c][i] * dFC2Out[j]
				dGeluOut[i] += dFC2Out[j] * layer.fc2W[i][j]
			}
		}

		// GELU backward.
		dFC1Out := make([]float64, dFF)
		for i := 0; i < dFF; i++ {
			dFC1Out[i] = dGeluOut[i] * geluGrad(lc.fc1Out[c][i])
		}

		// fc1: ln1Out @ fc1W + fc1B -> fc1Out
		for j := 0; j < dFF; j++ {
			lg.dFC1B[j] += dFC1Out[j]
		}
		for i := 0; i < dModel; i++ {
			for j := 0; j < dFF; j++ {
				lg.dFC1W[i][j] += lc.ln1Out[c][i] * dFC1Out[j]
				dLN1Out[c][i] += dFC1Out[j] * layer.fc1W[i][j]
			}
		}
	}

	// === Backward through LN1 ===
	dPreLN1 := make([][]float64, channels)
	for c := 0; c < channels; c++ {
		dPreLN1[c] = layerNormBackward(dLN1Out[c], lc.preLN1[c], layer.ln1Scale, lc.ln1Mu[c], lc.ln1Std[c], lg.dLN1Scale, lg.dLN1Bias)
	}

	// Residual split: preLN1 = inputTokens + attnOut
	// dInputTokens = dPreLN1 (residual)
	// dAttnOut = dPreLN1 (attention path)
	dInputTokens := make([][]float64, channels)
	for c := 0; c < channels; c++ {
		dInputTokens[c] = make([]float64, dModel)
		copy(dInputTokens[c], dPreLN1[c])
	}

	// === Backward through attention output projection ===
	// attnOut = attnConcat @ oW + oB
	dAttnConcat := make([][]float64, channels)
	for c := 0; c < channels; c++ {
		dAttnConcat[c] = make([]float64, dModel)
		dAttnOut := dPreLN1[c]
		for j := 0; j < dModel; j++ {
			lg.dOB[j] += dAttnOut[j]
		}
		for i := 0; i < dModel; i++ {
			for j := 0; j < dModel; j++ {
				lg.dOW[i][j] += lc.attnConcat[c][i] * dAttnOut[j]
				dAttnConcat[c][i] += dAttnOut[j] * layer.oW[i][j]
			}
		}
	}

	// === Backward through multi-head attention ===
	dQ := make([][]float64, channels)
	dK := make([][]float64, channels)
	dV := make([][]float64, channels)
	for c := 0; c < channels; c++ {
		dQ[c] = make([]float64, dModel)
		dK[c] = make([]float64, dModel)
		dV[c] = make([]float64, dModel)
	}

	scale := 1.0 / math.Sqrt(float64(headDim))
	for h := 0; h < nHeads; h++ {
		off := h * headDim

		// attnConcat[i][off:off+headDim] = sum_j scores[i][j] * V[j][off:off+headDim]
		// dScores[i][j] += dAttnConcat[i][off+d] * V[j][off+d]  (for each d)
		// dV[j][off+d] += scores[i][j] * dAttnConcat[i][off+d]  (for each i)
		dScoresPost := make([][]float64, channels) // post-softmax scores gradient
		for i := 0; i < channels; i++ {
			dScoresPost[i] = make([]float64, channels)
			for j := 0; j < channels; j++ {
				for d := 0; d < headDim; d++ {
					dScoresPost[i][j] += dAttnConcat[i][off+d] * lc.V[j][off+d]
					dV[j][off+d] += lc.attnScores[h][i][j] * dAttnConcat[i][off+d]
				}
			}
		}

		// Softmax backward: dScoresPre[i] = softmaxBackward(dScoresPost[i], scores[h][i])
		for i := 0; i < channels; i++ {
			dPre := softmaxBackward(dScoresPost[i], lc.attnScores[h][i])
			// dPre are gradients w.r.t. pre-softmax scores (already scaled by 'scale')
			// Pre-softmax: scores[i][j] = (Q[i] . K[j]) * scale
			// dQ[i][off+d] += sum_j dPre[j] * scale * K[j][off+d]
			// dK[j][off+d] += dPre[j] * scale * Q[i][off+d]
			for j := 0; j < channels; j++ {
				for d := 0; d < headDim; d++ {
					dQ[i][off+d] += dPre[j] * scale * lc.K[j][off+d]
					dK[j][off+d] += dPre[j] * scale * lc.Q[i][off+d]
				}
			}
		}
	}

	// Backward through Q/K/V projections.
	for c := 0; c < channels; c++ {
		// Q = input @ qW + qB
		for j := 0; j < dModel; j++ {
			lg.dQB[j] += dQ[c][j]
		}
		for i := 0; i < dModel; i++ {
			for j := 0; j < dModel; j++ {
				lg.dQW[i][j] += lc.inputTokens[c][i] * dQ[c][j]
				dInputTokens[c][i] += dQ[c][j] * layer.qW[i][j]
			}
		}

		// K = input @ kW + kB
		for j := 0; j < dModel; j++ {
			lg.dKB[j] += dK[c][j]
		}
		for i := 0; i < dModel; i++ {
			for j := 0; j < dModel; j++ {
				lg.dKW[i][j] += lc.inputTokens[c][i] * dK[c][j]
				dInputTokens[c][i] += dK[c][j] * layer.kW[i][j]
			}
		}

		// V = input @ vW + vB
		for j := 0; j < dModel; j++ {
			lg.dVB[j] += dV[c][j]
		}
		for i := 0; i < dModel; i++ {
			for j := 0; j < dModel; j++ {
				lg.dVW[i][j] += lc.inputTokens[c][i] * dV[c][j]
				dInputTokens[c][i] += dV[c][j] * layer.vW[i][j]
			}
		}
	}

	return dInputTokens
}

// layerNormBackward computes gradients through layer normalization.
// dOut: gradient from upstream, x: pre-norm input, scale: LN scale.
// mu, std: cached mean and standard deviation.
// Accumulates into dScale, dBias. Returns dx.
func layerNormBackward(dOut, x, scale []float64, mu, std float64, dScale, dBias []float64) []float64 {
	n := len(x)
	nf := float64(n)

	// xhat[i] = (x[i] - mu) / std
	// out[i] = scale[i] * xhat[i] + bias[i]
	// dScale[i] += dOut[i] * xhat[i]
	// dBias[i] += dOut[i]
	// dxhat[i] = dOut[i] * scale[i]
	dxhat := make([]float64, n)
	for i := 0; i < n; i++ {
		xhat := (x[i] - mu) / std
		dScale[i] += dOut[i] * xhat
		dBias[i] += dOut[i]
		dxhat[i] = dOut[i] * scale[i]
	}

	// dx[i] = (1/std) * (dxhat[i] - mean(dxhat) - xhat[i] * mean(dxhat * xhat))
	sumDxhat := 0.0
	sumDxhatXhat := 0.0
	for i := 0; i < n; i++ {
		xhat := (x[i] - mu) / std
		sumDxhat += dxhat[i]
		sumDxhatXhat += dxhat[i] * xhat
	}
	meanDxhat := sumDxhat / nf
	meanDxhatXhat := sumDxhatXhat / nf

	dx := make([]float64, n)
	for i := 0; i < n; i++ {
		xhat := (x[i] - mu) / std
		dx[i] = (dxhat[i] - meanDxhat - xhat*meanDxhatXhat) / std
	}
	return dx
}

// softmaxBackward computes gradient through softmax.
// dOut: upstream gradient, s: softmax output.
// Returns gradient w.r.t. pre-softmax logits.
func softmaxBackward(dOut, s []float64) []float64 {
	n := len(s)
	// ds_i/dz_j = s_i * (delta_ij - s_j)
	// dx_i = sum_j dOut_j * s_j * (delta_ij - s_i) = s_i * (dOut_i - sum_j dOut_j * s_j)
	dot := 0.0
	for j := 0; j < n; j++ {
		dot += dOut[j] * s[j]
	}
	dx := make([]float64, n)
	for i := 0; i < n; i++ {
		dx[i] = s[i] * (dOut[i] - dot)
	}
	return dx
}

// collectGrads maps iTransformerGrads into a flat slice matching flatParams order.
func (g *iTransformerGrads) collectGrads(cfg ITransformerConfig) []float64 {
	var grads []float64

	// Embedding.
	for i := range g.dEmbedW {
		grads = append(grads, g.dEmbedW[i]...)
	}
	grads = append(grads, g.dEmbedB...)

	// Layers.
	for li := range g.dLayers {
		lg := &g.dLayers[li]
		for _, w := range [][]float64{} {
			_ = w
		}
		// Q, K, V, O weights.
		for _, dw := range []*[][]float64{&lg.dQW, &lg.dKW, &lg.dVW, &lg.dOW} {
			for i := range *dw {
				grads = append(grads, (*dw)[i]...)
			}
		}
		// Q, K, V, O biases.
		for _, db := range []*[]float64{&lg.dQB, &lg.dKB, &lg.dVB, &lg.dOB} {
			grads = append(grads, (*db)...)
		}

		// LN1.
		grads = append(grads, lg.dLN1Scale...)
		grads = append(grads, lg.dLN1Bias...)

		// FFN.
		for i := range lg.dFC1W {
			grads = append(grads, lg.dFC1W[i]...)
		}
		grads = append(grads, lg.dFC1B...)
		for i := range lg.dFC2W {
			grads = append(grads, lg.dFC2W[i]...)
		}
		grads = append(grads, lg.dFC2B...)

		// LN2.
		grads = append(grads, lg.dLN2Scale...)
		grads = append(grads, lg.dLN2Bias...)
	}

	// Output projection.
	for i := range g.dProjW {
		grads = append(grads, g.dProjW[i]...)
	}
	grads = append(grads, g.dProjB...)

	return grads
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

			// Analytical gradient computation via backpropagation.
			accGrads := newITransformerGrads(m.config)
			batchLoss := 0.0
			scale := 1.0 / float64(bs*m.config.Channels*m.config.OutputLen)

			for s := 0; s < bs; s++ {
				pred, cache := m.forwardWithCache(batchWindows[s])

				// Compute per-sample MSE loss and dLoss/dOutput.
				dOutput := make([][]float64, m.config.Channels)
				for c := 0; c < m.config.Channels; c++ {
					dOutput[c] = make([]float64, m.config.OutputLen)
					for o := 0; o < m.config.OutputLen; o++ {
						labelIdx := s*m.config.Channels*m.config.OutputLen + c*m.config.OutputLen + o
						diff := pred[c][o] - batchLabels[labelIdx]
						batchLoss += diff * diff
						dOutput[c][o] = 2.0 * diff * scale
					}
				}

				m.backward(dOutput, cache, &accGrads)
			}
			batchLoss *= scale

			grads := accGrads.collectGrads(m.config)
			epochLoss += batchLoss
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
