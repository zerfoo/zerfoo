package timeseries

import (
	"context"
	"math"

	"github.com/zerfoo/zerfoo/layers/functional"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
)

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
	fc1Out  [][]float64 // [channels][dFF] before activation
	geluOut [][]float64 // [channels][dFF] after GELU
	fc2Out  [][]float64 // [channels][dModel] after fc2

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
	ctx := context.Background()
	cache := iTransformerCache{
		input: input,
	}

	// Step 1: Variate embedding via engine MatMul.
	tokens := linearBatchF64(ctx, input, m.embedW, m.embedB)
	cache.embedOut = copyMatrix(tokens)

	// Step 2: Encoder layers.
	cache.layerCaches = make([]iTransformerLayerCache, len(m.layers))
	for li, layer := range m.layers {
		var lc iTransformerLayerCache
		tokens, lc = m.encoderLayerForwardCached(tokens, layer)
		cache.layerCaches[li] = lc
	}

	// Store pre-projection tokens.
	cache.preProj = copyMatrix(tokens)

	// Step 3: Output projection via engine MatMul.
	output := linearBatchF64(ctx, tokens, m.projW, m.projB)
	return output, cache
}

// encoderLayerForwardCached runs one encoder layer, returning cached activations.
func (m *ITransformer) encoderLayerForwardCached(tokens [][]float64, layer iTransformerLayer) ([][]float64, iTransformerLayerCache) {
	channels := len(tokens)
	dModel := m.config.DModel
	nHeads := m.config.NHeads
	headDim := dModel / nHeads
	ctx := context.Background()

	lc := iTransformerLayerCache{
		inputTokens:   copyMatrix(tokens),
		preAttnTokens: copyMatrix(tokens),
	}

	// --- Multi-head self-attention ---
	// Q, K, V projections via engine MatMul.
	Q := linearBatchF64(ctx, tokens, layer.qW, layer.qB)
	K := linearBatchF64(ctx, tokens, layer.kW, layer.kB)
	V := linearBatchF64(ctx, tokens, layer.vW, layer.vB)
	lc.Q = copyMatrix(Q)
	lc.K = copyMatrix(K)
	lc.V = copyMatrix(V)

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
			scores[i] = softmaxF64(scores[i])
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
	lc.attnConcat = copyMatrix(attnConcat)

	// Output projection via engine MatMul.
	attnOut := linearBatchF64(ctx, attnConcat, layer.oW, layer.oB)
	lc.attnOut = copyMatrix(attnOut)

	// Residual + LN1 via engine Add.
	preLN1 := addMatricesF64(ctx, tokens, attnOut, channels, dModel)
	ln1Out := make([][]float64, channels)
	lc.ln1Mu = make([]float64, channels)
	lc.ln1Std = make([]float64, channels)
	for c := 0; c < channels; c++ {
		ln1Out[c], lc.ln1Mu[c], lc.ln1Std[c] = layerNorm1DCached(preLN1[c], layer.ln1Scale, layer.ln1Bias)
	}
	lc.preLN1 = copyMatrix(preLN1)
	lc.ln1Out = copyMatrix(ln1Out)

	// --- FFN ---
	// fc1 via engine MatMul.
	fc1Out := linearBatchF64(ctx, ln1Out, layer.fc1W, layer.fc1B)
	geluOut := make([][]float64, channels)
	for c := 0; c < channels; c++ {
		geluOut[c] = make([]float64, len(fc1Out[c]))
		for i := range fc1Out[c] {
			v := fc1Out[c][i]
			inner := math.Sqrt(2/math.Pi) * (v + 0.044715*v*v*v)
			geluOut[c][i] = 0.5 * v * (1 + math.Tanh(inner))
		}
	}
	// fc2 via engine MatMul.
	fc2Out := linearBatchF64(ctx, geluOut, layer.fc2W, layer.fc2B)

	// Residual + LN2 via engine Add.
	preLN2 := addMatricesF64(ctx, ln1Out, fc2Out, channels, dModel)
	ln2Out := make([][]float64, channels)
	lc.ln2Mu = make([]float64, channels)
	lc.ln2Std = make([]float64, channels)
	for c := 0; c < channels; c++ {
		ln2Out[c], lc.ln2Mu[c], lc.ln2Std[c] = layerNorm1DCached(preLN2[c], layer.ln2Scale, layer.ln2Bias)
	}
	lc.fc1Out = copyMatrix(fc1Out)
	lc.geluOut = copyMatrix(geluOut)
	lc.fc2Out = copyMatrix(fc2Out)
	lc.preLN2 = copyMatrix(preLN2)
	lc.ln2Out = copyMatrix(ln2Out)

	return ln2Out, lc
}

// backward computes analytical gradients given dLoss/dOutput.
// dOutput: [channels][outputLen], returns accumulated gradients.
func (m *ITransformer) backward(dOutput [][]float64, cache iTransformerCache, grads *iTransformerGrads) {
	channels := m.config.Channels
	dModel := m.config.DModel
	outputLen := m.config.OutputLen
	inputLen := m.config.InputLen

	// Backward through output projection via functional.LinearBackward.
	// iTransformer stores projW as [dModel][outputLen] (y = x @ W + b).
	// functional.LinearBackward expects weight as [out_features, in_features] (y = x @ W^T + b).
	dTokens := linearBackwardF64(dOutput, cache.preProj,
		m.projW, channels, dModel, outputLen,
		grads.dProjW, grads.dProjB)

	// Backward through encoder layers in reverse order.
	for li := len(m.layers) - 1; li >= 0; li-- {
		dTokens = m.encoderLayerBackward(dTokens, m.layers[li], cache.layerCaches[li], &grads.dLayers[li])
	}

	// Backward through embedding via functional.LinearBackward.
	// embedW is [inputLen][dModel] (y = x @ W + b).
	linearBackwardF64(dTokens, cache.input,
		m.embedW, channels, inputLen, dModel,
		grads.dEmbedW, grads.dEmbedB)
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

	// === Backward through LN2 via functional.LayerNormBackward ===
	dPreLN2, dLN2Scale, dLN2Bias := layerNormBackwardFunctional(dOut, lc.preLN2, layer.ln2Scale, channels, dModel)
	accumulateVec(lg.dLN2Scale, dLN2Scale)
	accumulateVec(lg.dLN2Bias, dLN2Bias)

	// Residual split: preLN2 = ln1Out + fc2Out
	// dLN1Out = dPreLN2 (residual path), dFC2Out = dPreLN2 (FFN path)
	dLN1Out := copyMatrix(dPreLN2)

	// === Backward through FFN via functional.MLPBackward ===
	// MLPBackward expects weights in [out, in] layout.
	// iTransformer stores fc1W as [dModel][dFF] and fc2W as [dFF][dModel].
	dMLPInput, dFC1W, dFC1B, dFC2W, dFC2B := mlpBackwardF64(
		dPreLN2, lc.ln1Out, layer.fc1W, layer.fc1B, layer.fc2W, layer.fc2B,
		lc.fc1Out, lc.geluOut,
		channels, dModel, dFF)
	accumulateMatrix(lg.dFC1W, dFC1W)
	accumulateVec(lg.dFC1B, dFC1B)
	accumulateMatrix(lg.dFC2W, dFC2W)
	accumulateVec(lg.dFC2B, dFC2B)

	// Add FFN path gradient to residual path via engine Add.
	ctx := context.Background()
	dLN1Out = addMatricesF64(ctx, dLN1Out, dMLPInput, channels, dModel)

	// === Backward through LN1 via functional.LayerNormBackward ===
	dPreLN1, dLN1Scale, dLN1Bias := layerNormBackwardFunctional(dLN1Out, lc.preLN1, layer.ln1Scale, channels, dModel)
	accumulateVec(lg.dLN1Scale, dLN1Scale)
	accumulateVec(lg.dLN1Bias, dLN1Bias)

	// Residual split: preLN1 = inputTokens + attnOut
	dInputTokens := copyMatrix(dPreLN1)

	// === Backward through attention output projection via functional.LinearBackward ===
	dAttnConcat := linearBackwardF64(dPreLN1, lc.attnConcat,
		layer.oW, channels, dModel, dModel,
		lg.dOW, lg.dOB)

	// === Backward through multi-head attention via functional.MultiHeadAttentionBackward ===
	dQ, dK, dV := multiHeadAttentionBackwardF64(dAttnConcat, lc.Q, lc.K, lc.V, nHeads, channels, dModel)

	// === Backward through Q/K/V projections via functional.LinearBackward ===
	dInputFromQ := linearBackwardF64(dQ, lc.inputTokens,
		layer.qW, channels, dModel, dModel,
		lg.dQW, lg.dQB)
	dInputFromK := linearBackwardF64(dK, lc.inputTokens,
		layer.kW, channels, dModel, dModel,
		lg.dKW, lg.dKB)
	dInputFromV := linearBackwardF64(dV, lc.inputTokens,
		layer.vW, channels, dModel, dModel,
		lg.dVW, lg.dVB)

	// Accumulate Q/K/V input gradients into residual path via engine Add.
	dInputTokens = addMatricesF64(ctx, dInputTokens, dInputFromQ, channels, dModel)
	dInputTokens = addMatricesF64(ctx, dInputTokens, dInputFromK, channels, dModel)
	dInputTokens = addMatricesF64(ctx, dInputTokens, dInputFromV, channels, dModel)

	return dInputTokens
}

// linearBackwardF64 computes backward pass for a linear layer y = x @ W + b
// where W is stored as [inDim][outDim] (iTransformer convention).
// Uses functional.LinearBackward internally (which expects [outDim, inDim] layout).
// Accumulates weight and bias gradients into dW and dB.
// Returns dInput: [batch][inDim].
func linearBackwardF64(dOutput, input [][]float64, w [][]float64,
	batch, inDim, outDim int,
	dW [][]float64, dB []float64) [][]float64 {

	ctx := context.Background()

	// Flatten dOutput [batch][outDim] -> tensor [batch, outDim].
	dOutFlat := make([]float64, batch*outDim)
	for b := 0; b < batch; b++ {
		copy(dOutFlat[b*outDim:], dOutput[b])
	}
	dOutT, _ := tensor.New[float64]([]int{batch, outDim}, dOutFlat)

	// Flatten input [batch][inDim] -> tensor [batch, inDim].
	inFlat := make([]float64, batch*inDim)
	for b := 0; b < batch; b++ {
		copy(inFlat[b*inDim:], input[b])
	}
	inputT, _ := tensor.New[float64]([]int{batch, inDim}, inFlat)

	// Transpose weight from [inDim][outDim] to [outDim, inDim] for functional.
	wFlat := make([]float64, outDim*inDim)
	for i := 0; i < inDim; i++ {
		for j := 0; j < outDim; j++ {
			wFlat[j*inDim+i] = w[i][j]
		}
	}
	weightT, _ := tensor.New[float64]([]int{outDim, inDim}, wFlat)

	dInputT, dWeightT, dBiasT, err := functional.LinearBackward(ctx, cpuEngine64, dOutT, inputT, weightT)
	if err != nil {
		panic("linearBackwardF64: " + err.Error())
	}

	// Extract dInput [batch][inDim].
	dInput := make([][]float64, batch)
	dData := dInputT.Data()
	for b := 0; b < batch; b++ {
		dInput[b] = make([]float64, inDim)
		copy(dInput[b], dData[b*inDim:(b+1)*inDim])
	}

	// Accumulate dWeight: functional returns [outDim, inDim], transpose back to [inDim][outDim].
	dwData := dWeightT.Data()
	for j := 0; j < outDim; j++ {
		for i := 0; i < inDim; i++ {
			dW[i][j] += dwData[j*inDim+i]
		}
	}

	// Accumulate dBias.
	dbData := dBiasT.Data()
	for j := 0; j < outDim; j++ {
		dB[j] += dbData[j]
	}

	return dInput
}

// layerNormBackwardFunctional computes backward pass through layer normalization
// via functional.LayerNormBackward.
// dOut: [batch][d], input: [batch][d], scale: [d].
// Returns dInput [batch][d], dScale [d], dBias [d].
func layerNormBackwardFunctional(dOut, input [][]float64, scale []float64, batch, d int) ([][]float64, []float64, []float64) {
	ctx := context.Background()

	dOutFlat := make([]float64, batch*d)
	inFlat := make([]float64, batch*d)
	for b := 0; b < batch; b++ {
		copy(dOutFlat[b*d:], dOut[b])
		copy(inFlat[b*d:], input[b])
	}
	dOutT, _ := tensor.New[float64]([]int{batch, d}, dOutFlat)
	inputT, _ := tensor.New[float64]([]int{batch, d}, inFlat)
	scaleT, _ := tensor.New[float64]([]int{1, d}, scale)

	dInputT, dScaleT, dBiasT, err := functional.LayerNormBackward(ctx, cpuEngine64, dOutT, inputT, scaleT, 1e-5)
	if err != nil {
		panic("layerNormBackwardFunctional: " + err.Error())
	}

	// Extract dInput.
	dInput := make([][]float64, batch)
	dData := dInputT.Data()
	for b := 0; b < batch; b++ {
		dInput[b] = make([]float64, d)
		copy(dInput[b], dData[b*d:(b+1)*d])
	}

	dScale := make([]float64, d)
	copy(dScale, dScaleT.Data())

	dBias := make([]float64, d)
	copy(dBias, dBiasT.Data())

	return dInput, dScale, dBias
}

// mlpBackwardF64 computes backward pass through a 2-layer MLP with GELU activation
// via functional.MLPBackward.
// dOutput: [batch][outDim], input: [batch][inDim].
// fc1W: [inDim][hiddenDim], fc2W: [hiddenDim][outDim] (iTransformer layout).
// hidden: [batch][hiddenDim] (fc1 output, pre-activation).
// activated: [batch][hiddenDim] (post-GELU).
// Returns dInput [batch][inDim] and weight/bias gradients in iTransformer layout.
func mlpBackwardF64(
	dOutput, input [][]float64,
	fc1W [][]float64, fc1B []float64,
	fc2W [][]float64, fc2B []float64,
	hidden, activated [][]float64,
	batch, inDim, hiddenDim int,
) (dInput [][]float64, dFC1W [][]float64, dFC1B []float64, dFC2W [][]float64, dFC2B []float64) {

	ctx := context.Background()
	ops := numeric.Float64Ops{}
	outDim := inDim // fc2 output dim = dModel = inDim

	// Flatten all inputs to tensors.
	dOutT, _ := tensor.New[float64]([]int{batch, outDim}, flatten2D(dOutput, batch, outDim))
	inputT, _ := tensor.New[float64]([]int{batch, inDim}, flatten2D(input, batch, inDim))
	hiddenT, _ := tensor.New[float64]([]int{batch, hiddenDim}, flatten2D(hidden, batch, hiddenDim))
	activatedT, _ := tensor.New[float64]([]int{batch, hiddenDim}, flatten2D(activated, batch, hiddenDim))

	// Transpose weights: [in, out] -> [out, in] for functional.
	w1T, _ := tensor.New[float64]([]int{hiddenDim, inDim}, transposeFlat(fc1W, inDim, hiddenDim))
	b1T, _ := tensor.New[float64]([]int{hiddenDim}, fc1B)
	w2T, _ := tensor.New[float64]([]int{outDim, hiddenDim}, transposeFlat(fc2W, hiddenDim, outDim))
	b2T, _ := tensor.New[float64]([]int{outDim}, fc2B)

	dInputT, dW1T, dB1T, dW2T, dB2T, err := functional.MLPBackward(ctx, cpuEngine64, ops,
		dOutT, inputT, w1T, b1T, w2T, b2T, hiddenT, activatedT, "gelu")
	if err != nil {
		panic("mlpBackwardF64: " + err.Error())
	}

	// Extract dInput.
	dInput = unflatten2D(dInputT.Data(), batch, inDim)

	// Transpose dW1 from [hiddenDim, inDim] back to [inDim][hiddenDim].
	dFC1W = transposeToMatrix(dW1T.Data(), hiddenDim, inDim)
	dFC1B = make([]float64, hiddenDim)
	copy(dFC1B, dB1T.Data())

	// Transpose dW2 from [outDim, hiddenDim] back to [hiddenDim][outDim].
	dFC2W = transposeToMatrix(dW2T.Data(), outDim, hiddenDim)
	dFC2B = make([]float64, outDim)
	copy(dFC2B, dB2T.Data())

	return
}

// multiHeadAttentionBackwardF64 computes backward pass through multi-head attention
// via functional.MultiHeadAttentionBackward.
// dOutput: [seq][dModel], q/k/v: [seq][dModel].
// Returns dQ, dK, dV: [seq][dModel].
func multiHeadAttentionBackwardF64(dOutput, q, k, v [][]float64, nHeads, seq, dModel int) (dQ, dK, dV [][]float64) {
	ctx := context.Background()
	ops := numeric.Float64Ops{}

	dOutT, _ := tensor.New[float64]([]int{seq, dModel}, flatten2D(dOutput, seq, dModel))
	qT, _ := tensor.New[float64]([]int{seq, dModel}, flatten2D(q, seq, dModel))
	kT, _ := tensor.New[float64]([]int{seq, dModel}, flatten2D(k, seq, dModel))
	vT, _ := tensor.New[float64]([]int{seq, dModel}, flatten2D(v, seq, dModel))

	dQT, dKT, dVT, err := functional.MultiHeadAttentionBackward(ctx, cpuEngine64, ops, dOutT, qT, kT, vT, nHeads)
	if err != nil {
		panic("multiHeadAttentionBackwardF64: " + err.Error())
	}

	dQ = unflatten2D(dQT.Data(), seq, dModel)
	dK = unflatten2D(dKT.Data(), seq, dModel)
	dV = unflatten2D(dVT.Data(), seq, dModel)
	return
}

// flatten2D converts [][]float64 to a flat []float64.
func flatten2D(m [][]float64, rows, cols int) []float64 {
	flat := make([]float64, rows*cols)
	for i := 0; i < rows; i++ {
		copy(flat[i*cols:], m[i])
	}
	return flat
}

// unflatten2D converts flat []float64 to [][]float64.
func unflatten2D(data []float64, rows, cols int) [][]float64 {
	m := make([][]float64, rows)
	for i := 0; i < rows; i++ {
		m[i] = make([]float64, cols)
		copy(m[i], data[i*cols:(i+1)*cols])
	}
	return m
}

// transposeFlat transposes a [rows][cols] matrix stored as [][]float64 to a flat
// [cols*rows] slice in row-major [cols, rows] order.
func transposeFlat(m [][]float64, rows, cols int) []float64 {
	flat := make([]float64, cols*rows)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			flat[j*rows+i] = m[i][j]
		}
	}
	return flat
}

// transposeToMatrix converts flat [rows*cols] data in [rows, cols] layout
// to a [][]float64 in [cols][rows] layout (transpose).
func transposeToMatrix(data []float64, rows, cols int) [][]float64 {
	m := make([][]float64, cols)
	for j := 0; j < cols; j++ {
		m[j] = make([]float64, rows)
		for i := 0; i < rows; i++ {
			m[j][i] = data[i*cols+j]
		}
	}
	return m
}

// accumulateVec adds src element-wise into dst.
func accumulateVec(dst, src []float64) {
	for i := range src {
		dst[i] += src[i]
	}
}

// accumulateMatrix adds src element-wise into dst (2D).
func accumulateMatrix(dst, src [][]float64) {
	for i := range src {
		for j := range src[i] {
			dst[i][j] += src[i][j]
		}
	}
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

// copyMatrix creates a deep copy of a 2D float64 slice.
func copyMatrix(x [][]float64) [][]float64 {
	out := make([][]float64, len(x))
	for i := range x {
		out[i] = make([]float64, len(x[i]))
		copy(out[i], x[i])
	}
	return out
}

// softmaxF64 computes softmax over a 1D float64 slice with numerical stability.
func softmaxF64(x []float64) []float64 {
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
