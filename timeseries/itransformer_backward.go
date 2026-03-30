package timeseries

import "math"

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

	// Step 3: Output projection.
	output := make([][]float64, channels)
	for c := 0; c < channels; c++ {
		output[c] = linearForwardVec(tokens[c], m.projW, m.projB)
	}

	_ = dModel
	return output, cache
}

// encoderLayerForwardCached runs one encoder layer, returning cached activations.
func (m *ITransformer) encoderLayerForwardCached(tokens [][]float64, layer iTransformerLayer) ([][]float64, iTransformerLayerCache) {
	channels := len(tokens)
	dModel := m.config.DModel
	nHeads := m.config.NHeads
	headDim := dModel / nHeads

	lc := iTransformerLayerCache{
		inputTokens:   copyMatrix(tokens),
		preAttnTokens: copyMatrix(tokens),
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

	// Output projection.
	attnOut := make([][]float64, channels)
	for c := 0; c < channels; c++ {
		attnOut[c] = linearForwardVec(attnConcat[c], layer.oW, layer.oB)
	}
	lc.attnOut = copyMatrix(attnOut)

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
		ln1Out[c], lc.ln1Mu[c], lc.ln1Std[c] = layerNorm1DCached(preLN1[c], layer.ln1Scale, layer.ln1Bias)
	}
	lc.preLN1 = copyMatrix(preLN1)
	lc.ln1Out = copyMatrix(ln1Out)

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
			geluOut[c][i] = geluScalar[float64](fc1Out[c][i])
		}
		fc2Out[c] = linearForwardVec(geluOut[c], layer.fc2W, layer.fc2B)

		preLN2[c] = make([]float64, dModel)
		for d := 0; d < dModel; d++ {
			preLN2[c][d] = ln1Out[c][d] + fc2Out[c][d]
		}
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
			dFC1Out[i] = dGeluOut[i] * geluDeriv[float64](lc.fc1Out[c][i])
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
