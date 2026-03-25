package timeseries

import (
	"math"
)

// patchTSTCacheF64 stores activations from the forward pass for backpropagation.
type patchTSTCacheF64 struct {
	// Per-channel caches (one per input channel).
	channels []patchTSTChannelCache
}

// patchTSTChannelCache stores activations for a single channel forward pass.
type patchTSTChannelCache struct {
	patches  [][]float64 // [numPatches][patchLength]
	embedded [][]float64 // [numPatches][dModel] after patch embedding (before posEmb)
	// Per-layer caches.
	layerCaches []encoderLayerCache
	// Input to each encoder layer (x before the layer).
	layerInputs [][][]float64 // [nLayers][numPatches][dModel]
	// Final x after all encoder layers, before flatten.
	finalX [][]float64 // [numPatches][dModel]
	// Flattened input to head.
	flatInput []float64 // [numPatches * dModel]
}

// encoderLayerCache stores activations for one encoder layer.
type encoderLayerCache struct {
	// Pre-norm 1.
	xBeforeNorm1 [][]float64 // input to layer norm 1
	normed1      [][]float64 // output of layer norm 1
	mean1        []float64   // per-position means
	invStd1      []float64   // per-position 1/sqrt(var+eps)
	centered1    [][]float64 // x - mean

	// Attention.
	q, k, v      [][]float64   // [seq][dModel] after projection
	scores       [][][]float64 // [nHeads][seq][seq] attention weights (post-softmax)
	attnOut      [][]float64   // [seq][dModel] after weighted sum (before oProj)
	attnProjOut  [][]float64   // [seq][dModel] after oProj

	// After first residual.
	xAfterAttn [][]float64 // [seq][dModel]

	// Pre-norm 2.
	xBeforeNorm2 [][]float64
	normed2      [][]float64
	mean2        []float64
	invStd2      []float64
	centered2    [][]float64

	// FFN.
	ffn1PreAct [][]float64 // [seq][ffnDim] before GELU
	ffn1Out    [][]float64 // [seq][ffnDim] after GELU
	ffn2Out    [][]float64 // [seq][dModel]
}

// forwardF64WithCache runs the PatchTST forward pass in float64, caching activations.
func (m *PatchTST) forwardF64WithCache(input [][]float64, params *patchTSTParamsF64) ([]float64, *patchTSTCacheF64) {
	numPatches := m.config.NumPatches()
	dModel := m.config.DModel
	nHeads := m.config.NHeads
	headDim := dModel / nHeads
	ffnDim := dModel * 4

	channels := len(input)
	cache := &patchTSTCacheF64{
		channels: make([]patchTSTChannelCache, channels),
	}
	chanOutputs := make([][]float64, channels)

	for ch := 0; ch < channels; ch++ {
		cc := &cache.channels[ch]

		// Extract patches.
		cc.patches = make([][]float64, numPatches)
		for p := 0; p < numPatches; p++ {
			start := p * m.config.Stride
			cc.patches[p] = make([]float64, m.config.PatchLength)
			copy(cc.patches[p], input[ch][start:start+m.config.PatchLength])
		}

		// Patch embedding: each patch through linear [patchLen -> dModel].
		cc.embedded = make([][]float64, numPatches)
		for p := 0; p < numPatches; p++ {
			cc.embedded[p] = make([]float64, dModel)
			for j := 0; j < dModel; j++ {
				val := params.patchEmbB[j]
				for k := 0; k < m.config.PatchLength; k++ {
					val += cc.patches[p][k] * params.patchEmbW[k*dModel+j]
				}
				cc.embedded[p][j] = val
			}
		}

		// Add positional embeddings.
		x := make([][]float64, numPatches)
		for p := 0; p < numPatches; p++ {
			x[p] = make([]float64, dModel)
			for j := 0; j < dModel; j++ {
				x[p][j] = cc.embedded[p][j] + params.posEmb[p*dModel+j]
			}
		}

		// Encoder layers.
		cc.layerCaches = make([]encoderLayerCache, m.config.NLayers)
		cc.layerInputs = make([][][]float64, m.config.NLayers)

		for li := 0; li < m.config.NLayers; li++ {
			layer := &params.layers[li]
			lc := &cc.layerCaches[li]

			// Save input to this layer.
			cc.layerInputs[li] = copyMatrix(x)

			// Pre-norm 1.
			lc.xBeforeNorm1 = copyMatrix(x)
			lc.normed1, lc.mean1, lc.invStd1, lc.centered1 = layerNormF64WithCache(x, layer.norm1, layer.bias1, dModel)

			// Multi-head self-attention with cache.
			lc.q = linearF64(lc.normed1, layer.qW, layer.qB, dModel, dModel)
			lc.k = linearF64(lc.normed1, layer.kW, layer.kB, dModel, dModel)
			lc.v = linearF64(lc.normed1, layer.vW, layer.vB, dModel, dModel)

			// Compute attention per head.
			seq := numPatches
			lc.scores = make([][][]float64, nHeads)
			lc.attnOut = make([][]float64, seq)
			for s := range lc.attnOut {
				lc.attnOut[s] = make([]float64, dModel)
			}

			scale := 1.0 / math.Sqrt(float64(headDim))
			for h := 0; h < nHeads; h++ {
				hOff := h * headDim
				lc.scores[h] = make([][]float64, seq)
				for i := 0; i < seq; i++ {
					lc.scores[h][i] = make([]float64, seq)
					for j := 0; j < seq; j++ {
						dot := 0.0
						for d := 0; d < headDim; d++ {
							dot += lc.q[i][hOff+d] * lc.k[j][hOff+d]
						}
						lc.scores[h][i][j] = dot * scale
					}
				}
				// Softmax.
				for i := 0; i < seq; i++ {
					maxS := lc.scores[h][i][0]
					for j := 1; j < seq; j++ {
						if lc.scores[h][i][j] > maxS {
							maxS = lc.scores[h][i][j]
						}
					}
					sumExp := 0.0
					for j := 0; j < seq; j++ {
						lc.scores[h][i][j] = math.Exp(lc.scores[h][i][j] - maxS)
						sumExp += lc.scores[h][i][j]
					}
					for j := 0; j < seq; j++ {
						lc.scores[h][i][j] /= sumExp
					}
				}
				// Weighted sum.
				for i := 0; i < seq; i++ {
					for d := 0; d < headDim; d++ {
						val := 0.0
						for j := 0; j < seq; j++ {
							val += lc.scores[h][i][j] * lc.v[j][hOff+d]
						}
						lc.attnOut[i][hOff+d] = val
					}
				}
			}

			// Output projection.
			lc.attnProjOut = linearF64(lc.attnOut, layer.oW, layer.oB, dModel, dModel)

			// Residual 1.
			for p := 0; p < numPatches; p++ {
				for j := 0; j < dModel; j++ {
					x[p][j] += lc.attnProjOut[p][j]
				}
			}
			lc.xAfterAttn = copyMatrix(x)

			// Pre-norm 2.
			lc.xBeforeNorm2 = copyMatrix(x)
			lc.normed2, lc.mean2, lc.invStd2, lc.centered2 = layerNormF64WithCache(x, layer.norm2, layer.bias2, dModel)

			// FFN.
			lc.ffn1PreAct = make([][]float64, numPatches)
			lc.ffn1Out = make([][]float64, numPatches)
			lc.ffn2Out = make([][]float64, numPatches)
			for p := 0; p < numPatches; p++ {
				lc.ffn1PreAct[p] = make([]float64, ffnDim)
				lc.ffn1Out[p] = make([]float64, ffnDim)
				for j := 0; j < ffnDim; j++ {
					val := layer.ffn1B[j]
					for k := 0; k < dModel; k++ {
						val += lc.normed2[p][k] * layer.ffn1W[k*ffnDim+j]
					}
					lc.ffn1PreAct[p][j] = val
					lc.ffn1Out[p][j] = geluF64(val)
				}
				lc.ffn2Out[p] = make([]float64, dModel)
				for j := 0; j < dModel; j++ {
					val := layer.ffn2B[j]
					for k := 0; k < ffnDim; k++ {
						val += lc.ffn1Out[p][k] * layer.ffn2W[k*dModel+j]
					}
					lc.ffn2Out[p][j] = val
				}
			}

			// Residual 2.
			for p := 0; p < numPatches; p++ {
				for j := 0; j < dModel; j++ {
					x[p][j] += lc.ffn2Out[p][j]
				}
			}
		}

		cc.finalX = copyMatrix(x)

		// Flatten.
		cc.flatInput = make([]float64, numPatches*dModel)
		for p := 0; p < numPatches; p++ {
			copy(cc.flatInput[p*dModel:(p+1)*dModel], x[p])
		}

		// Output head.
		headIn := numPatches * dModel
		out := make([]float64, m.config.OutputDim)
		for j := 0; j < m.config.OutputDim; j++ {
			val := params.headB[j]
			for k := 0; k < headIn; k++ {
				val += cc.flatInput[k] * params.headW[k*m.config.OutputDim+j]
			}
			out[j] = val
		}
		chanOutputs[ch] = out
	}

	// Average across channels.
	result := make([]float64, m.config.OutputDim)
	for ch := 0; ch < channels; ch++ {
		for j := 0; j < m.config.OutputDim; j++ {
			result[j] += chanOutputs[ch][j]
		}
	}
	for j := range result {
		result[j] /= float64(channels)
	}
	return result, cache
}

// backwardF64 computes analytical gradients for the PatchTST model.
// dOutput: gradient of loss w.r.t. model output [outputDim].
// Returns gradient vector in same order as flatParams().
func (m *PatchTST) backwardF64(dOutput []float64, params *patchTSTParamsF64, cache *patchTSTCacheF64) []float64 {
	numPatches := m.config.NumPatches()
	dModel := m.config.DModel
	nHeads := m.config.NHeads
	headDim := dModel / nHeads
	ffnDim := dModel * 4
	outDim := m.config.OutputDim
	channels := len(cache.channels)
	headIn := numPatches * dModel

	// Initialize gradient accumulators for all parameters.
	dPatchEmbW := make([]float64, len(params.patchEmbW))
	dPatchEmbB := make([]float64, len(params.patchEmbB))
	dPosEmb := make([]float64, len(params.posEmb))
	dLayers := make([]encoderLayerF64Grad, m.config.NLayers)
	for li := range dLayers {
		dLayers[li] = newEncoderLayerF64Grad(dModel, ffnDim)
	}
	dHeadW := make([]float64, len(params.headW))
	dHeadB := make([]float64, len(params.headB))

	// dOutput is dL/d(averaged output). Each channel contributes 1/channels.
	chanScale := 1.0 / float64(channels)

	for ch := 0; ch < channels; ch++ {
		cc := &cache.channels[ch]

		// dL/d(channel output) = dOutput * chanScale.
		dChanOut := make([]float64, outDim)
		for j := range dChanOut {
			dChanOut[j] = dOutput[j] * chanScale
		}

		// Head backward: out = flatInput @ headW + headB
		// dHeadW += flatInput^T @ dChanOut (outer product).
		// dHeadB += dChanOut.
		// dFlat = dChanOut @ headW^T.
		for k := 0; k < headIn; k++ {
			for j := 0; j < outDim; j++ {
				dHeadW[k*outDim+j] += cc.flatInput[k] * dChanOut[j]
			}
		}
		for j := 0; j < outDim; j++ {
			dHeadB[j] += dChanOut[j]
		}
		dFlat := make([]float64, headIn)
		for k := 0; k < headIn; k++ {
			for j := 0; j < outDim; j++ {
				dFlat[k] += dChanOut[j] * params.headW[k*outDim+j]
			}
		}

		// Unflatten: dX [numPatches][dModel].
		dX := make([][]float64, numPatches)
		for p := 0; p < numPatches; p++ {
			dX[p] = make([]float64, dModel)
			copy(dX[p], dFlat[p*dModel:(p+1)*dModel])
		}

		// Backward through encoder layers (reverse order).
		for li := m.config.NLayers - 1; li >= 0; li-- {
			layer := &params.layers[li]
			lc := &cc.layerCaches[li]
			dg := &dLayers[li]

			// Residual 2 backward: dX flows to both ffn output and the input after attn.
			dFFN2Out := copyMatrix(dX)

			// FFN backward.
			// ffn2Out[p][j] = sum_k(ffn1Out[p][k] * ffn2W[k*dModel+j]) + ffn2B[j]
			dFFN1Out := make([][]float64, numPatches)
			for p := 0; p < numPatches; p++ {
				dFFN1Out[p] = make([]float64, ffnDim)
				for k := 0; k < ffnDim; k++ {
					for j := 0; j < dModel; j++ {
						dg.ffn2W[k*dModel+j] += lc.ffn1Out[p][k] * dFFN2Out[p][j]
						dFFN1Out[p][k] += dFFN2Out[p][j] * layer.ffn2W[k*dModel+j]
					}
				}
				for j := 0; j < dModel; j++ {
					dg.ffn2B[j] += dFFN2Out[p][j]
				}
			}

			// GELU backward.
			dFFN1PreAct := make([][]float64, numPatches)
			for p := 0; p < numPatches; p++ {
				dFFN1PreAct[p] = make([]float64, ffnDim)
				for j := 0; j < ffnDim; j++ {
					dFFN1PreAct[p][j] = dFFN1Out[p][j] * geluDerivF64(lc.ffn1PreAct[p][j])
				}
			}

			// FFN1 backward: ffn1PreAct[p][j] = sum_k(normed2[p][k] * ffn1W[k*ffnDim+j]) + ffn1B[j]
			dNormed2 := make([][]float64, numPatches)
			for p := 0; p < numPatches; p++ {
				dNormed2[p] = make([]float64, dModel)
				for k := 0; k < dModel; k++ {
					for j := 0; j < ffnDim; j++ {
						dg.ffn1W[k*ffnDim+j] += lc.normed2[p][k] * dFFN1PreAct[p][j]
						dNormed2[p][k] += dFFN1PreAct[p][j] * layer.ffn1W[k*ffnDim+j]
					}
				}
				for j := 0; j < ffnDim; j++ {
					dg.ffn1B[j] += dFFN1PreAct[p][j]
				}
			}

			// LayerNorm2 backward.
			dXAfterAttn := layerNormBackwardF64(dNormed2, lc.centered2, lc.invStd2, layer.norm2, dg.norm2, dg.bias2, dModel)

			// Add residual gradient from FFN path.
			for p := 0; p < numPatches; p++ {
				for j := 0; j < dModel; j++ {
					dXAfterAttn[p][j] += dX[p][j]
				}
			}

			// Residual 1 backward: dXAfterAttn flows to both attention output and layer input.
			dAttnProjOut := copyMatrix(dXAfterAttn)

			// oProj backward: attnProjOut = attnOut @ oW + oB.
			dAttnOut := make([][]float64, numPatches)
			for p := 0; p < numPatches; p++ {
				dAttnOut[p] = make([]float64, dModel)
				for k := 0; k < dModel; k++ {
					for j := 0; j < dModel; j++ {
						dg.oW[k*dModel+j] += lc.attnOut[p][k] * dAttnProjOut[p][j]
						dAttnOut[p][k] += dAttnProjOut[p][j] * layer.oW[k*dModel+j]
					}
				}
				for j := 0; j < dModel; j++ {
					dg.oB[j] += dAttnProjOut[p][j]
				}
			}

			// Attention backward.
			seq := numPatches
			dQ := make([][]float64, seq)
			dK := make([][]float64, seq)
			dV := make([][]float64, seq)
			for s := 0; s < seq; s++ {
				dQ[s] = make([]float64, dModel)
				dK[s] = make([]float64, dModel)
				dV[s] = make([]float64, dModel)
			}

			attnScale := 1.0 / math.Sqrt(float64(headDim))
			for h := 0; h < nHeads; h++ {
				hOff := h * headDim

				// dAttnOut -> dScores, dV.
				// attnOut[i][hOff+d] = sum_j scores[h][i][j] * v[j][hOff+d]
				dScores := make([][]float64, seq)
				for i := 0; i < seq; i++ {
					dScores[i] = make([]float64, seq)
					for j := 0; j < seq; j++ {
						for d := 0; d < headDim; d++ {
							dScores[i][j] += dAttnOut[i][hOff+d] * lc.v[j][hOff+d]
							dV[j][hOff+d] += lc.scores[h][i][j] * dAttnOut[i][hOff+d]
						}
					}
				}

				// Softmax backward: dLogits = scores * (dScores - sum(scores * dScores)).
				dLogits := make([][]float64, seq)
				for i := 0; i < seq; i++ {
					dLogits[i] = make([]float64, seq)
					dot := 0.0
					for j := 0; j < seq; j++ {
						dot += lc.scores[h][i][j] * dScores[i][j]
					}
					for j := 0; j < seq; j++ {
						dLogits[i][j] = lc.scores[h][i][j] * (dScores[i][j] - dot)
					}
				}

				// Scale backward.
				for i := 0; i < seq; i++ {
					for j := 0; j < seq; j++ {
						dLogits[i][j] *= attnScale
					}
				}

				// QK^T backward: logits[i][j] = sum_d q[i][hOff+d] * k[j][hOff+d]
				for i := 0; i < seq; i++ {
					for j := 0; j < seq; j++ {
						for d := 0; d < headDim; d++ {
							dQ[i][hOff+d] += dLogits[i][j] * lc.k[j][hOff+d]
							dK[j][hOff+d] += dLogits[i][j] * lc.q[i][hOff+d]
						}
					}
				}
			}

			// Q/K/V projection backward.
			dNormed1 := make([][]float64, numPatches)
			for p := 0; p < numPatches; p++ {
				dNormed1[p] = make([]float64, dModel)
			}

			linearBackwardF64Accum(dQ, lc.normed1, layer.qW, dNormed1, dg.qW, dg.qB, dModel, dModel)
			linearBackwardF64Accum(dK, lc.normed1, layer.kW, dNormed1, dg.kW, dg.kB, dModel, dModel)
			linearBackwardF64Accum(dV, lc.normed1, layer.vW, dNormed1, dg.vW, dg.vB, dModel, dModel)

			// LayerNorm1 backward.
			dLayerInput := layerNormBackwardF64(dNormed1, lc.centered1, lc.invStd1, layer.norm1, dg.norm1, dg.bias1, dModel)

			// Add residual gradient from attention path.
			for p := 0; p < numPatches; p++ {
				for j := 0; j < dModel; j++ {
					dLayerInput[p][j] += dXAfterAttn[p][j]
				}
			}

			dX = dLayerInput
		}

		// Positional embedding gradient.
		for p := 0; p < numPatches; p++ {
			for j := 0; j < dModel; j++ {
				dPosEmb[p*dModel+j] += dX[p][j]
			}
		}

		// Patch embedding backward.
		// embedded[p][j] = sum_k(patches[p][k] * patchEmbW[k*dModel+j]) + patchEmbB[j]
		for p := 0; p < numPatches; p++ {
			for k := 0; k < m.config.PatchLength; k++ {
				for j := 0; j < dModel; j++ {
					dPatchEmbW[k*dModel+j] += cc.patches[p][k] * dX[p][j]
				}
			}
			for j := 0; j < dModel; j++ {
				dPatchEmbB[j] += dX[p][j]
			}
		}
	}

	// Assemble flat gradient vector in same order as flatParams().
	grads := make([]float64, params.paramCount())
	gi := 0
	gi += copy(grads[gi:], dPatchEmbW)
	gi += copy(grads[gi:], dPatchEmbB)
	gi += copy(grads[gi:], dPosEmb)
	for li := range dLayers {
		dg := &dLayers[li]
		gi += copy(grads[gi:], dg.qW)
		gi += copy(grads[gi:], dg.qB)
		gi += copy(grads[gi:], dg.kW)
		gi += copy(grads[gi:], dg.kB)
		gi += copy(grads[gi:], dg.vW)
		gi += copy(grads[gi:], dg.vB)
		gi += copy(grads[gi:], dg.oW)
		gi += copy(grads[gi:], dg.oB)
		gi += copy(grads[gi:], dg.ffn1W)
		gi += copy(grads[gi:], dg.ffn1B)
		gi += copy(grads[gi:], dg.ffn2W)
		gi += copy(grads[gi:], dg.ffn2B)
		gi += copy(grads[gi:], dg.norm1)
		gi += copy(grads[gi:], dg.bias1)
		gi += copy(grads[gi:], dg.norm2)
		gi += copy(grads[gi:], dg.bias2)
	}
	gi += copy(grads[gi:], dHeadW)
	copy(grads[gi:], dHeadB)
	return grads
}

// encoderLayerF64Grad holds gradient accumulators for one encoder layer.
type encoderLayerF64Grad struct {
	qW, qB       []float64
	kW, kB       []float64
	vW, vB       []float64
	oW, oB       []float64
	ffn1W, ffn1B []float64
	ffn2W, ffn2B []float64
	norm1, bias1 []float64
	norm2, bias2 []float64
}

func newEncoderLayerF64Grad(dModel, ffnDim int) encoderLayerF64Grad {
	return encoderLayerF64Grad{
		qW: make([]float64, dModel*dModel), qB: make([]float64, dModel),
		kW: make([]float64, dModel*dModel), kB: make([]float64, dModel),
		vW: make([]float64, dModel*dModel), vB: make([]float64, dModel),
		oW: make([]float64, dModel*dModel), oB: make([]float64, dModel),
		ffn1W: make([]float64, dModel*ffnDim), ffn1B: make([]float64, ffnDim),
		ffn2W: make([]float64, ffnDim*dModel), ffn2B: make([]float64, dModel),
		norm1: make([]float64, dModel), bias1: make([]float64, dModel),
		norm2: make([]float64, dModel), bias2: make([]float64, dModel),
	}
}

// layerNormF64WithCache applies layer normalization and returns cached intermediates.
func layerNormF64WithCache(x [][]float64, scale, bias []float64, dModel int) (normed [][]float64, means []float64, invStds []float64, centered [][]float64) {
	seq := len(x)
	normed = make([][]float64, seq)
	means = make([]float64, seq)
	invStds = make([]float64, seq)
	centered = make([][]float64, seq)

	for s := 0; s < seq; s++ {
		mean := 0.0
		for j := 0; j < dModel; j++ {
			mean += x[s][j]
		}
		mean /= float64(dModel)
		means[s] = mean

		variance := 0.0
		centered[s] = make([]float64, dModel)
		for j := 0; j < dModel; j++ {
			centered[s][j] = x[s][j] - mean
			variance += centered[s][j] * centered[s][j]
		}
		variance /= float64(dModel)

		invStd := 1.0 / math.Sqrt(variance+1e-5)
		invStds[s] = invStd

		normed[s] = make([]float64, dModel)
		for j := 0; j < dModel; j++ {
			normed[s][j] = centered[s][j]*invStd*scale[j] + bias[j]
		}
	}
	return
}

// layerNormBackwardF64 computes the backward pass through layer normalization.
// dOut: [seq][dModel], centered: [seq][dModel], invStd: [seq], scale: [dModel].
// Accumulates into dScale and dBias. Returns dInput: [seq][dModel].
func layerNormBackwardF64(dOut [][]float64, centered [][]float64, invStd []float64, scale []float64, dScale, dBias []float64, dModel int) [][]float64 {
	seq := len(dOut)
	dInput := make([][]float64, seq)
	d := float64(dModel)

	for s := 0; s < seq; s++ {
		// dNormed = dOut * scale (element-wise).
		dNormed := make([]float64, dModel)
		for j := 0; j < dModel; j++ {
			dNormed[j] = dOut[s][j] * scale[j]
			// Accumulate scale/bias gradients.
			dScale[j] += dOut[s][j] * centered[s][j] * invStd[s]
			dBias[j] += dOut[s][j]
		}

		// LayerNorm backward:
		// xhat = centered * invStd
		// dVar = sum(dNormed * centered) * (-0.5) * invStd^3
		// dMean = sum(dNormed) * (-invStd) + dVar * (-2/d) * sum(centered)
		//       = sum(dNormed) * (-invStd)  [since sum(centered)=0]
		// dInput = dNormed * invStd + dVar * (2*centered/d) + dMean/d
		//
		// Simplified:
		// dInput[j] = invStd * (dNormed[j] - mean(dNormed) - xhat[j]*mean(dNormed*xhat))

		// Compute mean(dNormed) and mean(dNormed * xhat).
		meanDN := 0.0
		meanDNxhat := 0.0
		for j := 0; j < dModel; j++ {
			xhat := centered[s][j] * invStd[s]
			meanDN += dNormed[j]
			meanDNxhat += dNormed[j] * xhat
		}
		meanDN /= d
		meanDNxhat /= d

		dInput[s] = make([]float64, dModel)
		for j := 0; j < dModel; j++ {
			xhat := centered[s][j] * invStd[s]
			dInput[s][j] = invStd[s] * (dNormed[j] - meanDN - xhat*meanDNxhat)
		}
	}
	return dInput
}

// linearBackwardF64Accum computes backward for y = x @ W + b and accumulates gradients.
// dY: [n][outDim], x: [n][inDim], W: [inDim*outDim] row-major.
// Accumulates into dX, dW, dB.
func linearBackwardF64Accum(dY, x [][]float64, w []float64, dX [][]float64, dW, dB []float64, inDim, outDim int) {
	n := len(dY)
	for i := 0; i < n; i++ {
		for j := 0; j < outDim; j++ {
			dB[j] += dY[i][j]
			for k := 0; k < inDim; k++ {
				dW[k*outDim+j] += x[i][k] * dY[i][j]
				dX[i][k] += dY[i][j] * w[k*outDim+j]
			}
		}
	}
}

// geluDerivF64 computes the derivative of the GELU approximation.
func geluDerivF64(x float64) float64 {
	c := math.Sqrt(2.0 / math.Pi)
	inner := c * (x + 0.044715*x*x*x)
	tanh := math.Tanh(inner)
	// d/dx tanh(inner) = (1 - tanh^2) * d(inner)/dx
	// d(inner)/dx = c * (1 + 3*0.044715*x^2)
	dInner := c * (1 + 3*0.044715*x*x)
	// GELU(x) = 0.5 * x * (1 + tanh(inner))
	// GELU'(x) = 0.5 * (1 + tanh) + 0.5 * x * (1 - tanh^2) * dInner
	return 0.5*(1+tanh) + 0.5*x*(1-tanh*tanh)*dInner
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

// geluF64 computes the GELU approximation for a float64 value.
func geluF64(x float64) float64 {
	inner := math.Sqrt(2/math.Pi) * (x + 0.044715*x*x*x)
	return 0.5 * x * (1 + math.Tanh(inner))
}

// patchTSTParamsF64 holds a float64 copy of all PatchTST parameters for training.
type patchTSTParamsF64 struct {
	patchEmbW []float64 // [patchLen * dModel]
	patchEmbB []float64 // [dModel]
	posEmb    []float64 // [numPatches * dModel]
	layers    []encoderLayerF64
	headW     []float64 // [numPatches*dModel * outputDim]
	headB     []float64 // [outputDim]
}

type encoderLayerF64 struct {
	qW, qB []float64 // [dModel * dModel], [dModel]
	kW, kB []float64
	vW, vB []float64
	oW, oB []float64
	ffn1W  []float64 // [dModel * 4*dModel]
	ffn1B  []float64 // [4*dModel]
	ffn2W  []float64 // [4*dModel * dModel]
	ffn2B  []float64 // [dModel]
	norm1  []float64 // [dModel]
	bias1  []float64 // [dModel]
	norm2  []float64 // [dModel]
	bias2  []float64 // [dModel]
}

// extractParamsF64 copies float32 tensor parameters to float64.
func (m *PatchTST) extractParamsF64() *patchTSTParamsF64 {
	p := &patchTSTParamsF64{}
	p.patchEmbW = float32ToFloat64(m.patchEmb.weights.Data())
	p.patchEmbB = float32ToFloat64(m.patchEmb.biases.Data())
	p.posEmb = float32ToFloat64(m.posEmb.Data())

	p.layers = make([]encoderLayerF64, len(m.layers))
	for i, l := range m.layers {
		p.layers[i] = encoderLayerF64{
			qW: float32ToFloat64(l.qProj.weights.Data()),
			qB: float32ToFloat64(l.qProj.biases.Data()),
			kW: float32ToFloat64(l.kProj.weights.Data()),
			kB: float32ToFloat64(l.kProj.biases.Data()),
			vW: float32ToFloat64(l.vProj.weights.Data()),
			vB: float32ToFloat64(l.vProj.biases.Data()),
			oW: float32ToFloat64(l.oProj.weights.Data()),
			oB: float32ToFloat64(l.oProj.biases.Data()),
			ffn1W: float32ToFloat64(l.ffn1.weights.Data()),
			ffn1B: float32ToFloat64(l.ffn1.biases.Data()),
			ffn2W: float32ToFloat64(l.ffn2.weights.Data()),
			ffn2B: float32ToFloat64(l.ffn2.biases.Data()),
			norm1: float32ToFloat64(l.norm1.Data()),
			bias1: float32ToFloat64(l.bias1.Data()),
			norm2: float32ToFloat64(l.norm2.Data()),
			bias2: float32ToFloat64(l.bias2.Data()),
		}
	}

	p.headW = float32ToFloat64(m.head.weights.Data())
	p.headB = float32ToFloat64(m.head.biases.Data())
	return p
}

// writeBackF32 copies float64 parameters back to the float32 tensors.
func (m *PatchTST) writeBackF32(p *patchTSTParamsF64) {
	copy(m.patchEmb.weights.Data(), float64ToFloat32(p.patchEmbW))
	copy(m.patchEmb.biases.Data(), float64ToFloat32(p.patchEmbB))
	copy(m.posEmb.Data(), float64ToFloat32(p.posEmb))

	for i := range m.layers {
		copy(m.layers[i].qProj.weights.Data(), float64ToFloat32(p.layers[i].qW))
		copy(m.layers[i].qProj.biases.Data(), float64ToFloat32(p.layers[i].qB))
		copy(m.layers[i].kProj.weights.Data(), float64ToFloat32(p.layers[i].kW))
		copy(m.layers[i].kProj.biases.Data(), float64ToFloat32(p.layers[i].kB))
		copy(m.layers[i].vProj.weights.Data(), float64ToFloat32(p.layers[i].vW))
		copy(m.layers[i].vProj.biases.Data(), float64ToFloat32(p.layers[i].vB))
		copy(m.layers[i].oProj.weights.Data(), float64ToFloat32(p.layers[i].oW))
		copy(m.layers[i].oProj.biases.Data(), float64ToFloat32(p.layers[i].oB))
		copy(m.layers[i].ffn1.weights.Data(), float64ToFloat32(p.layers[i].ffn1W))
		copy(m.layers[i].ffn1.biases.Data(), float64ToFloat32(p.layers[i].ffn1B))
		copy(m.layers[i].ffn2.weights.Data(), float64ToFloat32(p.layers[i].ffn2W))
		copy(m.layers[i].ffn2.biases.Data(), float64ToFloat32(p.layers[i].ffn2B))
		copy(m.layers[i].norm1.Data(), float64ToFloat32(p.layers[i].norm1))
		copy(m.layers[i].bias1.Data(), float64ToFloat32(p.layers[i].bias1))
		copy(m.layers[i].norm2.Data(), float64ToFloat32(p.layers[i].norm2))
		copy(m.layers[i].bias2.Data(), float64ToFloat32(p.layers[i].bias2))
	}

	copy(m.head.weights.Data(), float64ToFloat32(p.headW))
	copy(m.head.biases.Data(), float64ToFloat32(p.headB))
}

// flatParams returns pointers to all trainable parameters in a flat slice (float64).
// The order is: patchEmbW, patchEmbB, posEmb, per-layer (qW,qB,kW,kB,vW,vB,oW,oB,
// ffn1W,ffn1B,ffn2W,ffn2B,norm1,bias1,norm2,bias2), headW, headB.
func (p *patchTSTParamsF64) flatParams() []*float64 {
	n := p.paramCount()
	ptrs := make([]*float64, 0, n)
	for i := range p.patchEmbW {
		ptrs = append(ptrs, &p.patchEmbW[i])
	}
	for i := range p.patchEmbB {
		ptrs = append(ptrs, &p.patchEmbB[i])
	}
	for i := range p.posEmb {
		ptrs = append(ptrs, &p.posEmb[i])
	}
	for l := range p.layers {
		for i := range p.layers[l].qW {
			ptrs = append(ptrs, &p.layers[l].qW[i])
		}
		for i := range p.layers[l].qB {
			ptrs = append(ptrs, &p.layers[l].qB[i])
		}
		for i := range p.layers[l].kW {
			ptrs = append(ptrs, &p.layers[l].kW[i])
		}
		for i := range p.layers[l].kB {
			ptrs = append(ptrs, &p.layers[l].kB[i])
		}
		for i := range p.layers[l].vW {
			ptrs = append(ptrs, &p.layers[l].vW[i])
		}
		for i := range p.layers[l].vB {
			ptrs = append(ptrs, &p.layers[l].vB[i])
		}
		for i := range p.layers[l].oW {
			ptrs = append(ptrs, &p.layers[l].oW[i])
		}
		for i := range p.layers[l].oB {
			ptrs = append(ptrs, &p.layers[l].oB[i])
		}
		for i := range p.layers[l].ffn1W {
			ptrs = append(ptrs, &p.layers[l].ffn1W[i])
		}
		for i := range p.layers[l].ffn1B {
			ptrs = append(ptrs, &p.layers[l].ffn1B[i])
		}
		for i := range p.layers[l].ffn2W {
			ptrs = append(ptrs, &p.layers[l].ffn2W[i])
		}
		for i := range p.layers[l].ffn2B {
			ptrs = append(ptrs, &p.layers[l].ffn2B[i])
		}
		for i := range p.layers[l].norm1 {
			ptrs = append(ptrs, &p.layers[l].norm1[i])
		}
		for i := range p.layers[l].bias1 {
			ptrs = append(ptrs, &p.layers[l].bias1[i])
		}
		for i := range p.layers[l].norm2 {
			ptrs = append(ptrs, &p.layers[l].norm2[i])
		}
		for i := range p.layers[l].bias2 {
			ptrs = append(ptrs, &p.layers[l].bias2[i])
		}
	}
	for i := range p.headW {
		ptrs = append(ptrs, &p.headW[i])
	}
	for i := range p.headB {
		ptrs = append(ptrs, &p.headB[i])
	}
	return ptrs
}

// paramCount returns the total number of trainable parameters.
func (p *patchTSTParamsF64) paramCount() int {
	n := len(p.patchEmbW) + len(p.patchEmbB) + len(p.posEmb)
	for _, l := range p.layers {
		n += len(l.qW) + len(l.qB) + len(l.kW) + len(l.kB) +
			len(l.vW) + len(l.vB) + len(l.oW) + len(l.oB) +
			len(l.ffn1W) + len(l.ffn1B) + len(l.ffn2W) + len(l.ffn2B) +
			len(l.norm1) + len(l.bias1) + len(l.norm2) + len(l.bias2)
	}
	n += len(p.headW) + len(p.headB)
	return n
}

// forwardF64 runs the PatchTST forward pass purely in float64 (no ztensor).
// input: [channels][inputLen]. Returns flat output of length outputDim.
func (m *PatchTST) forwardF64(input [][]float64, params *patchTSTParamsF64) []float64 {
	numPatches := m.config.NumPatches()
	dModel := m.config.DModel
	nHeads := m.config.NHeads
	headDim := dModel / nHeads
	ffnDim := dModel * 4

	// For channel-independent mode, process each channel separately and average.
	// For simplicity and following DLinear pattern, treat as single-sample:
	// flatten channels * inputLen and process channel 0 only for univariate,
	// or average channels for multivariate.
	// Actually, following the Forward() method pattern: each channel is processed
	// independently through the same transformer. Output is per-channel.
	// For training with [channels][inputLen] -> outputDim labels, we average channels.

	channels := len(input)
	chanOutputs := make([][]float64, channels)

	for ch := 0; ch < channels; ch++ {
		// Extract patches from this channel.
		patches := make([][]float64, numPatches)
		for p := 0; p < numPatches; p++ {
			start := p * m.config.Stride
			patches[p] = make([]float64, m.config.PatchLength)
			copy(patches[p], input[ch][start:start+m.config.PatchLength])
		}

		// Patch embedding: each patch through linear [patchLen -> dModel].
		embedded := make([][]float64, numPatches)
		for p := 0; p < numPatches; p++ {
			embedded[p] = make([]float64, dModel)
			for j := 0; j < dModel; j++ {
				val := params.patchEmbB[j]
				for k := 0; k < m.config.PatchLength; k++ {
					val += patches[p][k] * params.patchEmbW[k*dModel+j]
				}
				embedded[p][j] = val
			}
		}

		// Add positional embeddings.
		for p := 0; p < numPatches; p++ {
			for j := 0; j < dModel; j++ {
				embedded[p][j] += params.posEmb[p*dModel+j]
			}
		}

		// Transformer encoder layers.
		x := embedded // [numPatches][dModel]
		for li := 0; li < m.config.NLayers; li++ {
			layer := &params.layers[li]

			// Pre-norm 1 (layer norm).
			normed := layerNormF64(x, layer.norm1, layer.bias1, dModel)

			// Multi-head self-attention.
			attnOut := multiHeadAttentionF64(normed, layer, nHeads, headDim, dModel)

			// Residual.
			for p := 0; p < numPatches; p++ {
				for j := 0; j < dModel; j++ {
					x[p][j] += attnOut[p][j]
				}
			}

			// Pre-norm 2.
			normed = layerNormF64(x, layer.norm2, layer.bias2, dModel)

			// FFN.
			ffnOut := make([][]float64, numPatches)
			for p := 0; p < numPatches; p++ {
				// Linear 1: dModel -> ffnDim.
				h := make([]float64, ffnDim)
				for j := 0; j < ffnDim; j++ {
					val := layer.ffn1B[j]
					for k := 0; k < dModel; k++ {
						val += normed[p][k] * layer.ffn1W[k*ffnDim+j]
					}
					h[j] = geluF64(val)
				}
				// Linear 2: ffnDim -> dModel.
				ffnOut[p] = make([]float64, dModel)
				for j := 0; j < dModel; j++ {
					val := layer.ffn2B[j]
					for k := 0; k < ffnDim; k++ {
						val += h[k] * layer.ffn2W[k*dModel+j]
					}
					ffnOut[p][j] = val
				}
			}

			// Residual.
			for p := 0; p < numPatches; p++ {
				for j := 0; j < dModel; j++ {
					x[p][j] += ffnOut[p][j]
				}
			}
		}

		// Flatten: [numPatches * dModel].
		flat := make([]float64, numPatches*dModel)
		for p := 0; p < numPatches; p++ {
			copy(flat[p*dModel:(p+1)*dModel], x[p])
		}

		// Output head: [numPatches*dModel -> outputDim].
		headIn := numPatches * dModel
		out := make([]float64, m.config.OutputDim)
		for j := 0; j < m.config.OutputDim; j++ {
			val := params.headB[j]
			for k := 0; k < headIn; k++ {
				val += flat[k] * params.headW[k*m.config.OutputDim+j]
			}
			out[j] = val
		}
		chanOutputs[ch] = out
	}

	// Average across channels.
	result := make([]float64, m.config.OutputDim)
	for ch := 0; ch < channels; ch++ {
		for j := 0; j < m.config.OutputDim; j++ {
			result[j] += chanOutputs[ch][j]
		}
	}
	for j := range result {
		result[j] /= float64(channels)
	}
	return result
}

// layerNormF64 applies layer normalization in float64.
// x: [seq][dModel], scale/bias: [dModel].
func layerNormF64(x [][]float64, scale, bias []float64, dModel int) [][]float64 {
	seq := len(x)
	out := make([][]float64, seq)
	for s := 0; s < seq; s++ {
		// Mean.
		mean := 0.0
		for j := 0; j < dModel; j++ {
			mean += x[s][j]
		}
		mean /= float64(dModel)

		// Variance.
		variance := 0.0
		for j := 0; j < dModel; j++ {
			d := x[s][j] - mean
			variance += d * d
		}
		variance /= float64(dModel)

		invStd := 1.0 / math.Sqrt(variance+1e-5)
		out[s] = make([]float64, dModel)
		for j := 0; j < dModel; j++ {
			out[s][j] = (x[s][j]-mean)*invStd*scale[j] + bias[j]
		}
	}
	return out
}

// multiHeadAttentionF64 computes multi-head self-attention in float64.
// x: [seq][dModel].
func multiHeadAttentionF64(x [][]float64, layer *encoderLayerF64, nHeads, headDim, dModel int) [][]float64 {
	seq := len(x)

	// Q, K, V projections.
	q := linearF64(x, layer.qW, layer.qB, dModel, dModel)
	k := linearF64(x, layer.kW, layer.kB, dModel, dModel)
	v := linearF64(x, layer.vW, layer.vB, dModel, dModel)

	// Split into heads and compute attention.
	attnOut := make([][]float64, seq)
	for s := range attnOut {
		attnOut[s] = make([]float64, dModel)
	}

	scale := 1.0 / math.Sqrt(float64(headDim))
	for h := 0; h < nHeads; h++ {
		hOff := h * headDim

		// Compute attention scores for this head.
		scores := make([][]float64, seq)
		for i := 0; i < seq; i++ {
			scores[i] = make([]float64, seq)
			for j := 0; j < seq; j++ {
				dot := 0.0
				for d := 0; d < headDim; d++ {
					dot += q[i][hOff+d] * k[j][hOff+d]
				}
				scores[i][j] = dot * scale
			}
		}

		// Softmax.
		for i := 0; i < seq; i++ {
			maxScore := scores[i][0]
			for j := 1; j < seq; j++ {
				if scores[i][j] > maxScore {
					maxScore = scores[i][j]
				}
			}
			sumExp := 0.0
			for j := 0; j < seq; j++ {
				scores[i][j] = math.Exp(scores[i][j] - maxScore)
				sumExp += scores[i][j]
			}
			for j := 0; j < seq; j++ {
				scores[i][j] /= sumExp
			}
		}

		// Weighted sum of values.
		for i := 0; i < seq; i++ {
			for d := 0; d < headDim; d++ {
				val := 0.0
				for j := 0; j < seq; j++ {
					val += scores[i][j] * v[j][hOff+d]
				}
				attnOut[i][hOff+d] = val
			}
		}
	}

	// Output projection.
	return linearF64(attnOut, layer.oW, layer.oB, dModel, dModel)
}

// linearF64 computes x @ W + b in float64.
// x: [n][inDim], W: [inDim*outDim] (row-major), b: [outDim].
func linearF64(x [][]float64, w, b []float64, inDim, outDim int) [][]float64 {
	n := len(x)
	out := make([][]float64, n)
	for i := 0; i < n; i++ {
		out[i] = make([]float64, outDim)
		for j := 0; j < outDim; j++ {
			val := b[j]
			for k := 0; k < inDim; k++ {
				val += x[i][k] * w[k*outDim+j]
			}
			out[i][j] = val
		}
	}
	return out
}
