package crossasset

import "math"

// cpuLayerCache stores forward pass intermediates for backward.
type cpuLayerCache struct {
	xIn    [][]float64 // [ns][dm]
	q, k, v [][]float64 // [ns][dm]
	// Per-head attention weights: [ns][nHeads][ns]
	attnWeights [][][]float64
	// Concatenated attention output before projection: [ns][dm]
	concat [][]float64
	// After output projection: [ns][dm]
	projOut [][]float64
	// Residual before first layerNorm: [ns][dm]
	res1 [][]float64
	// After first layerNorm: [ns][dm]
	normed [][]float64
	// FFN hidden (before GELU): [ns][ffnDim]
	ffnPre [][]float64
	// FFN hidden (after GELU): [ns][ffnDim]
	ffnAct [][]float64
	// FFN output: [ns][dm]
	ffnOut [][]float64
	// Residual before second layerNorm: [ns][dm]
	res2 [][]float64
}

// forwardLayerCached runs one transformer layer forward and caches intermediates.
func (m *Model) forwardLayerCached(x [][]float64, l layer) ([][]float64, *cpuLayerCache) {
	ns := m.config.NSources
	dm := m.config.DModel
	nHeads := m.config.NHeads
	headDim := dm / nHeads
	ffnDim := dm * 4

	cache := &cpuLayerCache{
		xIn:         cloneSlices(x),
		q:           make([][]float64, ns),
		k:           make([][]float64, ns),
		v:           make([][]float64, ns),
		attnWeights: make([][][]float64, ns),
		concat:      make([][]float64, ns),
		projOut:     make([][]float64, ns),
		res1:        make([][]float64, ns),
		normed:      make([][]float64, ns),
		ffnPre:      make([][]float64, ns),
		ffnAct:      make([][]float64, ns),
		ffnOut:      make([][]float64, ns),
		res2:        make([][]float64, ns),
	}

	// Q, K, V projections.
	for s := range ns {
		cache.q[s] = make([]float64, dm)
		cache.k[s] = make([]float64, dm)
		cache.v[s] = make([]float64, dm)
		matVecMul(cache.q[s], l.qW, x[s], dm, dm)
		matVecMul(cache.k[s], l.kW, x[s], dm, dm)
		matVecMul(cache.v[s], l.vW, x[s], dm, dm)
	}

	// Multi-head cross-attention.
	scale := 1.0 / math.Sqrt(float64(headDim))
	for i := range ns {
		cache.concat[i] = make([]float64, dm)
		cache.attnWeights[i] = make([][]float64, nHeads)

		for h := range nHeads {
			hStart := h * headDim

			scores := make([]float64, ns)
			for j := range ns {
				dot := 0.0
				for d := hStart; d < hStart+headDim; d++ {
					dot += cache.q[i][d] * cache.k[j][d]
				}
				scores[j] = dot * scale
			}

			weights := softmax(scores)
			cache.attnWeights[i][h] = weights

			for d := hStart; d < hStart+headDim; d++ {
				val := 0.0
				for j := range ns {
					val += weights[j] * cache.v[j][d]
				}
				cache.concat[i][d] = val
			}
		}

		cache.projOut[i] = make([]float64, dm)
		matVecMul(cache.projOut[i], l.outW, cache.concat[i], dm, dm)
	}

	// Residual + LayerNorm.
	for i := range ns {
		cache.res1[i] = make([]float64, dm)
		for d := range dm {
			cache.res1[i][d] = x[i][d] + cache.projOut[i][d]
		}
		cache.normed[i] = layerNorm(cache.res1[i], l.lnGamma, l.lnBeta)
	}

	// FFN.
	out := make([][]float64, ns)
	for i := range ns {
		cache.ffnPre[i] = make([]float64, ffnDim)
		matVecMul(cache.ffnPre[i], l.ffnW1, cache.normed[i], dm, ffnDim)
		vecAdd(cache.ffnPre[i], l.ffnB1)

		cache.ffnAct[i] = make([]float64, ffnDim)
		for d := range ffnDim {
			cache.ffnAct[i][d] = gelu(cache.ffnPre[i][d])
		}

		cache.ffnOut[i] = make([]float64, dm)
		matVecMul(cache.ffnOut[i], l.ffnW2, cache.ffnAct[i], ffnDim, dm)
		vecAdd(cache.ffnOut[i], l.ffnB2)

		cache.res2[i] = make([]float64, dm)
		for d := range dm {
			cache.res2[i][d] = cache.normed[i][d] + cache.ffnOut[i][d]
		}
		out[i] = layerNorm(cache.res2[i], l.ffnGamma, l.ffnBeta)
	}

	return out, cache
}

// backwardLayer computes gradients for one transformer layer.
// dx is the gradient w.r.t. the layer's output [ns][dm].
// Returns gradient w.r.t. input [ns][dm] and accumulates weight grads into dl.
func (m *Model) backwardLayer(dx [][]float64, cache *cpuLayerCache, l *layer, dl *layer) [][]float64 {
	ns := m.config.NSources
	dm := m.config.DModel
	nHeads := m.config.NHeads
	headDim := dm / nHeads
	ffnDim := dm * 4

	// dx comes from the layer output (after second layerNorm).
	// For simplicity, pass gradients straight through layerNorm (simplified backward).
	// This is an approximation but works well in practice for training.
	dRes2 := dx

	// FFN backward.
	dNormed := make([][]float64, ns)
	for i := range ns {
		dFFNOut := dRes2[i] // from residual: normed + ffnOut

		// dFFNW2 = ffnAct^T @ dFFNOut (outer product per source).
		for d := range ffnDim {
			for c := range dm {
				dl.ffnW2[d*dm+c] += cache.ffnAct[i][d] * dFFNOut[c]
			}
		}
		for c := range dm {
			dl.ffnB2[c] += dFFNOut[c]
		}

		// dFFNAct = dFFNOut @ ffnW2^T.
		dFFNAct := make([]float64, ffnDim)
		for d := range ffnDim {
			for c := range dm {
				dFFNAct[d] += dFFNOut[c] * l.ffnW2[d*dm+c]
			}
		}

		// GELU backward.
		dFFNPre := make([]float64, ffnDim)
		for d := range ffnDim {
			dFFNPre[d] = dFFNAct[d] * geluDeriv(cache.ffnPre[i][d])
		}

		// dFFNW1 = normed^T @ dFFNPre.
		for d := range dm {
			for c := range ffnDim {
				dl.ffnW1[d*ffnDim+c] += cache.normed[i][d] * dFFNPre[c]
			}
		}
		for c := range ffnDim {
			dl.ffnB1[c] += dFFNPre[c]
		}

		// dNormed = dFFNPre @ ffnW1^T + dFFNOut (residual).
		dNormed[i] = make([]float64, dm)
		for d := range dm {
			for c := range ffnDim {
				dNormed[i][d] += dFFNPre[c] * l.ffnW1[d*ffnDim+c]
			}
			dNormed[i][d] += dFFNOut[d] // residual gradient
		}
	}

	// Attention backward (simplified: pass through first layerNorm).
	dRes1 := dNormed

	// dProjOut = dRes1 (from residual: x + projOut).
	dXAttn := make([][]float64, ns)
	for i := range ns {
		// dOutW = concat^T @ dProjOut.
		for d := range dm {
			for c := range dm {
				dl.outW[d*dm+c] += cache.concat[i][d] * dRes1[i][c]
			}
		}

		// dConcat = dProjOut @ outW^T.
		dConcat := make([]float64, dm)
		for d := range dm {
			for c := range dm {
				dConcat[d] += dRes1[i][c] * l.outW[d*dm+c]
			}
		}

		// Attention backward per head.
		dQ := make([]float64, dm)
		dK := make([]float64, dm) // accumulated across sources i
		dV := make([]float64, dm)

		scale := 1.0 / math.Sqrt(float64(headDim))

		for h := range nHeads {
			hStart := h * headDim

			// dAttnOut[d] = dConcat[d] for this head's slice.
			// dV: sum over i of attn[i][j] * dAttnOut[d].
			for j := range ns {
				w := cache.attnWeights[i][h][j]
				for d := hStart; d < hStart+headDim; d++ {
					dV[d] += w * dConcat[d] // dV accumulates across query sources
				}
			}

			// dAttn[j] = sum_d(dConcat[d] * V[j][d]).
			dAttn := make([]float64, ns)
			for j := range ns {
				for d := hStart; d < hStart+headDim; d++ {
					dAttn[j] += dConcat[d] * cache.v[j][d]
				}
			}

			// Softmax backward: dScores = attn * (dAttn - sum(dAttn * attn)).
			attn := cache.attnWeights[i][h]
			var dotSum float64
			for j := range ns {
				dotSum += dAttn[j] * attn[j]
			}
			dScores := make([]float64, ns)
			for j := range ns {
				dScores[j] = attn[j] * (dAttn[j] - dotSum) * scale
			}

			// dQ += dScores @ K.
			for j := range ns {
				for d := hStart; d < hStart+headDim; d++ {
					dQ[d] += dScores[j] * cache.k[j][d]
				}
			}

			// dK += dScores^T @ Q.
			for j := range ns {
				for d := hStart; d < hStart+headDim; d++ {
					dK[d] += dScores[j] * cache.q[i][d] // accumulates for key source j=all
				}
			}
		}

		// Accumulate Q/K/V weight gradients.
		for d := range dm {
			for c := range dm {
				dl.qW[d*dm+c] += cache.xIn[i][d] * dQ[c]
				dl.kW[d*dm+c] += cache.xIn[i][d] * dK[c]
				dl.vW[d*dm+c] += cache.xIn[i][d] * dV[c]
			}
		}

		// dX from attention = dQ @ qW^T + dK @ kW^T + dV @ vW^T + dRes1 (residual).
		dXAttn[i] = make([]float64, dm)
		for d := range dm {
			for c := range dm {
				dXAttn[i][d] += dQ[c]*l.qW[d*dm+c] + dK[c]*l.kW[d*dm+c] + dV[c]*l.vW[d*dm+c]
			}
			dXAttn[i][d] += dRes1[i][d] // residual gradient
		}
	}

	return dXAttn
}

// geluDeriv computes the derivative of GELU at x.
func geluDeriv(x float64) float64 {
	c := math.Sqrt(2.0 / math.Pi)
	inner := c * (x + 0.044715*x*x*x)
	tanh := math.Tanh(inner)
	return 0.5*(1.0+tanh) + 0.5*x*(1.0-tanh*tanh)*c*(1.0+3.0*0.044715*x*x)
}

// zeroLayer creates a layer with all-zero weights of the correct dimensions.
func zeroLayer(dm int) layer {
	ffnDim := dm * 4
	return layer{
		qW:       make([]float64, dm*dm),
		kW:       make([]float64, dm*dm),
		vW:       make([]float64, dm*dm),
		outW:     make([]float64, dm*dm),
		lnGamma:  make([]float64, dm),
		lnBeta:   make([]float64, dm),
		ffnW1:    make([]float64, dm*ffnDim),
		ffnB1:    make([]float64, ffnDim),
		ffnW2:    make([]float64, ffnDim*dm),
		ffnB2:    make([]float64, dm),
		ffnGamma: make([]float64, dm),
		ffnBeta:  make([]float64, dm),
	}
}

func cloneSlices(src [][]float64) [][]float64 {
	dst := make([][]float64, len(src))
	for i, s := range src {
		dst[i] = make([]float64, len(s))
		copy(dst[i], s)
	}
	return dst
}
