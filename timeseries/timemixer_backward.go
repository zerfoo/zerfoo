package timeseries

import (
	"context"

	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/zerfoo/layers/functional"
)

// timeMixerCache stores intermediate activations needed for the backward pass.
type timeMixerCache struct {
	// input is the original input [numFeatures][inputLen].
	input [][]float64

	// decomposed holds the raw trend/seasonal decomposition at each scale
	// (before mixing).
	decomposed []scaleDecomposition

	// mixingCaches holds per-layer intermediate values from pastDecomposableMixing.
	mixingCaches []timeMixerMixingLayerCache
}

// timeMixerMixingLayerCache stores per-layer intermediates for the mixing MLPs.
type timeMixerMixingLayerCache struct {
	// scalesIn holds the input scales to this layer [numScales].trend/seasonal.
	scalesIn []scaleDecomposition

	// seasonalHidden[f][t] is the hidden activations [hiddenSize] for the seasonal MLP.
	// Indexed [numFeatures][seqLen][hiddenSize].
	seasonalHidden [][][]float64

	// seasonalPreReLU[f][t] is the pre-ReLU activations [hiddenSize] for the seasonal MLP.
	seasonalPreReLU [][][]float64

	// trendHidden/trendPreReLU: same for trend MLP.
	trendHidden  [][][]float64
	trendPreReLU [][][]float64

	// mlpOutSeasonal[s][f][t] is the MLP output for seasonal at scale s.
	mlpOutSeasonal [][][]float64
	// mlpOutTrend[s][f][t] is the MLP output for trend at scale s.
	mlpOutTrend [][][]float64
}

// forwardWithCache runs the TimeMixer forward pass while caching all
// intermediate activations needed for backward.
func (m *TimeMixer) forwardWithCache(input [][]float64) (*MultiScaleOutput, *timeMixerCache) {
	cache := &timeMixerCache{
		input: input,
	}

	// Decompose.
	scales := m.decompose(input)
	cache.decomposed = make([]scaleDecomposition, len(scales))
	for s := range scales {
		cache.decomposed[s] = copyScaleDecomp(scales[s])
	}

	// Past decomposable mixing with caching.
	mixed, mixCaches := m.pastDecomposableMixingWithCache(scales)
	cache.mixingCaches = mixCaches

	return &MultiScaleOutput{Scales: mixed}, cache
}

// copyScaleDecomp makes a deep copy of a scaleDecomposition.
func copyScaleDecomp(sd scaleDecomposition) scaleDecomposition {
	cp := scaleDecomposition{
		trend:    make([][]float64, len(sd.trend)),
		seasonal: make([][]float64, len(sd.seasonal)),
	}
	for f := range sd.trend {
		cp.trend[f] = make([]float64, len(sd.trend[f]))
		copy(cp.trend[f], sd.trend[f])
	}
	for f := range sd.seasonal {
		cp.seasonal[f] = make([]float64, len(sd.seasonal[f]))
		copy(cp.seasonal[f], sd.seasonal[f])
	}
	return cp
}

// pastDecomposableMixingWithCache runs the mixing layers and caches intermediates.
func (m *TimeMixer) pastDecomposableMixingWithCache(scales []scaleDecomposition) ([]scaleDecomposition, []timeMixerMixingLayerCache) {
	numScales := len(scales)
	nf := len(scales[0].trend)
	seqLen := len(scales[0].trend[0])
	hiddenSize := m.config.HiddenSize

	caches := make([]timeMixerMixingLayerCache, m.config.NumLayers)

	for l := 0; l < m.config.NumLayers; l++ {
		seasonalMLP := m.seasonalMLPs[l]
		trendMLP := m.trendMLPs[l]

		lc := timeMixerMixingLayerCache{
			scalesIn:        make([]scaleDecomposition, numScales),
			seasonalHidden:  make([][][]float64, nf),
			seasonalPreReLU: make([][][]float64, nf),
			trendHidden:     make([][][]float64, nf),
			trendPreReLU:    make([][][]float64, nf),
			mlpOutSeasonal:  make([][][]float64, numScales),
			mlpOutTrend:     make([][][]float64, numScales),
		}

		// Cache input scales.
		for s := 0; s < numScales; s++ {
			lc.scalesIn[s] = copyScaleDecomp(scales[s])
			lc.mlpOutSeasonal[s] = make([][]float64, nf)
			lc.mlpOutTrend[s] = make([][]float64, nf)
			for f := 0; f < nf; f++ {
				lc.mlpOutSeasonal[s][f] = make([]float64, seqLen)
				lc.mlpOutTrend[s][f] = make([]float64, seqLen)
			}
		}

		for f := 0; f < nf; f++ {
			lc.seasonalHidden[f] = make([][]float64, seqLen)
			lc.seasonalPreReLU[f] = make([][]float64, seqLen)
			lc.trendHidden[f] = make([][]float64, seqLen)
			lc.trendPreReLU[f] = make([][]float64, seqLen)
		}

		// New outputs.
		newSeasonal := make([]scaleDecomposition, numScales)
		newTrend := make([]scaleDecomposition, numScales)
		for s := 0; s < numScales; s++ {
			newSeasonal[s] = scaleDecomposition{
				trend:    scales[s].trend,
				seasonal: make([][]float64, nf),
			}
			newTrend[s] = scaleDecomposition{
				trend:    make([][]float64, nf),
				seasonal: scales[s].seasonal,
			}
			for f := 0; f < nf; f++ {
				newSeasonal[s].seasonal[f] = make([]float64, seqLen)
				newTrend[s].trend[f] = make([]float64, seqLen)
			}
		}

		scaleVec := make([]float64, numScales)

		// Mix seasonal across scales with caching.
		for f := 0; f < nf; f++ {
			for t := 0; t < seqLen; t++ {
				for s := 0; s < numScales; s++ {
					scaleVec[s] = scales[s].seasonal[f][t]
				}
				hidden, preReLU, out := mlpForwardWithCache(seasonalMLP, scaleVec)
				lc.seasonalHidden[f][t] = hidden
				lc.seasonalPreReLU[f][t] = preReLU
				for s := 0; s < numScales; s++ {
					newSeasonal[s].seasonal[f][t] = out[s]
					lc.mlpOutSeasonal[s][f][t] = out[s]
				}
			}
		}

		// Mix trend across scales with caching.
		for f := 0; f < nf; f++ {
			for t := 0; t < seqLen; t++ {
				for s := 0; s < numScales; s++ {
					scaleVec[s] = scales[s].trend[f][t]
				}
				hidden, preReLU, out := mlpForwardWithCache(trendMLP, scaleVec)
				lc.trendHidden[f][t] = hidden
				lc.trendPreReLU[f][t] = preReLU
				for s := 0; s < numScales; s++ {
					newTrend[s].trend[f][t] = out[s]
					lc.mlpOutTrend[s][f][t] = out[s]
				}
			}
		}

		_ = hiddenSize

		// Bottom-up residual.
		for s := numScales - 2; s >= 0; s-- {
			for f := 0; f < nf; f++ {
				for t := 0; t < seqLen; t++ {
					newSeasonal[s].seasonal[f][t] += newSeasonal[s+1].seasonal[f][t]
					newTrend[s].trend[f][t] += newTrend[s+1].trend[f][t]
				}
			}
		}

		// Assemble for next layer.
		for s := 0; s < numScales; s++ {
			scales[s] = scaleDecomposition{
				trend:    newTrend[s].trend,
				seasonal: newSeasonal[s].seasonal,
			}
		}

		caches[l] = lc
	}

	return scales, caches
}

// mlpForwardWithCache runs a mixing MLP forward and returns hidden activations,
// pre-ReLU values, and the output.
func mlpForwardWithCache(mlp *mixingMLP, x []float64) (hidden, preReLU, out []float64) {
	hiddenSize := len(mlp.b1)
	numScales := len(x)

	preReLU = make([]float64, hiddenSize)
	hidden = make([]float64, hiddenSize)
	for i := 0; i < hiddenSize; i++ {
		sum := mlp.b1[i]
		for j := 0; j < numScales; j++ {
			sum += mlp.w1[i][j] * x[j]
		}
		preReLU[i] = sum
		if sum > 0 {
			hidden[i] = sum
		}
	}

	out = make([]float64, len(mlp.b2))
	for i := range out {
		sum := mlp.b2[i]
		for j := 0; j < hiddenSize; j++ {
			sum += mlp.w2[i][j] * hidden[j]
		}
		out[i] = sum
	}
	return hidden, preReLU, out
}

// timeMixerGrads holds gradient accumulators for all learnable parameters.
type timeMixerGrads struct {
	// dMAWeights[s][k] is the gradient for maWeights[s][k].
	dMAWeights [][]float64

	// dSeasonalMLPs[l] holds gradients for seasonal MLP layer l.
	dSeasonalMLPs []mlpGrads

	// dTrendMLPs[l] holds gradients for trend MLP layer l.
	dTrendMLPs []mlpGrads
}

// mlpGrads holds gradient accumulators for a mixing MLP.
type mlpGrads struct {
	dW1 [][]float64
	dB1 []float64
	dW2 [][]float64
	dB2 []float64
}

func newTimeMixerGrads(m *TimeMixer) timeMixerGrads {
	g := timeMixerGrads{
		dMAWeights:    make([][]float64, len(m.maWeights)),
		dSeasonalMLPs: make([]mlpGrads, m.config.NumLayers),
		dTrendMLPs:    make([]mlpGrads, m.config.NumLayers),
	}
	for s := range m.maWeights {
		g.dMAWeights[s] = make([]float64, len(m.maWeights[s]))
	}
	for l := 0; l < m.config.NumLayers; l++ {
		g.dSeasonalMLPs[l] = newMLPGrads(m.seasonalMLPs[l])
		g.dTrendMLPs[l] = newMLPGrads(m.trendMLPs[l])
	}
	return g
}

func newMLPGrads(mlp *mixingMLP) mlpGrads {
	hiddenSize := len(mlp.b1)
	numScales := len(mlp.w1[0])
	return mlpGrads{
		dW1: zeroMatrix(hiddenSize, numScales),
		dB1: make([]float64, hiddenSize),
		dW2: zeroMatrix(numScales, hiddenSize),
		dB2: make([]float64, numScales),
	}
}

// backward computes gradients for all learnable parameters given the gradient
// of the loss w.r.t. the output (dScales). dScales has the same shape as
// MultiScaleOutput.Scales.
//
// dScales[s].trend[f][t] is dL/d(output_trend[s][f][t]).
// dScales[s].seasonal[f][t] is dL/d(output_seasonal[s][f][t]).
func (m *TimeMixer) backward(dScales []scaleDecomposition, cache *timeMixerCache, grads *timeMixerGrads) {
	numScales := m.config.NumScales
	nf := m.config.NumFeatures
	seqLen := m.config.InputLen

	// --- Backward through pastDecomposableMixing ---

	// Current gradients w.r.t. each scale's trend and seasonal.
	dCur := make([]scaleDecomposition, numScales)
	for s := 0; s < numScales; s++ {
		dCur[s] = scaleDecomposition{
			trend:    make([][]float64, nf),
			seasonal: make([][]float64, nf),
		}
		for f := 0; f < nf; f++ {
			dCur[s].trend[f] = make([]float64, seqLen)
			dCur[s].seasonal[f] = make([]float64, seqLen)
			copy(dCur[s].trend[f], dScales[s].trend[f])
			copy(dCur[s].seasonal[f], dScales[s].seasonal[f])
		}
	}

	// Backward through mixing layers in reverse order.
	for l := m.config.NumLayers - 1; l >= 0; l-- {
		lc := &cache.mixingCaches[l]
		seasonalMLP := m.seasonalMLPs[l]
		trendMLP := m.trendMLPs[l]
		dSeasMLP := &grads.dSeasonalMLPs[l]
		dTrendMLP := &grads.dTrendMLPs[l]

		// --- Backward through bottom-up residual ---
		// Forward: for s = numScales-2 down to 0:
		//   newSeasonal[s] += newSeasonal[s+1]
		//   newTrend[s] += newTrend[s+1]
		// Backward: for s = 0 up to numScales-2:
		//   dMlpOut[s+1] += dCur[s]  (because newSeasonal[s+1] was added to newSeasonal[s])
		//   dMlpOut[s] = dCur[s]

		// dMlpOutSeasonal[s][f][t] is dL/d(MLP_output_seasonal[s][f][t]) (before residual).
		dMlpOutSeasonal := make([]scaleDecomposition, numScales)
		dMlpOutTrend := make([]scaleDecomposition, numScales)
		for s := 0; s < numScales; s++ {
			dMlpOutSeasonal[s] = scaleDecomposition{
				seasonal: make([][]float64, nf),
			}
			dMlpOutTrend[s] = scaleDecomposition{
				trend: make([][]float64, nf),
			}
			for f := 0; f < nf; f++ {
				dMlpOutSeasonal[s].seasonal[f] = make([]float64, seqLen)
				dMlpOutTrend[s].trend[f] = make([]float64, seqLen)
				copy(dMlpOutSeasonal[s].seasonal[f], dCur[s].seasonal[f])
				copy(dMlpOutTrend[s].trend[f], dCur[s].trend[f])
			}
		}

		// Reverse the bottom-up residual: for s = 0 up to numScales-2,
		// the residual added s+1's output to s, so gradient flows from s to s+1.
		for s := 0; s < numScales-1; s++ {
			for f := 0; f < nf; f++ {
				for t := 0; t < seqLen; t++ {
					dMlpOutSeasonal[s+1].seasonal[f][t] += dMlpOutSeasonal[s].seasonal[f][t]
					dMlpOutTrend[s+1].trend[f][t] += dMlpOutTrend[s].trend[f][t]
				}
			}
		}

		// --- Backward through MLP for seasonal ---
		// The MLP takes scaleVec[s] = scales[s].seasonal[f][t] for all s,
		// produces mixed[s] for all s.
		// We need dL/d(scaleVec[s]) for each (f,t) to propagate to previous layer.
		dPrevSeasonal := make([]scaleDecomposition, numScales)
		dPrevTrend := make([]scaleDecomposition, numScales)
		for s := 0; s < numScales; s++ {
			dPrevSeasonal[s] = scaleDecomposition{
				seasonal: make([][]float64, nf),
			}
			dPrevTrend[s] = scaleDecomposition{
				trend: make([][]float64, nf),
			}
			for f := 0; f < nf; f++ {
				dPrevSeasonal[s].seasonal[f] = make([]float64, seqLen)
				dPrevTrend[s].trend[f] = make([]float64, seqLen)
			}
		}

		for f := 0; f < nf; f++ {
			for t := 0; t < seqLen; t++ {
				// Seasonal MLP backward.
				dOut := make([]float64, numScales)
				for s := 0; s < numScales; s++ {
					dOut[s] = dMlpOutSeasonal[s].seasonal[f][t]
				}
				scaleVec := make([]float64, numScales)
				for s := 0; s < numScales; s++ {
					scaleVec[s] = lc.scalesIn[s].seasonal[f][t]
				}
				dInput := mlpBackward(seasonalMLP, dOut, scaleVec, lc.seasonalHidden[f][t], lc.seasonalPreReLU[f][t], dSeasMLP)
				for s := 0; s < numScales; s++ {
					dPrevSeasonal[s].seasonal[f][t] = dInput[s]
				}

				// Trend MLP backward.
				for s := 0; s < numScales; s++ {
					dOut[s] = dMlpOutTrend[s].trend[f][t]
				}
				for s := 0; s < numScales; s++ {
					scaleVec[s] = lc.scalesIn[s].trend[f][t]
				}
				dInput = mlpBackward(trendMLP, dOut, scaleVec, lc.trendHidden[f][t], lc.trendPreReLU[f][t], dTrendMLP)
				for s := 0; s < numScales; s++ {
					dPrevTrend[s].trend[f][t] = dInput[s]
				}
			}
		}

		// Update dCur for the next (earlier) layer.
		for s := 0; s < numScales; s++ {
			for f := 0; f < nf; f++ {
				copy(dCur[s].seasonal[f], dPrevSeasonal[s].seasonal[f])
				copy(dCur[s].trend[f], dPrevTrend[s].trend[f])
			}
		}
	}

	// --- Backward through decomposition ---
	// decompose: trend[s][f] = weightedMovingAverage(input[f], maWeights[s])
	//            seasonal[s][f][i] = input[f][i] - trend[s][f][i]
	//
	// dL/d(input[f][i]) += dCur[s].seasonal[f][i] * d(seasonal)/d(input)
	//                    + dCur[s].trend[f][i] * d(trend)/d(input)
	//
	// d(seasonal[s][f][i])/d(input[f][i]) = 1 - d(trend[s][f][i])/d(input[f][i])
	// d(trend[s][f][i])/d(input[f][j]) = toeplitz[i][j] (from the causal conv)
	//
	// dL/d(maWeights[s][k]) = sum over f,i of dCur[s].trend[f][i] * d(trend)/d(kernel[k])
	//                       + sum over f,i of dCur[s].seasonal[f][i] * d(seasonal)/d(kernel[k])
	// Since seasonal = input - trend:
	//   d(seasonal)/d(kernel) = -d(trend)/d(kernel)
	// So: dL/d(maWeights[s][k]) = sum_f,i (dCur[s].trend[f][i] - dCur[s].seasonal[f][i]) * d(trend)/d(kernel[k])

	for s := 0; s < numScales; s++ {
		kernel := m.maWeights[s]
		k := len(kernel)

		for f := 0; f < nf; f++ {
			input := cache.input[f]

			// dL/d(kernel[j]) for this scale and feature.
			// trend[i] = sum_{j=0}^{k-1} kernel[j] * input[max(0, i-j)]
			// dTrend[i]/dKernel[j] = input[max(0, i-j)]
			//
			// Combined gradient from trend and seasonal:
			// dL/d(kernel[j]) += sum_i (dTrend[i] - dSeasonal[i]) * input[max(0, i-j)]
			// where dTrend[i] = dCur[s].trend[f][i], dSeasonal[i] = dCur[s].seasonal[f][i]
			for j := 0; j < k; j++ {
				sum := 0.0
				for i := 0; i < seqLen; i++ {
					idx := i - j
					if idx < 0 {
						idx = 0
					}
					dTrendI := dCur[s].trend[f][i]
					dSeasonalI := dCur[s].seasonal[f][i]
					sum += (dTrendI - dSeasonalI) * input[idx]
				}
				grads.dMAWeights[s][j] += sum
			}
		}
	}
}

// mlpBackward computes backward pass for a two-layer MLP with ReLU
// via functional.MLPBackward.
// dOut: [numScales] gradient of loss w.r.t. MLP output.
// x: [numScales] input to MLP.
// hidden: [hiddenSize] post-ReLU hidden activations.
// preReLU: [hiddenSize] pre-ReLU hidden activations.
// Accumulates into grads.
// Returns dInput: [numScales] gradient w.r.t. input.
func mlpBackward(mlp *mixingMLP, dOut, x, hidden, preReLU []float64, grads *mlpGrads) []float64 {
	hiddenSize := len(mlp.b1)
	numScales := len(x)
	ctx := context.Background()
	ops := numeric.Float64Ops{}

	// Convert slices to tensors. MLPBackward expects batch dim, so shape [1, N].
	dOutT, _ := tensor.New[float64]([]int{1, numScales}, dOut)
	inputT, _ := tensor.New[float64]([]int{1, numScales}, x)

	// Flatten 2D weight slices to 1D for tensor creation.
	w1Flat := make([]float64, hiddenSize*numScales)
	for i := 0; i < hiddenSize; i++ {
		copy(w1Flat[i*numScales:], mlp.w1[i])
	}
	w1T, _ := tensor.New[float64]([]int{hiddenSize, numScales}, w1Flat)
	b1T, _ := tensor.New[float64]([]int{hiddenSize}, mlp.b1)

	w2Flat := make([]float64, numScales*hiddenSize)
	for i := 0; i < numScales; i++ {
		copy(w2Flat[i*hiddenSize:], mlp.w2[i])
	}
	w2T, _ := tensor.New[float64]([]int{numScales, hiddenSize}, w2Flat)
	b2T, _ := tensor.New[float64]([]int{numScales}, mlp.b2)

	// hidden param of MLPBackward = pre-activation (Linear1 output).
	// activated param = post-activation (post-ReLU).
	hiddenT, _ := tensor.New[float64]([]int{1, hiddenSize}, preReLU)
	activatedT, _ := tensor.New[float64]([]int{1, hiddenSize}, hidden)

	dInput, dW1, dB1, dW2, dB2, err := functional.MLPBackward(
		ctx, cpuEngine64, ops,
		dOutT, inputT, w1T, b1T, w2T, b2T, hiddenT, activatedT,
		"relu")
	if err != nil {
		panic("mlpBackward: " + err.Error())
	}

	// Accumulate gradients.
	dW1Data := dW1.Data()
	for i := 0; i < hiddenSize; i++ {
		for j := 0; j < numScales; j++ {
			grads.dW1[i][j] += dW1Data[i*numScales+j]
		}
	}
	dB1Data := dB1.Data()
	for i := 0; i < hiddenSize; i++ {
		grads.dB1[i] += dB1Data[i]
	}
	dW2Data := dW2.Data()
	for i := 0; i < numScales; i++ {
		for j := 0; j < hiddenSize; j++ {
			grads.dW2[i][j] += dW2Data[i*hiddenSize+j]
		}
	}
	dB2Data := dB2.Data()
	for i := 0; i < numScales; i++ {
		grads.dB2[i] += dB2Data[i]
	}

	return dInput.Data()
}

// collectGrads flattens the gradient accumulators in the same order as FlatParams.
func (g *timeMixerGrads) collectGrads(m *TimeMixer) []float64 {
	var grads []float64

	// MA weights.
	for s := range g.dMAWeights {
		grads = append(grads, g.dMAWeights[s]...)
	}

	// MLP gradients per layer.
	for l := 0; l < m.config.NumLayers; l++ {
		grads = append(grads, collectMLPGrads(&g.dSeasonalMLPs[l])...)
		grads = append(grads, collectMLPGrads(&g.dTrendMLPs[l])...)
	}

	return grads
}

// collectMLPGrads flattens MLP gradients in the same order as flatMLP.
func collectMLPGrads(g *mlpGrads) []float64 {
	var grads []float64
	for i := range g.dW1 {
		grads = append(grads, g.dW1[i]...)
	}
	grads = append(grads, g.dB1...)
	for i := range g.dW2 {
		grads = append(grads, g.dW2[i]...)
	}
	grads = append(grads, g.dB2...)
	return grads
}

// predict computes the final prediction from a MultiScaleOutput by averaging
// trend across scales and taking the last outputLen timesteps.
// Returns [numFeatures][outputLen].
func (m *TimeMixer) predict(msOut *MultiScaleOutput) [][]float64 {
	nf := m.config.NumFeatures
	outLen := m.config.OutputLen
	if outLen <= 0 {
		outLen = m.config.InputLen
	}
	nScales := float64(len(msOut.Scales))

	result := make([][]float64, nf)
	for f := 0; f < nf; f++ {
		result[f] = make([]float64, outLen)
		for i := 0; i < outLen; i++ {
			srcIdx := m.config.InputLen - outLen + i
			if srcIdx < 0 {
				srcIdx = 0
			}
			avg := 0.0
			for _, sc := range msOut.Scales {
				avg += sc.trend[f][srcIdx]
			}
			result[f][i] = avg / nScales
		}
	}
	return result
}
