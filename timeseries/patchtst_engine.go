package timeseries

import (
	"context"
	"fmt"
	"math"

	"github.com/zerfoo/ztensor/tensor"
)

// matMulEngine performs matrix multiplication via the compute engine.
// a: [M][K], b: [K][N] -> result: [M][N].
// Converts float64 inputs to float32 tensors, calls engine.MatMul, and
// converts the float32 result back to float64.
func (m *PatchTST) matMulEngine(ctx context.Context, a, b [][]float64) ([][]float64, error) {
	rows := len(a)
	if rows == 0 {
		return nil, nil
	}
	inner := len(a[0])
	cols := len(b[0])

	// Flatten a to float32.
	aFlat := make([]float32, rows*inner)
	for i, row := range a {
		off := i * inner
		for j, v := range row {
			aFlat[off+j] = float32(v)
		}
	}

	// Flatten b to float32.
	bFlat := make([]float32, inner*cols)
	for i, row := range b {
		off := i * cols
		for j, v := range row {
			bFlat[off+j] = float32(v)
		}
	}

	aTensor, err := tensor.New[float32]([]int{rows, inner}, aFlat)
	if err != nil {
		return nil, fmt.Errorf("matMulEngine: create a tensor: %w", err)
	}
	bTensor, err := tensor.New[float32]([]int{inner, cols}, bFlat)
	if err != nil {
		return nil, fmt.Errorf("matMulEngine: create b tensor: %w", err)
	}

	cTensor, err := m.engine.MatMul(ctx, aTensor, bTensor)
	if err != nil {
		return nil, fmt.Errorf("matMulEngine: matmul: %w", err)
	}

	// Convert result back to [][]float64.
	cData := cTensor.Data()
	result := make([][]float64, rows)
	for i := 0; i < rows; i++ {
		result[i] = make([]float64, cols)
		off := i * cols
		for j := 0; j < cols; j++ {
			result[i][j] = float64(cData[off+j])
		}
	}
	return result, nil
}

// linearF64Engine computes x @ W + b using the engine for the MatMul.
// x: [n][inDim], W: [inDim*outDim] (row-major), b: [outDim].
// The matrix multiplication is performed via engine.MatMul in float32;
// bias addition remains in float64.
func (m *PatchTST) linearF64Engine(ctx context.Context, x [][]float64, w, b []float64, inDim, outDim int) ([][]float64, error) {
	n := len(x)

	// Reshape flat w [inDim*outDim] into [inDim][outDim] for matMulEngine.
	wMat := make([][]float64, inDim)
	for i := 0; i < inDim; i++ {
		wMat[i] = w[i*outDim : (i+1)*outDim]
	}

	out, err := m.matMulEngine(ctx, x, wMat)
	if err != nil {
		return nil, err
	}

	// Add bias in float64.
	for i := 0; i < n; i++ {
		for j := 0; j < outDim; j++ {
			out[i][j] += b[j]
		}
	}
	return out, nil
}

// forwardF64WithCacheEngine runs the PatchTST forward pass in float64, using
// the engine for MatMul operations (Q/K/V/O projections, FFN layers, patch
// embedding, and output head). Softmax, GELU, layer norm, and residual
// connections remain on CPU. The cached activations are identical in structure
// to forwardF64WithCache so that backwardF64 can consume them unchanged.
func (m *PatchTST) forwardF64WithCacheEngine(ctx context.Context, input [][]float64, params *patchTSTParamsF64) ([]float64, *patchTSTCacheF64, error) {
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
		var err error
		cc.embedded, err = m.linearF64Engine(ctx, cc.patches, params.patchEmbW, params.patchEmbB, m.config.PatchLength, dModel)
		if err != nil {
			return nil, nil, fmt.Errorf("patch embedding: %w", err)
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

			// Multi-head self-attention: Q/K/V projections via engine.
			lc.q, err = m.linearF64Engine(ctx, lc.normed1, layer.qW, layer.qB, dModel, dModel)
			if err != nil {
				return nil, nil, fmt.Errorf("layer %d q proj: %w", li, err)
			}
			lc.k, err = m.linearF64Engine(ctx, lc.normed1, layer.kW, layer.kB, dModel, dModel)
			if err != nil {
				return nil, nil, fmt.Errorf("layer %d k proj: %w", li, err)
			}
			lc.v, err = m.linearF64Engine(ctx, lc.normed1, layer.vW, layer.vB, dModel, dModel)
			if err != nil {
				return nil, nil, fmt.Errorf("layer %d v proj: %w", li, err)
			}

			// Compute attention per head (softmax + weighted sum on CPU).
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

			// Output projection via engine.
			lc.attnProjOut, err = m.linearF64Engine(ctx, lc.attnOut, layer.oW, layer.oB, dModel, dModel)
			if err != nil {
				return nil, nil, fmt.Errorf("layer %d o proj: %w", li, err)
			}

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

			// FFN layer 1 via engine (matmul + bias), then GELU on CPU.
			ffn1Raw, err := m.linearF64Engine(ctx, lc.normed2, layer.ffn1W, layer.ffn1B, dModel, ffnDim)
			if err != nil {
				return nil, nil, fmt.Errorf("layer %d ffn1: %w", li, err)
			}
			lc.ffn1PreAct = ffn1Raw
			lc.ffn1Out = make([][]float64, numPatches)
			for p := 0; p < numPatches; p++ {
				lc.ffn1Out[p] = make([]float64, ffnDim)
				for j := 0; j < ffnDim; j++ {
					lc.ffn1Out[p][j] = geluF64(ffn1Raw[p][j])
				}
			}

			// FFN layer 2 via engine.
			lc.ffn2Out, err = m.linearF64Engine(ctx, lc.ffn1Out, layer.ffn2W, layer.ffn2B, ffnDim, dModel)
			if err != nil {
				return nil, nil, fmt.Errorf("layer %d ffn2: %w", li, err)
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

		// Output head via engine.
		headIn := numPatches * dModel
		flatMat := [][]float64{cc.flatInput}
		headOut, err := m.linearF64Engine(ctx, flatMat, params.headW, params.headB, headIn, m.config.OutputDim)
		if err != nil {
			return nil, nil, fmt.Errorf("output head: %w", err)
		}
		chanOutputs[ch] = headOut[0]
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
	return result, cache, nil
}

// trainWindowedEngine runs the GPU/engine-accelerated training path.
// It uses analytical backpropagation via float64 parameters, then writes
// updated weights back to the float32 engine tensors each epoch.
// The forward pass uses engine.MatMul for all linear projections (Q/K/V/O,
// FFN, patch embedding, output head) while keeping softmax, GELU, layer norm,
// and residual connections on CPU.
func (m *PatchTST) trainWindowedEngine(windows [][][]float64, labels []float64, config TrainConfig) (*TrainResult, error) {
	ctx := context.Background()
	nSamples := len(windows)
	outDim := m.config.OutputDim

	params := m.extractParamsF64()
	nParams := params.paramCount()
	adamM := make([]float64, nParams)
	adamV := make([]float64, nParams)

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

			grads := make([]float64, nParams)
			batchLoss := 0.0

			for s := 0; s < bs; s++ {
				pred, cache, err := m.forwardF64WithCacheEngine(ctx, windows[start+s], params)
				if err != nil {
					return nil, fmt.Errorf("patchtst: engine forward at sample %d: %w", start+s, err)
				}
				sampleLabels := labels[(start+s)*outDim : (start+s+1)*outDim]

				// Compute MSE loss and dL/dPred.
				dOutput := make([]float64, outDim)
				for j := 0; j < outDim; j++ {
					diff := pred[j] - sampleLabels[j]
					batchLoss += diff * diff
					dOutput[j] = 2.0 * diff / float64(bs*outDim)
				}

				// Analytical backward pass.
				sampleGrads := m.backwardF64(dOutput, params, cache)
				for pi := range grads {
					grads[pi] += sampleGrads[pi]
				}
			}

			batchLoss /= float64(bs * outDim)
			epochLoss += batchLoss
			nBatches++

			if config.GradClip > 0 {
				norm := 0.0
				for _, g := range grads {
					norm += g * g
				}
				norm = math.Sqrt(norm)
				if norm > config.GradClip {
					scale := config.GradClip / norm
					for i := range grads {
						grads[i] *= scale
					}
				}
			}

			lr := warmupLR(config.LR, epoch, config.WarmupEpochs)
			t := float64(epoch*((nSamples+batchSize-1)/batchSize) + nBatches)
			flatP := params.flatParams()
			for i := range flatP {
				adamM[i] = config.Beta1*adamM[i] + (1-config.Beta1)*grads[i]
				adamV[i] = config.Beta2*adamV[i] + (1-config.Beta2)*grads[i]*grads[i]
				mHat := adamM[i] / (1 - math.Pow(config.Beta1, t))
				vHat := adamV[i] / (1 - math.Pow(config.Beta2, t))
				*flatP[i] = *flatP[i] - lr*(mHat/(math.Sqrt(vHat)+config.Epsilon)+config.WeightDecay*(*flatP[i]))
			}
		}

		result.LossHistory[epoch] = epochLoss / float64(nBatches)
		result.FinalLoss = result.LossHistory[epoch]

		if !isFinite(result.FinalLoss) {
			return nil, fmt.Errorf("patchtst: training diverged at epoch %d: loss=%v", epoch, result.FinalLoss)
		}
	}

	// Write optimized float64 params back to float32 tensors.
	m.writeBackF32(params)

	result.Metrics = map[string]float64{"mse": result.FinalLoss}
	return result, nil
}

// trainWindowedCPU runs the original CPU-only training path using float64
// parameters and forwardF64. Used as fallback when no engine is available.
func (m *PatchTST) trainWindowedCPU(windows [][][]float64, labels []float64, config TrainConfig) (*TrainResult, error) {
	nSamples := len(windows)

	params := m.extractParamsF64()
	nParams := params.paramCount()
	adamM := make([]float64, nParams)
	adamV := make([]float64, nParams)

	result := &TrainResult{
		LossHistory: make([]float64, config.Epochs),
	}

	batchSize := nSamples
	if config.BatchSize > 0 && config.BatchSize < nSamples {
		batchSize = config.BatchSize
	}

	outDim := m.config.OutputDim

	for epoch := 0; epoch < config.Epochs; epoch++ {
		epochLoss := 0.0
		nBatches := 0

		for start := 0; start < nSamples; start += batchSize {
			end := start + batchSize
			if end > nSamples {
				end = nSamples
			}
			bs := end - start

			grads := make([]float64, nParams)
			batchLoss := 0.0

			for s := 0; s < bs; s++ {
				pred, cache := m.forwardF64WithCache(windows[start+s], params)
				sampleLabels := labels[(start+s)*outDim : (start+s+1)*outDim]

				// Compute MSE loss and dL/dPred.
				dOutput := make([]float64, outDim)
				for j := 0; j < outDim; j++ {
					diff := pred[j] - sampleLabels[j]
					batchLoss += diff * diff
					dOutput[j] = 2.0 * diff / float64(bs*outDim)
				}

				// Analytical backward pass.
				sampleGrads := m.backwardF64(dOutput, params, cache)
				for pi := range grads {
					grads[pi] += sampleGrads[pi]
				}
			}

			batchLoss /= float64(bs * outDim)
			epochLoss += batchLoss
			nBatches++

			if config.GradClip > 0 {
				norm := 0.0
				for _, g := range grads {
					norm += g * g
				}
				norm = math.Sqrt(norm)
				if norm > config.GradClip {
					scale := config.GradClip / norm
					for i := range grads {
						grads[i] *= scale
					}
				}
			}

			lr := warmupLR(config.LR, epoch, config.WarmupEpochs)
			t := float64(epoch*((nSamples+batchSize-1)/batchSize) + nBatches)
			flatP := params.flatParams()
			for i := range flatP {
				adamM[i] = config.Beta1*adamM[i] + (1-config.Beta1)*grads[i]
				adamV[i] = config.Beta2*adamV[i] + (1-config.Beta2)*grads[i]*grads[i]
				mHat := adamM[i] / (1 - math.Pow(config.Beta1, t))
				vHat := adamV[i] / (1 - math.Pow(config.Beta2, t))
				*flatP[i] = *flatP[i] - lr*(mHat/(math.Sqrt(vHat)+config.Epsilon)+config.WeightDecay*(*flatP[i]))
			}
		}

		result.LossHistory[epoch] = epochLoss / float64(nBatches)
		result.FinalLoss = result.LossHistory[epoch]

		if !isFinite(result.FinalLoss) {
			return nil, fmt.Errorf("patchtst: training diverged at epoch %d: loss=%v", epoch, result.FinalLoss)
		}
	}

	m.writeBackF32(params)

	result.Metrics = map[string]float64{"mse": result.FinalLoss}
	return result, nil
}
