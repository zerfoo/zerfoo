package timeseries

import (
	"context"
	"fmt"
	"math"
)

// linearBackwardF64EngineAccum computes backward for y = x @ W + b using the engine
// for the two matmul operations (dW = x^T @ dY, dX = dY @ W^T).
// dY: [n][outDim], x: [n][inDim], W: [inDim*outDim] row-major.
// Accumulates into dX, dW, dB.
func (m *PatchTST) linearBackwardF64EngineAccum(ctx context.Context, dY, x [][]float64, w []float64, dX [][]float64, dW, dB []float64, inDim, outDim int) error {
	return m.linearBackwardF64EngineAccumWithBufs(ctx, dY, x, w, dX, dW, dB, inDim, outDim, nil)
}

func (m *PatchTST) linearBackwardF64EngineAccumWithBufs(ctx context.Context, dY, x [][]float64, w []float64, dX [][]float64, dW, dB []float64, inDim, outDim int, bufs *matMulBufs) error {
	n := len(dY)

	// dW += x^T @ dY : [inDim][n] @ [n][outDim] = [inDim][outDim]
	xT := make([][]float64, inDim)
	for k := 0; k < inDim; k++ {
		xT[k] = make([]float64, n)
		for i := 0; i < n; i++ {
			xT[k][i] = x[i][k]
		}
	}
	dWMat, err := m.matMulEngineWithBufs(ctx, xT, dY, bufs)
	if err != nil {
		return fmt.Errorf("linearBackwardF64EngineAccum dW: %w", err)
	}
	for k := 0; k < inDim; k++ {
		for j := 0; j < outDim; j++ {
			dW[k*outDim+j] += dWMat[k][j]
		}
	}

	// dX += dY @ W^T : [n][outDim] @ [outDim][inDim] = [n][inDim]
	wT := make([][]float64, outDim)
	for j := 0; j < outDim; j++ {
		wT[j] = make([]float64, inDim)
		for k := 0; k < inDim; k++ {
			wT[j][k] = w[k*outDim+j]
		}
	}
	dXMat, err := m.matMulEngineWithBufs(ctx, dY, wT, bufs)
	if err != nil {
		return fmt.Errorf("linearBackwardF64EngineAccum dX: %w", err)
	}
	for i := 0; i < n; i++ {
		for k := 0; k < inDim; k++ {
			dX[i][k] += dXMat[i][k]
		}
	}

	// dB += sum(dY, axis=0)
	for i := 0; i < n; i++ {
		for j := 0; j < outDim; j++ {
			dB[j] += dY[i][j]
		}
	}
	return nil
}

// backwardBatchF64Engine computes analytical gradients for a batch of samples,
// using engine MatMul for all linear backward passes. It concatenates per-sample
// data across the batch for each linear projection backward, making one engine
// call per projection per layer per channel direction, regardless of batch size.
//
// dOutputs: [batchSize][outputDim] — gradient of loss w.r.t. each sample's output.
// caches: [batchSize] — per-sample forward pass caches.
// Returns gradient vector in same order as flatParams().
func (m *PatchTST) backwardBatchF64Engine(ctx context.Context, dOutputs [][]float64, params *patchTSTParamsF64, caches []*patchTSTCacheF64) ([]float64, error) {
	batchSize := len(dOutputs)
	numPatches := m.config.NumPatches()
	dModel := m.config.DModel
	nHeads := m.config.NHeads
	headDim := dModel / nHeads
	ffnDim := dModel * 4
	outDim := m.config.OutputDim
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

	// Shared buffer pool for engine MatMul calls to reduce allocation churn.
	var mmBufs matMulBufs

	channels := len(caches[0].channels)
	chanScale := 1.0 / float64(channels)

	for ch := 0; ch < channels; ch++ {
		// --- Output head backward (batched) ---
		// out[s] = flatInput[s] @ headW + headB
		// Concatenate flatInputs and dChanOuts across batch for one engine call.
		batchFlatInput := make([][]float64, batchSize)
		batchDChanOut := make([][]float64, batchSize)
		for s := 0; s < batchSize; s++ {
			cc := &caches[s].channels[ch]
			batchFlatInput[s] = cc.flatInput

			dChanOut := make([]float64, outDim)
			for j := range dChanOut {
				dChanOut[j] = dOutputs[s][j] * chanScale
			}
			batchDChanOut[s] = dChanOut
		}

		// dHeadW += flatInput^T @ dChanOut, dFlat = dChanOut @ headW^T
		batchDFlat := make([][]float64, batchSize)
		for s := 0; s < batchSize; s++ {
			batchDFlat[s] = make([]float64, headIn)
		}
		err := m.linearBackwardF64EngineAccumWithBufs(ctx, batchDChanOut, batchFlatInput, params.headW, batchDFlat, dHeadW, dHeadB, headIn, outDim, &mmBufs)
		if err != nil {
			return nil, fmt.Errorf("head backward: %w", err)
		}

		// Unflatten dFlat -> dX per sample: [numPatches][dModel]
		perSampleDX := make([][][]float64, batchSize)
		for s := 0; s < batchSize; s++ {
			dX := make([][]float64, numPatches)
			for p := 0; p < numPatches; p++ {
				dX[p] = make([]float64, dModel)
				copy(dX[p], batchDFlat[s][p*dModel:(p+1)*dModel])
			}
			perSampleDX[s] = dX
		}

		// --- Backward through encoder layers (reverse order) ---
		for li := m.config.NLayers - 1; li >= 0; li-- {
			layer := &params.layers[li]
			dg := &dLayers[li]

			// Residual 2 backward: dX flows to both FFN output and the input after attn.
			// dFFN2Out = dX (copy per sample)
			// FFN2 backward (batched): ffn2Out = ffn1Out @ ffn2W + ffn2B
			batchDFFN2Out := make([][]float64, batchSize*numPatches)
			batchFFN1Out := make([][]float64, batchSize*numPatches)
			for s := 0; s < batchSize; s++ {
				lc := &caches[s].channels[ch].layerCaches[li]
				for p := 0; p < numPatches; p++ {
					idx := s*numPatches + p
					batchDFFN2Out[idx] = make([]float64, dModel)
					copy(batchDFFN2Out[idx], perSampleDX[s][p])
					batchFFN1Out[idx] = lc.ffn1Out[p]
				}
			}

			batchDFFN1Out := make([][]float64, batchSize*numPatches)
			for i := range batchDFFN1Out {
				batchDFFN1Out[i] = make([]float64, ffnDim)
			}
			err = m.linearBackwardF64EngineAccumWithBufs(ctx, batchDFFN2Out, batchFFN1Out, layer.ffn2W, batchDFFN1Out, dg.ffn2W, dg.ffn2B, ffnDim, dModel, &mmBufs)
			if err != nil {
				return nil, fmt.Errorf("layer %d ffn2 backward: %w", li, err)
			}

			// GELU backward (CPU, per element).
			batchDFFN1PreAct := make([][]float64, batchSize*numPatches)
			for s := 0; s < batchSize; s++ {
				lc := &caches[s].channels[ch].layerCaches[li]
				for p := 0; p < numPatches; p++ {
					idx := s*numPatches + p
					batchDFFN1PreAct[idx] = make([]float64, ffnDim)
					for j := 0; j < ffnDim; j++ {
						batchDFFN1PreAct[idx][j] = batchDFFN1Out[idx][j] * geluDeriv[float64](lc.ffn1PreAct[p][j])
					}
				}
			}

			// FFN1 backward (batched): ffn1PreAct = normed2 @ ffn1W + ffn1B
			batchNormed2 := make([][]float64, batchSize*numPatches)
			for s := 0; s < batchSize; s++ {
				lc := &caches[s].channels[ch].layerCaches[li]
				for p := 0; p < numPatches; p++ {
					batchNormed2[s*numPatches+p] = lc.normed2[p]
				}
			}
			batchDNormed2 := make([][]float64, batchSize*numPatches)
			for i := range batchDNormed2 {
				batchDNormed2[i] = make([]float64, dModel)
			}
			err = m.linearBackwardF64EngineAccumWithBufs(ctx, batchDFFN1PreAct, batchNormed2, layer.ffn1W, batchDNormed2, dg.ffn1W, dg.ffn1B, dModel, ffnDim, &mmBufs)
			if err != nil {
				return nil, fmt.Errorf("layer %d ffn1 backward: %w", li, err)
			}

			// LayerNorm2 backward + residual add (CPU, per sample).
			perSampleDXAfterAttn := make([][][]float64, batchSize)
			for s := 0; s < batchSize; s++ {
				lc := &caches[s].channels[ch].layerCaches[li]
				// Split batchDNormed2 back to per-sample.
				dNormed2 := batchDNormed2[s*numPatches : (s+1)*numPatches]
				dXAfterAttn := layerNormBackwardF64(dNormed2, lc.centered2, lc.invStd2, layer.norm2, dg.norm2, dg.bias2, dModel)
				// Add residual gradient from FFN path.
				for p := 0; p < numPatches; p++ {
					for j := 0; j < dModel; j++ {
						dXAfterAttn[p][j] += perSampleDX[s][p][j]
					}
				}
				perSampleDXAfterAttn[s] = dXAfterAttn
			}

			// oProj backward (batched): attnProjOut = attnOut @ oW + oB
			batchDAttnProjOut := make([][]float64, batchSize*numPatches)
			batchAttnOut := make([][]float64, batchSize*numPatches)
			for s := 0; s < batchSize; s++ {
				lc := &caches[s].channels[ch].layerCaches[li]
				for p := 0; p < numPatches; p++ {
					idx := s*numPatches + p
					batchDAttnProjOut[idx] = make([]float64, dModel)
					copy(batchDAttnProjOut[idx], perSampleDXAfterAttn[s][p])
					batchAttnOut[idx] = lc.attnOut[p]
				}
			}
			batchDAttnOut := make([][]float64, batchSize*numPatches)
			for i := range batchDAttnOut {
				batchDAttnOut[i] = make([]float64, dModel)
			}
			err = m.linearBackwardF64EngineAccumWithBufs(ctx, batchDAttnProjOut, batchAttnOut, layer.oW, batchDAttnOut, dg.oW, dg.oB, dModel, dModel, &mmBufs)
			if err != nil {
				return nil, fmt.Errorf("layer %d oProj backward: %w", li, err)
			}

			// Attention backward (CPU with pre-allocated scratch buffers).
			// For small sequence lengths (typical: 6 patches), CPU is faster than GPU
			// due to kernel launch overhead. Pre-allocate scratch to reduce GC pressure.
			seq := numPatches
			batchDQ := make([][]float64, batchSize*numPatches)
			batchDK := make([][]float64, batchSize*numPatches)
			batchDV := make([][]float64, batchSize*numPatches)
			for i := range batchDQ {
				batchDQ[i] = make([]float64, dModel)
				batchDK[i] = make([]float64, dModel)
				batchDV[i] = make([]float64, dModel)
			}

			attnScale := 1.0 / math.Sqrt(float64(headDim))

			// Pre-allocate scratch for softmax backward (reused per sample per head).
			dScoresScratch := make([]float64, seq*seq)
			dLogitsScratch := make([]float64, seq*seq)

			for s := 0; s < batchSize; s++ {
				lc := &caches[s].channels[ch].layerCaches[li]
				dAttnOut := batchDAttnOut[s*numPatches : (s+1)*numPatches]
				dQ := batchDQ[s*numPatches : (s+1)*numPatches]
				dK := batchDK[s*numPatches : (s+1)*numPatches]
				dV := batchDV[s*numPatches : (s+1)*numPatches]

				for h := 0; h < nHeads; h++ {
					hOff := h * headDim

					// Zero scratch buffers.
					for i := range dScoresScratch {
						dScoresScratch[i] = 0
					}

					// dScores[i][j] = sum_d(dAttnOut[i][hOff+d] * V[j][hOff+d])
					// dV[j][hOff+d] += scores[h][i][j] * dAttnOut[i][hOff+d]
					for i := 0; i < seq; i++ {
						for j := 0; j < seq; j++ {
							ds := 0.0
							score := lc.scores[h][i][j]
							for d := 0; d < headDim; d++ {
								da := dAttnOut[i][hOff+d]
								ds += da * lc.v[j][hOff+d]
								dV[j][hOff+d] += score * da
							}
							dScoresScratch[i*seq+j] = ds
						}
					}

					// Softmax backward + scale: dLogits[i][j] = scores[i][j] * (dScores[i][j] - dot) * scale
					for i := 0; i < seq; i++ {
						dot := 0.0
						for j := 0; j < seq; j++ {
							dot += lc.scores[h][i][j] * dScoresScratch[i*seq+j]
						}
						for j := 0; j < seq; j++ {
							dLogitsScratch[i*seq+j] = lc.scores[h][i][j] * (dScoresScratch[i*seq+j] - dot) * attnScale
						}
					}

					// QK^T backward: dQ[i] += dLogits[i][j]*K[j], dK[j] += dLogits[i][j]*Q[i]
					for i := 0; i < seq; i++ {
						for j := 0; j < seq; j++ {
							dl := dLogitsScratch[i*seq+j]
							if dl == 0 {
								continue
							}
							for d := 0; d < headDim; d++ {
								dQ[i][hOff+d] += dl * lc.k[j][hOff+d]
								dK[j][hOff+d] += dl * lc.q[i][hOff+d]
							}
						}
					}
				}
			}

			// Q/K/V projection backward (batched): q = normed1 @ qW + qB, etc.
			batchNormed1 := make([][]float64, batchSize*numPatches)
			for s := 0; s < batchSize; s++ {
				lc := &caches[s].channels[ch].layerCaches[li]
				for p := 0; p < numPatches; p++ {
					batchNormed1[s*numPatches+p] = lc.normed1[p]
				}
			}

			batchDNormed1 := make([][]float64, batchSize*numPatches)
			for i := range batchDNormed1 {
				batchDNormed1[i] = make([]float64, dModel)
			}

			err = m.linearBackwardF64EngineAccumWithBufs(ctx, batchDQ, batchNormed1, layer.qW, batchDNormed1, dg.qW, dg.qB, dModel, dModel, &mmBufs)
			if err != nil {
				return nil, fmt.Errorf("layer %d qProj backward: %w", li, err)
			}
			err = m.linearBackwardF64EngineAccumWithBufs(ctx, batchDK, batchNormed1, layer.kW, batchDNormed1, dg.kW, dg.kB, dModel, dModel, &mmBufs)
			if err != nil {
				return nil, fmt.Errorf("layer %d kProj backward: %w", li, err)
			}
			err = m.linearBackwardF64EngineAccumWithBufs(ctx, batchDV, batchNormed1, layer.vW, batchDNormed1, dg.vW, dg.vB, dModel, dModel, &mmBufs)
			if err != nil {
				return nil, fmt.Errorf("layer %d vProj backward: %w", li, err)
			}

			// LayerNorm1 backward + residual add (CPU, per sample).
			for s := 0; s < batchSize; s++ {
				lc := &caches[s].channels[ch].layerCaches[li]
				dNormed1 := batchDNormed1[s*numPatches : (s+1)*numPatches]
				dLayerInput := layerNormBackwardF64(dNormed1, lc.centered1, lc.invStd1, layer.norm1, dg.norm1, dg.bias1, dModel)
				// Add residual gradient from attention path.
				for p := 0; p < numPatches; p++ {
					for j := 0; j < dModel; j++ {
						dLayerInput[p][j] += perSampleDXAfterAttn[s][p][j]
					}
				}
				perSampleDX[s] = dLayerInput
			}
		}

		// Positional embedding gradient (CPU, per sample).
		for s := 0; s < batchSize; s++ {
			for p := 0; p < numPatches; p++ {
				for j := 0; j < dModel; j++ {
					dPosEmb[p*dModel+j] += perSampleDX[s][p][j]
				}
			}
		}

		// Patch embedding backward (batched): embedded = patches @ patchEmbW + patchEmbB
		batchPatches := make([][]float64, batchSize*numPatches)
		batchDX := make([][]float64, batchSize*numPatches)
		for s := 0; s < batchSize; s++ {
			cc := &caches[s].channels[ch]
			for p := 0; p < numPatches; p++ {
				idx := s*numPatches + p
				batchPatches[idx] = cc.patches[p]
				batchDX[idx] = perSampleDX[s][p]
			}
		}
		// We don't need dInput for patches (input data, not parameters),
		// but linearBackwardF64EngineAccum needs a dX buffer.
		batchDPatches := make([][]float64, batchSize*numPatches)
		for i := range batchDPatches {
			batchDPatches[i] = make([]float64, m.config.PatchLength)
		}
		err = m.linearBackwardF64EngineAccumWithBufs(ctx, batchDX, batchPatches, params.patchEmbW, batchDPatches, dPatchEmbW, dPatchEmbB, m.config.PatchLength, dModel, &mmBufs)
		if err != nil {
			return nil, fmt.Errorf("patch embedding backward: %w", err)
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
	return grads, nil
}
