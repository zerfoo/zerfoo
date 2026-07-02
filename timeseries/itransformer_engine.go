package timeseries

import (
	"context"
	"fmt"
	"math"

	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/zerfoo/training/scheduler"
)

// linearBatchEngine computes Y = X @ W + bias using engine.MatMul.
// X: [rows x inDim], W: [inDim x outDim], bias: [outDim].
// Returns [rows][outDim] as float64 slices (for cache compatibility).
// Falls back to CPU linearForwardVec on any engine error.
func (m *ITransformer) linearBatchEngine(ctx context.Context, xRows [][]float64, w [][]float64, bias []float64) [][]float64 {
	rows := len(xRows)
	inDim := len(w)
	outDim := len(bias)

	// Flatten X into [rows * inDim] float32.
	xFlat := make([]float32, rows*inDim)
	for r := 0; r < rows; r++ {
		for j := 0; j < inDim; j++ {
			xFlat[r*inDim+j] = float32(xRows[r][j])
		}
	}

	// Flatten W into [inDim * outDim] float32.
	wFlat := make([]float32, inDim*outDim)
	for i := 0; i < inDim; i++ {
		for j := 0; j < outDim; j++ {
			wFlat[i*outDim+j] = float32(w[i][j])
		}
	}

	xT, err := tensor.New[float32]([]int{rows, inDim}, xFlat)
	if err != nil {
		return linearBatchCPU(xRows, w, bias)
	}
	wT, err := tensor.New[float32]([]int{inDim, outDim}, wFlat)
	if err != nil {
		return linearBatchCPU(xRows, w, bias)
	}

	yT, err := m.engine.MatMul(ctx, xT, wT)
	if err != nil {
		return linearBatchCPU(xRows, w, bias)
	}

	// Read result back and add bias.
	yData := yT.Data()
	result := make([][]float64, rows)
	for r := 0; r < rows; r++ {
		result[r] = make([]float64, outDim)
		for j := 0; j < outDim; j++ {
			result[r][j] = float64(yData[r*outDim+j]) + float64(bias[j])
		}
	}
	return result
}

// linearBatchCPU is the CPU fallback for linearBatchEngine.
func linearBatchCPU(xRows [][]float64, w [][]float64, bias []float64) [][]float64 {
	result := make([][]float64, len(xRows))
	for r := range xRows {
		result[r] = linearForwardVec(xRows[r], w, bias)
	}
	return result
}

// forwardWithCacheEngine runs the forward pass using engine.MatMul for all
// linear projections while producing the same cache layout as forwardWithCache.
// This allows the existing CPU backward pass to work unchanged.
func (m *ITransformer) forwardWithCacheEngine(ctx context.Context, input [][]float64) ([][]float64, iTransformerCache) {
	channels := m.config.Channels
	dModel := m.config.DModel
	nHeads := m.config.NHeads
	headDim := dModel / nHeads

	cache := iTransformerCache{input: input}

	// Step 1: Variate embedding — [channels x inputLen] @ [inputLen x dModel] + embedB.
	tokens := m.linearBatchEngine(ctx, input, m.embedW, m.embedB)
	cache.embedOut = copyMatrix(tokens)

	// Step 2: Encoder layers.
	cache.layerCaches = make([]iTransformerLayerCache, len(m.layers))
	for li, layer := range m.layers {
		var lc iTransformerLayerCache
		lc.inputTokens = copyMatrix(tokens)
		lc.preAttnTokens = copyMatrix(tokens)

		// --- Multi-head self-attention ---
		// Q, K, V projections: [channels x dModel] @ [dModel x dModel].
		Q := m.linearBatchEngine(ctx, tokens, layer.qW, layer.qB)
		K := m.linearBatchEngine(ctx, tokens, layer.kW, layer.kB)
		V := m.linearBatchEngine(ctx, tokens, layer.vW, layer.vB)
		lc.Q = copyMatrix(Q)
		lc.K = copyMatrix(K)
		lc.V = copyMatrix(V)

		// Per-head scaled dot-product attention (kept on CPU — small channels x channels).
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

		// Output projection: [channels x dModel] @ [dModel x dModel].
		attnOut := m.linearBatchEngine(ctx, attnConcat, layer.oW, layer.oB)
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
		// fc1: [channels x dModel] @ [dModel x dFF].
		fc1Out := m.linearBatchEngine(ctx, ln1Out, layer.fc1W, layer.fc1B)
		geluOut := make([][]float64, channels)
		for c := 0; c < channels; c++ {
			geluOut[c] = make([]float64, len(fc1Out[c]))
			for i := range fc1Out[c] {
				v := fc1Out[c][i]
					inner := math.Sqrt(2/math.Pi) * (v + 0.044715*v*v*v)
					geluOut[c][i] = 0.5 * v * (1 + math.Tanh(inner))
			}
		}

		// fc2: [channels x dFF] @ [dFF x dModel].
		fc2Out := m.linearBatchEngine(ctx, geluOut, layer.fc2W, layer.fc2B)

		preLN2 := make([][]float64, channels)
		ln2Out := make([][]float64, channels)
		lc.ln2Mu = make([]float64, channels)
		lc.ln2Std = make([]float64, channels)
		for c := 0; c < channels; c++ {
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

		tokens = ln2Out
		cache.layerCaches[li] = lc
	}

	// Store pre-projection tokens.
	cache.preProj = copyMatrix(tokens)

	// Step 3: Output projection: [channels x dModel] @ [dModel x outputLen].
	output := m.linearBatchEngine(ctx, tokens, m.projW, m.projB)

	_ = dModel
	return output, cache
}

// forwardBatchEngine runs the iTransformer forward pass on a batch of samples
// using the compute engine for linear projections.
// Input shape: [batch, channels, inputLen]. Output shape: [batch, channels, outputLen].
func (m *ITransformer) forwardBatchEngine(ctx context.Context, input *tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
	shape := input.Shape()
	if len(shape) != 3 {
		return nil, fmt.Errorf("itransformer: forwardBatchEngine expects 3D input [batch, channels, inputLen], got shape %v", shape)
	}
	batch := shape[0]
	channels := shape[1]
	inputLen := shape[2]
	if channels != m.config.Channels {
		return nil, fmt.Errorf("itransformer: expected %d channels, got %d", m.config.Channels, channels)
	}
	if inputLen != m.config.InputLen {
		return nil, fmt.Errorf("itransformer: expected inputLen %d, got %d", m.config.InputLen, inputLen)
	}
	dModel := m.config.DModel
	nHeads := m.config.NHeads
	headDim := dModel / nHeads
	outputLen := m.config.OutputLen
	data := input.Data()
	outFlat := make([]float32, batch*channels*outputLen)
	for b := 0; b < batch; b++ {
		sampleInput := make([][]float64, channels)
		for c := 0; c < channels; c++ {
			sampleInput[c] = make([]float64, inputLen)
			off := b*channels*inputLen + c*inputLen
			for i := 0; i < inputLen; i++ {
				sampleInput[c][i] = float64(data[off+i])
			}
		}
		tokens := m.linearBatchEngine(ctx, sampleInput, m.embedW, m.embedB)
		for _, layer := range m.layers {
			Q := m.linearBatchEngine(ctx, tokens, layer.qW, layer.qB)
			K := m.linearBatchEngine(ctx, tokens, layer.kW, layer.kB)
			V := m.linearBatchEngine(ctx, tokens, layer.vW, layer.vB)
			scale := 1.0 / math.Sqrt(float64(headDim))
			attnConcat := make([][]float64, channels)
			for c := range attnConcat {
				attnConcat[c] = make([]float64, dModel)
			}
			for h := 0; h < nHeads; h++ {
				hoff := h * headDim
				scores := make([][]float64, channels)
				for i := 0; i < channels; i++ {
					scores[i] = make([]float64, channels)
					for j := 0; j < channels; j++ {
						dot := 0.0
						for d := 0; d < headDim; d++ {
							dot += Q[i][hoff+d] * K[j][hoff+d]
						}
						scores[i][j] = dot * scale
					}
				}
				for i := 0; i < channels; i++ {
					scores[i] = softmaxF64(scores[i])
				}
				for i := 0; i < channels; i++ {
					for d := 0; d < headDim; d++ {
						val := 0.0
						for j := 0; j < channels; j++ {
							val += scores[i][j] * V[j][hoff+d]
						}
						attnConcat[i][hoff+d] = val
					}
				}
			}
			attnOut := m.linearBatchEngine(ctx, attnConcat, layer.oW, layer.oB)
			for c := 0; c < channels; c++ {
				for d := 0; d < dModel; d++ {
					tokens[c][d] += attnOut[c][d]
				}
				tokens[c] = layerNorm1D(tokens[c], layer.ln1Scale, layer.ln1Bias)
			}
			fc1Out := m.linearBatchEngine(ctx, tokens, layer.fc1W, layer.fc1B)
			for c := 0; c < channels; c++ {
				for i := range fc1Out[c] {
					v := fc1Out[c][i]
						inner := math.Sqrt(2/math.Pi) * (v + 0.044715*v*v*v)
						fc1Out[c][i] = 0.5 * v * (1 + math.Tanh(inner))
				}
			}
			fc2Out := m.linearBatchEngine(ctx, fc1Out, layer.fc2W, layer.fc2B)
			for c := 0; c < channels; c++ {
				for d := 0; d < dModel; d++ {
					tokens[c][d] += fc2Out[c][d]
				}
				tokens[c] = layerNorm1D(tokens[c], layer.ln2Scale, layer.ln2Bias)
			}
		}
		output := m.linearBatchEngine(ctx, tokens, m.projW, m.projB)
		for c := 0; c < channels; c++ {
			off := b*channels*outputLen + c*outputLen
			for o := 0; o < outputLen; o++ {
				outFlat[off+o] = float32(output[c][o])
			}
		}
	}
	return tensor.New[float32]([]int{batch, channels, outputLen}, outFlat)
}

// trainWindowedEngine implements GPU-accelerated ITransformer training using
// DataLoader for batching and backwardBatchEngine for the forward+backward
// pass. AdamW updates are applied to the model's float64 parameters directly.
func (m *ITransformer) trainWindowedEngine(windows [][][]float64, labels []float64, config TrainConfig) (*TrainResult, error) {
	nSamples := len(windows)
	channels := m.config.Channels
	outputLen := m.config.OutputLen

	ctx := context.Background()

	params := m.FlatParams()
	nParams := len(params)
	mState := make([]float64, nParams)
	vState := make([]float64, nParams)

	result := &TrainResult{LossHistory: make([]float64, config.Epochs)}

	batchSize := nSamples
	if config.BatchSize > 0 && config.BatchSize < nSamples {
		batchSize = config.BatchSize
	}

	dl := NewDataLoader(windows, labels, batchSize, true)

	for epoch := 0; epoch < config.Epochs; epoch++ {
		dl.Reset()
		epochLoss := 0.0
		nBatches := 0

		for {
			inputBatch, labelBatch, ok := dl.Next()
			if !ok {
				break
			}

			// Reshape label from [bs, channels*outputLen] to [bs, channels, outputLen].
			lShape := labelBatch.Shape()
			bs := lShape[0]
			targetData := labelBatch.Data()
			target, err := tensor.New[float32]([]int{bs, channels, outputLen}, targetData)
			if err != nil {
				return nil, fmt.Errorf("itransformer: reshape target: %w", err)
			}

			grads, batchLoss, err := m.backwardBatchEngine(ctx, inputBatch, target)
			if err != nil {
				return nil, fmt.Errorf("itransformer: batched backward: %w", err)
			}

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

			// AdamW update.
			lr := scheduler.WarmupLR(config.LR, epoch, config.WarmupEpochs)
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
