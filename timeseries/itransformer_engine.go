package timeseries

import (
	"context"
	"fmt"
	"math"

	"github.com/zerfoo/ztensor/tensor"
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
	cache.embedOut = deepCopy2D(tokens)

	// Step 2: Encoder layers.
	cache.layerCaches = make([]iTransformerLayerCache, len(m.layers))
	for li, layer := range m.layers {
		var lc iTransformerLayerCache
		lc.inputTokens = deepCopy2D(tokens)
		lc.preAttnTokens = deepCopy2D(tokens)

		// --- Multi-head self-attention ---
		// Q, K, V projections: [channels x dModel] @ [dModel x dModel].
		Q := m.linearBatchEngine(ctx, tokens, layer.qW, layer.qB)
		K := m.linearBatchEngine(ctx, tokens, layer.kW, layer.kB)
		V := m.linearBatchEngine(ctx, tokens, layer.vW, layer.vB)
		lc.Q = deepCopy2D(Q)
		lc.K = deepCopy2D(K)
		lc.V = deepCopy2D(V)

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

		// Output projection: [channels x dModel] @ [dModel x dModel].
		attnOut := m.linearBatchEngine(ctx, attnConcat, layer.oW, layer.oB)
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
		// fc1: [channels x dModel] @ [dModel x dFF].
		fc1Out := m.linearBatchEngine(ctx, ln1Out, layer.fc1W, layer.fc1B)
		geluOut := make([][]float64, channels)
		for c := 0; c < channels; c++ {
			geluOut[c] = make([]float64, len(fc1Out[c]))
			for i := range fc1Out[c] {
				geluOut[c][i] = gelu(fc1Out[c][i])
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
			ln2Out[c], lc.ln2Mu[c], lc.ln2Std[c] = layerNormCached(preLN2[c], layer.ln2Scale, layer.ln2Bias)
		}
		lc.fc1Out = deepCopy2D(fc1Out)
		lc.geluOut = deepCopy2D(geluOut)
		lc.fc2Out = deepCopy2D(fc2Out)
		lc.preLN2 = deepCopy2D(preLN2)
		lc.ln2Out = deepCopy2D(ln2Out)

		tokens = ln2Out
		cache.layerCaches[li] = lc
	}

	// Store pre-projection tokens.
	cache.preProj = deepCopy2D(tokens)

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
					scores[i] = softmax(scores[i])
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
				tokens[c] = layerNorm(tokens[c], layer.ln1Scale, layer.ln1Bias)
			}
			fc1Out := m.linearBatchEngine(ctx, tokens, layer.fc1W, layer.fc1B)
			for c := 0; c < channels; c++ {
				for i := range fc1Out[c] {
					fc1Out[c][i] = gelu(fc1Out[c][i])
				}
			}
			fc2Out := m.linearBatchEngine(ctx, fc1Out, layer.fc2W, layer.fc2B)
			for c := 0; c < channels; c++ {
				for d := 0; d < dModel; d++ {
					tokens[c][d] += fc2Out[c][d]
				}
				tokens[c] = layerNorm(tokens[c], layer.ln2Scale, layer.ln2Bias)
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
// float32 tensor operations via the compute.Engine. The forward/backward
// analytical backpropagation logic mirrors the CPU path but uses float32
// working copies. AdamW updates use engine tensor ops for large parameter
// tensors.
func (m *ITransformer) trainWindowedEngine(windows [][][]float64, labels []float64, config TrainConfig) (*TrainResult, error) {
	nSamples := len(windows)
	channels := m.config.Channels
	outputLen := m.config.OutputLen

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

	ctx := context.Background()

	// paramRef holds a float32 working copy and pointers back into the model's
	// float64 parameters so updates can be written back after each step.
	type paramRef struct {
		f64Ptrs []*float64
		f32     []float32
	}

	// Build flat parameter references matching flatParams order.
	var allParams []paramRef

	// Embedding weights [inputLen][dModel].
	{
		n := m.config.InputLen * m.config.DModel
		ptrs := make([]*float64, 0, n)
		f32 := make([]float32, 0, n)
		for i := range m.embedW {
			for j := range m.embedW[i] {
				ptrs = append(ptrs, &m.embedW[i][j])
				f32 = append(f32, float32(m.embedW[i][j]))
			}
		}
		allParams = append(allParams, paramRef{ptrs, f32})
	}
	// Embedding bias [dModel].
	{
		ptrs := make([]*float64, m.config.DModel)
		f32 := make([]float32, m.config.DModel)
		for i := range m.embedB {
			ptrs[i] = &m.embedB[i]
			f32[i] = float32(m.embedB[i])
		}
		allParams = append(allParams, paramRef{ptrs, f32})
	}

	// Encoder layers.
	for li := range m.layers {
		l := &m.layers[li]
		dModel := m.config.DModel
		dFF := m.config.DFF

		// Q, K, V, O weights: each [dModel][dModel].
		for _, wPtr := range []*[][]float64{&l.qW, &l.kW, &l.vW, &l.oW} {
			n := dModel * dModel
			ptrs := make([]*float64, 0, n)
			f32 := make([]float32, 0, n)
			for i := range *wPtr {
				for j := range (*wPtr)[i] {
					ptrs = append(ptrs, &(*wPtr)[i][j])
					f32 = append(f32, float32((*wPtr)[i][j]))
				}
			}
			allParams = append(allParams, paramRef{ptrs, f32})
		}
		// Q, K, V, O biases: each [dModel].
		for _, bPtr := range []*[]float64{&l.qB, &l.kB, &l.vB, &l.oB} {
			ptrs := make([]*float64, dModel)
			f32 := make([]float32, dModel)
			for i := range *bPtr {
				ptrs[i] = &(*bPtr)[i]
				f32[i] = float32((*bPtr)[i])
			}
			allParams = append(allParams, paramRef{ptrs, f32})
		}

		// LN1 scale and bias.
		for _, bPtr := range []*[]float64{&l.ln1Scale, &l.ln1Bias} {
			ptrs := make([]*float64, dModel)
			f32 := make([]float32, dModel)
			for i := range *bPtr {
				ptrs[i] = &(*bPtr)[i]
				f32[i] = float32((*bPtr)[i])
			}
			allParams = append(allParams, paramRef{ptrs, f32})
		}

		// FFN: fc1W [dModel][dFF].
		{
			n := dModel * dFF
			ptrs := make([]*float64, 0, n)
			f32 := make([]float32, 0, n)
			for i := range l.fc1W {
				for j := range l.fc1W[i] {
					ptrs = append(ptrs, &l.fc1W[i][j])
					f32 = append(f32, float32(l.fc1W[i][j]))
				}
			}
			allParams = append(allParams, paramRef{ptrs, f32})
		}
		// fc1B [dFF].
		{
			ptrs := make([]*float64, dFF)
			f32 := make([]float32, dFF)
			for i := range l.fc1B {
				ptrs[i] = &l.fc1B[i]
				f32[i] = float32(l.fc1B[i])
			}
			allParams = append(allParams, paramRef{ptrs, f32})
		}
		// fc2W [dFF][dModel].
		{
			n := dFF * dModel
			ptrs := make([]*float64, 0, n)
			f32 := make([]float32, 0, n)
			for i := range l.fc2W {
				for j := range l.fc2W[i] {
					ptrs = append(ptrs, &l.fc2W[i][j])
					f32 = append(f32, float32(l.fc2W[i][j]))
				}
			}
			allParams = append(allParams, paramRef{ptrs, f32})
		}
		// fc2B [dModel].
		{
			ptrs := make([]*float64, dModel)
			f32 := make([]float32, dModel)
			for i := range l.fc2B {
				ptrs[i] = &l.fc2B[i]
				f32[i] = float32(l.fc2B[i])
			}
			allParams = append(allParams, paramRef{ptrs, f32})
		}

		// LN2 scale and bias.
		for _, bPtr := range []*[]float64{&l.ln2Scale, &l.ln2Bias} {
			ptrs := make([]*float64, dModel)
			f32 := make([]float32, dModel)
			for i := range *bPtr {
				ptrs[i] = &(*bPtr)[i]
				f32[i] = float32((*bPtr)[i])
			}
			allParams = append(allParams, paramRef{ptrs, f32})
		}
	}

	// Output projection: projW [dModel][outputLen].
	{
		n := m.config.DModel * outputLen
		ptrs := make([]*float64, 0, n)
		f32 := make([]float32, 0, n)
		for i := range m.projW {
			for j := range m.projW[i] {
				ptrs = append(ptrs, &m.projW[i][j])
				f32 = append(f32, float32(m.projW[i][j]))
			}
		}
		allParams = append(allParams, paramRef{ptrs, f32})
	}
	// projB [outputLen].
	{
		ptrs := make([]*float64, outputLen)
		f32 := make([]float32, outputLen)
		for i := range m.projB {
			ptrs[i] = &m.projB[i]
			f32[i] = float32(m.projB[i])
		}
		allParams = append(allParams, paramRef{ptrs, f32})
	}

	// AdamW state per parameter group.
	type adamF32 struct {
		m []float32
		v []float32
	}
	adamStates := make([]adamF32, len(allParams))
	for i, p := range allParams {
		adamStates[i] = adamF32{m: make([]float32, len(p.f32)), v: make([]float32, len(p.f32))}
	}

	result := &TrainResult{LossHistory: make([]float64, config.Epochs)}

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
			batchLabels := labels[start*channels*outputLen : end*channels*outputLen]

			// Write f32 working copies back to model weights for forward/backward.
			for _, p := range allParams {
				for i, ptr := range p.f64Ptrs {
					*ptr = float64(p.f32[i])
				}
			}

			// Use engine-accelerated forward with CPU backward.
			accGrads := newITransformerGrads(m.config)
			batchLoss := 0.0
			scale := 1.0 / float64(bs*channels*outputLen)

			for s := 0; s < bs; s++ {
				pred, cache := m.forwardWithCacheEngine(ctx, batchWindows[s])

				dOutput := make([][]float64, channels)
				for c := 0; c < channels; c++ {
					dOutput[c] = make([]float64, outputLen)
					for o := 0; o < outputLen; o++ {
						labelIdx := s*channels*outputLen + c*outputLen + o
						diff := pred[c][o] - batchLabels[labelIdx]
						batchLoss += diff * diff
						dOutput[c][o] = 2.0 * diff * scale
					}
				}

				m.backward(dOutput, cache, &accGrads)
			}
			batchLoss *= scale

			gradsF64 := accGrads.collectGrads(m.config)
			epochLoss += batchLoss
			nBatches++

			// Gradient clipping.
			if config.GradClip > 0 {
				norm := 0.0
				for _, g := range gradsF64 {
					norm += g * g
				}
				norm = math.Sqrt(norm)
				if norm > config.GradClip {
					s := config.GradClip / norm
					for i := range gradsF64 {
						gradsF64[i] *= s
					}
				}
			}

			// Distribute flat gradients back to per-parameter-group slices.
			paramGrads := make([][]float32, len(allParams))
			offset := 0
			for pi, p := range allParams {
				n := len(p.f32)
				paramGrads[pi] = make([]float32, n)
				for i := 0; i < n; i++ {
					paramGrads[pi][i] = float32(gradsF64[offset+i])
				}
				offset += n
			}

			// AdamW update using engine for large parameter tensors.
			lr := float32(warmupLR(config.LR, epoch, config.WarmupEpochs))
			tStep := float32(epoch*((nSamples+batchSize-1)/batchSize) + nBatches)
			beta1 := float32(config.Beta1)
			beta2 := float32(config.Beta2)
			eps := float32(config.Epsilon)
			wd := float32(config.WeightDecay)

			for pi, p := range allParams {
				st := &adamStates[pi]
				grads := paramGrads[pi]

				if len(p.f32) >= 64 {
					m.adamStepEngine(ctx, p.f32, grads, st.m, st.v, lr, tStep, beta1, beta2, eps, wd)
				} else {
					for i := range p.f32 {
						st.m[i] = beta1*st.m[i] + (1-beta1)*grads[i]
						st.v[i] = beta2*st.v[i] + (1-beta2)*grads[i]*grads[i]
						mHat := st.m[i] / (1 - float32(math.Pow(float64(beta1), float64(tStep))))
						vHat := st.v[i] / (1 - float32(math.Pow(float64(beta2), float64(tStep))))
						p.f32[i] -= lr * (mHat/(float32(math.Sqrt(float64(vHat)))+eps) + wd*p.f32[i])
					}
				}
			}
		}

		// Write back final f32 values to model weights.
		for _, p := range allParams {
			for i, ptr := range p.f64Ptrs {
				*ptr = float64(p.f32[i])
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

// adamStepEngine performs a single AdamW update step using engine tensor
// operations for vectorized math (Sqrt, Div, Add, Sub, MulScalar).
func (m *ITransformer) adamStepEngine(ctx context.Context, params, grads, mState, vState []float32, lr, tStep, beta1, beta2, eps, wd float32) {
	n := len(params)

	// Update moments.
	for i := 0; i < n; i++ {
		mState[i] = beta1*mState[i] + (1-beta1)*grads[i]
		vState[i] = beta2*vState[i] + (1-beta2)*grads[i]*grads[i]
	}

	mHatCorr := 1 - float32(math.Pow(float64(beta1), float64(tStep)))
	vHatCorr := 1 - float32(math.Pow(float64(beta2), float64(tStep)))

	shape := []int{n}
	pT, err := tensor.New[float32](shape, params)
	if err != nil {
		m.adamStepScalar(params, mState, vState, lr, tStep, beta1, beta2, eps, wd)
		return
	}
	mT, err := tensor.New[float32](shape, mState)
	if err != nil {
		m.adamStepScalar(params, mState, vState, lr, tStep, beta1, beta2, eps, wd)
		return
	}
	vT, err := tensor.New[float32](shape, vState)
	if err != nil {
		m.adamStepScalar(params, mState, vState, lr, tStep, beta1, beta2, eps, wd)
		return
	}

	mHat, err := m.engine.DivScalar(ctx, mT, mHatCorr)
	if err != nil {
		m.adamStepScalar(params, mState, vState, lr, tStep, beta1, beta2, eps, wd)
		return
	}
	vHat, err := m.engine.DivScalar(ctx, vT, vHatCorr)
	if err != nil {
		m.adamStepScalar(params, mState, vState, lr, tStep, beta1, beta2, eps, wd)
		return
	}
	vSqrt, err := m.engine.Sqrt(ctx, vHat)
	if err != nil {
		m.adamStepScalar(params, mState, vState, lr, tStep, beta1, beta2, eps, wd)
		return
	}
	vSqrtEps, err := m.engine.AddScalar(ctx, vSqrt, eps)
	if err != nil {
		m.adamStepScalar(params, mState, vState, lr, tStep, beta1, beta2, eps, wd)
		return
	}
	adamUpdate, err := m.engine.Div(ctx, mHat, vSqrtEps)
	if err != nil {
		m.adamStepScalar(params, mState, vState, lr, tStep, beta1, beta2, eps, wd)
		return
	}
	wdTerm, err := m.engine.MulScalar(ctx, pT, wd)
	if err != nil {
		m.adamStepScalar(params, mState, vState, lr, tStep, beta1, beta2, eps, wd)
		return
	}
	total, err := m.engine.Add(ctx, adamUpdate, wdTerm)
	if err != nil {
		m.adamStepScalar(params, mState, vState, lr, tStep, beta1, beta2, eps, wd)
		return
	}
	scaled, err := m.engine.MulScalar(ctx, total, lr)
	if err != nil {
		m.adamStepScalar(params, mState, vState, lr, tStep, beta1, beta2, eps, wd)
		return
	}
	newP, err := m.engine.Sub(ctx, pT, scaled)
	if err != nil {
		m.adamStepScalar(params, mState, vState, lr, tStep, beta1, beta2, eps, wd)
		return
	}

	copy(params, newP.Data())
}

// adamStepScalar is the scalar fallback for AdamW when engine operations fail.
func (m *ITransformer) adamStepScalar(params, mState, vState []float32, lr, tStep, beta1, beta2, eps, wd float32) {
	for i := range params {
		mHat := mState[i] / (1 - float32(math.Pow(float64(beta1), float64(tStep))))
		vHat := vState[i] / (1 - float32(math.Pow(float64(beta2), float64(tStep))))
		params[i] -= lr * (mHat/(float32(math.Sqrt(float64(vHat)))+eps) + wd*params[i])
	}
}
