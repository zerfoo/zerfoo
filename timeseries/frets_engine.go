package timeseries

import (
	"context"
	"fmt"
	"math"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/tensor"

	"github.com/zerfoo/zerfoo/training/optimizer"
	"github.com/zerfoo/zerfoo/training/scheduler"
)

// trainWindowedEngine implements TrainWindowed using the compute engine for
// GPU-accelerated tensor operations. The DFT/IDFT remain on the CPU (no
// engine DFT exists), while the channel mixing MLP, temporal mixing MLP,
// output projection, and AdamW update use float32 engine operations.
func (f *FreTS) trainWindowedEngine(windows [][][]float64, labels []float64, config TrainConfig) (*TrainResult, error) {
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

	nSamples := len(windows)
	windows, f.normMeans, f.normStds = normalizeWindows(windows)

	ctx := context.Background()
	eng := f.engine
	channels := f.config.Channels
	inputLen := f.config.InputLen
	outputLen := f.config.OutputLen
	topK := f.config.TopK
	hidden := f.config.HiddenSize

	// Convert float64 weights to float32 tensors.
	// Channel MLP: chanW1 [channels, hidden], chanB1 [1, hidden],
	//              chanW2 [hidden, channels], chanB2 [1, channels].
	chanW1T, err := tensor.New[float32]([]int{channels, hidden}, float64ToFloat32(f.chanW1))
	if err != nil {
		return nil, fmt.Errorf("frets engine: chanW1 tensor: %w", err)
	}
	chanB1T, err := tensor.New[float32]([]int{1, hidden}, float64ToFloat32(f.chanB1))
	if err != nil {
		return nil, fmt.Errorf("frets engine: chanB1 tensor: %w", err)
	}
	chanW2T, err := tensor.New[float32]([]int{hidden, channels}, float64ToFloat32(f.chanW2))
	if err != nil {
		return nil, fmt.Errorf("frets engine: chanW2 tensor: %w", err)
	}
	chanB2T, err := tensor.New[float32]([]int{1, channels}, float64ToFloat32(f.chanB2))
	if err != nil {
		return nil, fmt.Errorf("frets engine: chanB2 tensor: %w", err)
	}

	// Temporal MLP: tempW1 [topK, hidden], tempB1 [1, hidden],
	//               tempW2 [hidden, topK], tempB2 [1, topK].
	tempW1T, err := tensor.New[float32]([]int{topK, hidden}, float64ToFloat32(f.tempW1))
	if err != nil {
		return nil, fmt.Errorf("frets engine: tempW1 tensor: %w", err)
	}
	tempB1T, err := tensor.New[float32]([]int{1, hidden}, float64ToFloat32(f.tempB1))
	if err != nil {
		return nil, fmt.Errorf("frets engine: tempB1 tensor: %w", err)
	}
	tempW2T, err := tensor.New[float32]([]int{hidden, topK}, float64ToFloat32(f.tempW2))
	if err != nil {
		return nil, fmt.Errorf("frets engine: tempW2 tensor: %w", err)
	}
	tempB2T, err := tensor.New[float32]([]int{1, topK}, float64ToFloat32(f.tempB2))
	if err != nil {
		return nil, fmt.Errorf("frets engine: tempB2 tensor: %w", err)
	}

	// Output projection per channel: outW [channels][outputLen, inputLen], outB [channels][1, outputLen].
	outWT := make([]*tensor.TensorNumeric[float32], channels)
	outBT := make([]*tensor.TensorNumeric[float32], channels)
	for c := 0; c < channels; c++ {
		wOff := c * outputLen * inputLen
		bOff := c * outputLen
		outWT[c], err = tensor.New[float32]([]int{outputLen, inputLen}, float64ToFloat32(f.outW[wOff:wOff+outputLen*inputLen]))
		if err != nil {
			return nil, fmt.Errorf("frets engine: outW[%d] tensor: %w", c, err)
		}
		outBT[c], err = tensor.New[float32]([]int{1, outputLen}, float64ToFloat32(f.outB[bOff:bOff+outputLen]))
		if err != nil {
			return nil, fmt.Errorf("frets engine: outB[%d] tensor: %w", c, err)
		}
	}

	// Collect all parameter tensors for AdamW.
	// Order: chanW1, chanB1, chanW2, chanB2, tempW1, tempB1, tempW2, tempB2, outW[0..channels-1], outB[0..channels-1]
	allParams := []*tensor.TensorNumeric[float32]{
		chanW1T, chanB1T, chanW2T, chanB2T,
		tempW1T, tempB1T, tempW2T, tempB2T,
	}
	for c := 0; c < channels; c++ {
		allParams = append(allParams, outWT[c])
	}
	for c := 0; c < channels; c++ {
		allParams = append(allParams, outBT[c])
	}

	// Wrap parameters as graph.Parameter for optimizer.AdamW.
	graphParams := make([]*graph.Parameter[float32], len(allParams))
	for i, p := range allParams {
		graphParams[i], _ = graph.NewParameter(fmt.Sprintf("param%d", i), p, tensor.New[float32])
	}
	opt := optimizer.NewAdamW(eng, float32(config.LR), float32(config.Beta1), float32(config.Beta2), float32(config.Epsilon), float32(config.WeightDecay))
	if config.GradClip > 0 {
		opt.SetMaxGradNorm(config.GradClip)
	}

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
			batchLabels := labels[start*channels*outputLen : end*channels*outputLen]

			// Zero gradients.
			allGrads := make([][]float32, len(allParams))
			for i, p := range allParams {
				allGrads[i] = make([]float32, len(p.Data()))
			}

			batchLoss := float32(0)

			for s := 0; s < bs; s++ {
				sample := windows[start+s]

				// Forward pass: DFT on CPU, MLP and output projection via engine.MatMul.
				pred, cache := f.forwardWithCacheEngine(ctx, eng, sample,
					chanW1T, chanB1T, chanW2T, chanB2T,
					tempW1T, tempB1T, tempW2T, tempB2T,
					outWT, outBT)

				// Compute dOut and loss.
				dOut := make([][]float64, channels)
				for c := 0; c < channels; c++ {
					dOut[c] = make([]float64, outputLen)
					for o := 0; o < outputLen; o++ {
						labelIdx := s*channels*outputLen + c*outputLen + o
						diff := pred[c][o] - batchLabels[labelIdx]
						if !isFinite(diff) {
							diff = 0
						}
						batchLoss += float32(diff * diff)
						dOut[c][o] = 2.0 * diff / float64(bs*channels*outputLen)
					}
				}

				// CPU backward for all parameters.
				sampleGrads := f.backward(dOut, cache)

				// Accumulate into allGrads matching the parameter order.
				// Parameter order in flatParams: chanW1, chanB1, chanW2, chanB2,
				// tempW1, tempB1, tempW2, tempB2, outW (all channels), outB (all channels).
				off := 0
				// chanW1
				for i := 0; i < channels*hidden; i++ {
					allGrads[0][i] += float32(sampleGrads[off+i])
				}
				off += channels * hidden
				// chanB1
				for i := 0; i < hidden; i++ {
					allGrads[1][i] += float32(sampleGrads[off+i])
				}
				off += hidden
				// chanW2
				for i := 0; i < hidden*channels; i++ {
					allGrads[2][i] += float32(sampleGrads[off+i])
				}
				off += hidden * channels
				// chanB2
				for i := 0; i < channels; i++ {
					allGrads[3][i] += float32(sampleGrads[off+i])
				}
				off += channels
				// tempW1
				for i := 0; i < topK*hidden; i++ {
					allGrads[4][i] += float32(sampleGrads[off+i])
				}
				off += topK * hidden
				// tempB1
				for i := 0; i < hidden; i++ {
					allGrads[5][i] += float32(sampleGrads[off+i])
				}
				off += hidden
				// tempW2
				for i := 0; i < hidden*topK; i++ {
					allGrads[6][i] += float32(sampleGrads[off+i])
				}
				off += hidden * topK
				// tempB2
				for i := 0; i < topK; i++ {
					allGrads[7][i] += float32(sampleGrads[off+i])
				}
				off += topK
				// outW per channel
				for c := 0; c < channels; c++ {
					gradIdx := 8 + c
					sz := outputLen * inputLen
					for i := 0; i < sz; i++ {
						allGrads[gradIdx][i] += float32(sampleGrads[off+i])
					}
					off += sz
				}
				// outB per channel
				for c := 0; c < channels; c++ {
					gradIdx := 8 + channels + c
					for i := 0; i < outputLen; i++ {
						allGrads[gradIdx][i] += float32(sampleGrads[off+i])
					}
					off += outputLen
				}
			}

			batchLossF64 := float64(batchLoss) / float64(bs*channels*outputLen)
			epochLoss += batchLossF64
			nBatches++

			// Gradient clipping.
			if config.GradClip > 0 {
				norm := float64(0)
				for _, g := range allGrads {
					for _, v := range g {
						norm += float64(v) * float64(v)
					}
				}
				norm = math.Sqrt(norm)
				if norm > config.GradClip {
					clipScale := float32(config.GradClip / norm)
					for _, g := range allGrads {
						for i := range g {
							g[i] *= clipScale
						}
					}
				}
			}

			// Set gradients and apply AdamW step.
			opt.SetLR(float32(scheduler.WarmupLR(config.LR, epoch, config.WarmupEpochs)))
			for pi := range allParams {
				gradT, _ := tensor.New[float32](graphParams[pi].Value.Shape(), allGrads[pi])
				graphParams[pi].Gradient = gradT
			}
			if err := opt.Step(ctx, graphParams); err != nil {
				return nil, fmt.Errorf("frets: adamw step: %w", err)
			}

			// Sync float32 tensors back to float64 model weights so the
			// next forward pass (which uses f.forwardWithCache on float64
			// weights) sees the updated parameters.
			copyF32ToF64(chanW1T.Data(), f.chanW1)
			copyF32ToF64(chanB1T.Data(), f.chanB1)
			copyF32ToF64(chanW2T.Data(), f.chanW2)
			copyF32ToF64(chanB2T.Data(), f.chanB2)
			copyF32ToF64(tempW1T.Data(), f.tempW1)
			copyF32ToF64(tempB1T.Data(), f.tempB1)
			copyF32ToF64(tempW2T.Data(), f.tempW2)
			copyF32ToF64(tempB2T.Data(), f.tempB2)
			for c := 0; c < channels; c++ {
				wOff := c * outputLen * inputLen
				bOff := c * outputLen
				copyF32ToF64(outWT[c].Data(), f.outW[wOff:wOff+outputLen*inputLen])
				copyF32ToF64(outBT[c].Data(), f.outB[bOff:bOff+outputLen])
			}
		}

		result.LossHistory[epoch] = epochLoss / float64(nBatches)
		result.FinalLoss = result.LossHistory[epoch]

		if !isFinite(result.FinalLoss) {
			return nil, fmt.Errorf("frets: training diverged at epoch %d: loss=%v", epoch, result.FinalLoss)
		}
	}

	result.Metrics = map[string]float64{"mse": result.FinalLoss}
	return result, nil
}

// fretsEngineMatMul computes vec @ mat using engine.MatMul.
// vec has length rows, mat is [rows, cols] stored row-major.
// Returns a slice of length cols. Falls back to scalar multiply on error.
func fretsEngineMatMul(ctx context.Context, eng compute.Engine[float32], vec []float32, mat []float32, rows, cols int) []float32 {
	vT, err := tensor.New[float32]([]int{1, rows}, vec)
	if err != nil {
		return fretsScalarMatMul(vec, mat, rows, cols)
	}
	mT, err := tensor.New[float32]([]int{rows, cols}, mat)
	if err != nil {
		return fretsScalarMatMul(vec, mat, rows, cols)
	}
	out, err := eng.MatMul(ctx, vT, mT)
	if err != nil {
		return fretsScalarMatMul(vec, mat, rows, cols)
	}
	return out.Data()
}

// fretsScalarMatMul computes vec @ mat on the CPU as a fallback.
func fretsScalarMatMul(vec []float32, mat []float32, rows, cols int) []float32 {
	out := make([]float32, cols)
	for j := 0; j < cols; j++ {
		var s float32
		for i := 0; i < rows; i++ {
			s += vec[i] * mat[i*cols+j]
		}
		out[j] = s
	}
	return out
}

// forwardBatch runs the FreTS forward pass on a batch of samples.
// Input: [batch][channels][inputLen], returns: [batch][channels][outputLen].
// Each sample undergoes DFT -> channel mixing -> temporal mixing -> IDFT ->
// output projection using the same weights via Engine MatMul operations.
func (f *FreTS) forwardBatch(inputs [][][]float64) [][][]float64 {
	batch := len(inputs)
	out := make([][][]float64, batch)
	for b := 0; b < batch; b++ {
		out[b] = f.forward(inputs[b])
	}
	return out
}

// forwardWithCacheEngine runs the FreTS forward pass using engine.MatMul for
// all MLP and output projection matrix multiplications. DFT/IDFT remain on CPU.
// It populates the same fretsCache as forwardWithCache so the CPU backward pass
// can be reused without modification.
func (f *FreTS) forwardWithCacheEngine(ctx context.Context, eng compute.Engine[float32], input [][]float64,
	chanW1T, chanB1T, chanW2T, chanB2T *tensor.TensorNumeric[float32],
	tempW1T, tempB1T, tempW2T, tempB2T *tensor.TensorNumeric[float32],
	outWT, outBT []*tensor.TensorNumeric[float32],
) ([][]float64, *fretsCache) {
	channels := f.config.Channels
	inputLen := f.config.InputLen
	topK := f.config.TopK
	hidden := f.config.HiddenSize
	cache := &fretsCache{}

	// Step 1: DFT per channel, select top-K frequencies (CPU, complex arithmetic).
	cache.allCoeffs = make([][]complex128, channels)
	cache.topIndices = make([][]int, channels)
	cache.freqRealPreChan = make([][]float64, channels)
	cache.freqImagPreChan = make([][]float64, channels)

	freqReal := make([][]float64, channels)
	freqImag := make([][]float64, channels)

	for c := 0; c < channels; c++ {
		cache.allCoeffs[c] = dft(input[c])
		cache.topIndices[c] = topKIndices(cache.allCoeffs[c], topK)
		freqReal[c] = make([]float64, topK)
		freqImag[c] = make([]float64, topK)
		cache.freqRealPreChan[c] = make([]float64, topK)
		cache.freqImagPreChan[c] = make([]float64, topK)
		for i, idx := range cache.topIndices[c] {
			freqReal[c][i] = real(cache.allCoeffs[c][idx])
			freqImag[c][i] = imag(cache.allCoeffs[c][idx])
			cache.freqRealPreChan[c][i] = freqReal[c][i]
			cache.freqImagPreChan[c][i] = freqImag[c][i]
		}
	}

	// Step 2: Channel mixing via engine.MatMul.
	cache.chanHiddenReal = make([][]float64, topK)
	cache.chanHiddenImag = make([][]float64, topK)
	cache.chanPreActReal = make([][]float64, topK)
	cache.chanPreActImag = make([][]float64, topK)
	cache.chanInputReal = make([][]float64, topK)
	cache.chanInputImag = make([][]float64, topK)

	chanW1Data := chanW1T.Data()
	chanB1Data := chanB1T.Data()
	chanW2Data := chanW2T.Data()
	chanB2Data := chanB2T.Data()

	for k := 0; k < topK; k++ {
		realIn32 := make([]float32, channels)
		imagIn32 := make([]float32, channels)
		for c := 0; c < channels; c++ {
			realIn32[c] = float32(freqReal[c][k])
			imagIn32[c] = float32(freqImag[c][k])
		}
		cache.chanInputReal[k] = make([]float64, channels)
		cache.chanInputImag[k] = make([]float64, channels)
		for c := 0; c < channels; c++ {
			cache.chanInputReal[k][c] = freqReal[c][k]
			cache.chanInputImag[k][c] = freqImag[c][k]
		}

		// Layer 1: realIn[1, channels] @ chanW1[channels, hidden] + chanB1
		hReal32 := fretsEngineMatMul(ctx, eng, realIn32, chanW1Data, channels, hidden)
		hImag32 := fretsEngineMatMul(ctx, eng, imagIn32, chanW1Data, channels, hidden)

		preActReal := make([]float64, hidden)
		preActImag := make([]float64, hidden)
		hReal := make([]float64, hidden)
		hImag := make([]float64, hidden)
		hRealAct32 := make([]float32, hidden)
		hImagAct32 := make([]float32, hidden)
		for j := 0; j < hidden; j++ {
			val := float64(hReal32[j] + chanB1Data[j])
			preActReal[j] = val
			if val > 0 {
				hReal[j] = val
				hRealAct32[j] = float32(val)
			}
			val = float64(hImag32[j] + chanB1Data[j])
			preActImag[j] = val
			if val > 0 {
				hImag[j] = val
				hImagAct32[j] = float32(val)
			}
		}
		cache.chanHiddenReal[k] = hReal
		cache.chanHiddenImag[k] = hImag
		cache.chanPreActReal[k] = preActReal
		cache.chanPreActImag[k] = preActImag

		// Layer 2: hReal[1, hidden] @ chanW2[hidden, channels] + chanB2
		realOut32 := fretsEngineMatMul(ctx, eng, hRealAct32, chanW2Data, hidden, channels)
		imagOut32 := fretsEngineMatMul(ctx, eng, hImagAct32, chanW2Data, hidden, channels)

		// Residual connection.
		for c := 0; c < channels; c++ {
			freqReal[c][k] += float64(realOut32[c] + chanB2Data[c])
			freqImag[c][k] += float64(imagOut32[c] + chanB2Data[c])
		}
	}

	// Save pre-temporal values.
	cache.freqRealPreTemp = make([][]float64, channels)
	cache.freqImagPreTemp = make([][]float64, channels)
	for c := 0; c < channels; c++ {
		cache.freqRealPreTemp[c] = make([]float64, topK)
		cache.freqImagPreTemp[c] = make([]float64, topK)
		copy(cache.freqRealPreTemp[c], freqReal[c])
		copy(cache.freqImagPreTemp[c], freqImag[c])
	}

	// Step 3: Temporal mixing via engine.MatMul.
	cache.tempHiddenReal = make([][]float64, channels)
	cache.tempHiddenImag = make([][]float64, channels)
	cache.tempPreActReal = make([][]float64, channels)
	cache.tempPreActImag = make([][]float64, channels)
	cache.tempInputReal = make([][]float64, channels)
	cache.tempInputImag = make([][]float64, channels)

	tempW1Data := tempW1T.Data()
	tempB1Data := tempB1T.Data()
	tempW2Data := tempW2T.Data()
	tempB2Data := tempB2T.Data()

	for c := 0; c < channels; c++ {
		cache.tempInputReal[c] = make([]float64, topK)
		cache.tempInputImag[c] = make([]float64, topK)
		copy(cache.tempInputReal[c], freqReal[c])
		copy(cache.tempInputImag[c], freqImag[c])

		realIn32 := make([]float32, topK)
		imagIn32 := make([]float32, topK)
		for i := 0; i < topK; i++ {
			realIn32[i] = float32(freqReal[c][i])
			imagIn32[i] = float32(freqImag[c][i])
		}

		// Layer 1: freqReal[1, topK] @ tempW1[topK, hidden] + tempB1
		hReal32 := fretsEngineMatMul(ctx, eng, realIn32, tempW1Data, topK, hidden)
		hImag32 := fretsEngineMatMul(ctx, eng, imagIn32, tempW1Data, topK, hidden)

		preActReal := make([]float64, hidden)
		preActImag := make([]float64, hidden)
		hReal := make([]float64, hidden)
		hImag := make([]float64, hidden)
		hRealAct32 := make([]float32, hidden)
		hImagAct32 := make([]float32, hidden)
		for j := 0; j < hidden; j++ {
			val := float64(hReal32[j] + tempB1Data[j])
			preActReal[j] = val
			if val > 0 {
				hReal[j] = val
				hRealAct32[j] = float32(val)
			}
			val = float64(hImag32[j] + tempB1Data[j])
			preActImag[j] = val
			if val > 0 {
				hImag[j] = val
				hImagAct32[j] = float32(val)
			}
		}
		cache.tempHiddenReal[c] = hReal
		cache.tempHiddenImag[c] = hImag
		cache.tempPreActReal[c] = preActReal
		cache.tempPreActImag[c] = preActImag

		// Layer 2: hReal[1, hidden] @ tempW2[hidden, topK] + tempB2
		realOut32 := fretsEngineMatMul(ctx, eng, hRealAct32, tempW2Data, hidden, topK)
		imagOut32 := fretsEngineMatMul(ctx, eng, hImagAct32, tempW2Data, hidden, topK)

		// Residual connection.
		for k := 0; k < topK; k++ {
			freqReal[c][k] += float64(realOut32[k] + tempB2Data[k])
			freqImag[c][k] += float64(imagOut32[k] + tempB2Data[k])
		}
	}

	// Step 4: Reconstruct time domain signal via inverse DFT (CPU).
	cache.reconstructed = make([][]float64, channels)
	for c := 0; c < channels; c++ {
		mixed := make([]complex128, len(cache.allCoeffs[c]))
		for i, idx := range cache.topIndices[c] {
			mixed[idx] = complex(freqReal[c][i], freqImag[c][i])
		}
		cache.reconstructed[c] = idft(mixed, inputLen)
	}

	// Step 5: Output projection via engine.MatMul.
	output := make([][]float64, channels)
	for c := 0; c < channels; c++ {
		recon32 := make([]float32, inputLen)
		for i := 0; i < inputLen; i++ {
			recon32[i] = float32(cache.reconstructed[c][i])
		}
		// outW[outputLen, inputLen] @ recon[inputLen, 1] -> [outputLen, 1]
		reconT, err := tensor.New[float32]([]int{inputLen, 1}, recon32)
		if err != nil {
			// Fallback to CPU.
			output[c] = f.outputProjectCPU(c, cache.reconstructed[c])
			continue
		}
		projT, err := eng.MatMul(ctx, outWT[c], reconT)
		if err != nil {
			output[c] = f.outputProjectCPU(c, cache.reconstructed[c])
			continue
		}
		projData := projT.Data()
		outBData := outBT[c].Data()
		output[c] = make([]float64, f.config.OutputLen)
		for o := 0; o < f.config.OutputLen; o++ {
			output[c][o] = float64(projData[o] + outBData[o])
		}
	}

	return output, cache
}

// outputProjectCPU is the scalar fallback for output projection when engine.MatMul fails.
func (f *FreTS) outputProjectCPU(ch int, reconstructed []float64) []float64 {
	inputLen := f.config.InputLen
	outputLen := f.config.OutputLen
	wOff := ch * outputLen * inputLen
	bOff := ch * outputLen
	out := make([]float64, outputLen)
	for o := 0; o < outputLen; o++ {
		val := f.outB[bOff+o]
		for i := 0; i < inputLen; i++ {
			val += f.outW[wOff+o*inputLen+i] * reconstructed[i]
		}
		out[o] = val
	}
	return out
}
