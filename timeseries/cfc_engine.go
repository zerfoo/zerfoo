package timeseries

import (
	"context"
	"fmt"
	"math"

	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/tensor"

	"github.com/zerfoo/zerfoo/training/optimizer"
	"github.com/zerfoo/zerfoo/training/scheduler"
)

// trainWindowedEngine implements GPU-accelerated CfC training using float32
// tensor operations via the compute.Engine. The forward/backward BPTT logic
// mirrors the CPU path but uses float32 working copies and engine-based
// AdamW for large parameter tensors.
func (c *CfC) trainWindowedEngine(windows [][][]float64, labels []float64, config TrainConfig) (*TrainResult, error) {
	nSamples := len(windows)
	outDim := c.config.OutputSize * c.config.OutputLen
	hiddenSize := c.config.HiddenSize
	numLayers := c.config.NumLayers

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

	windows, c.normMeans, c.normStds = normalizeWindows(windows)
	ctx := context.Background()

	// paramRef holds a float32 working copy and pointers back into the model's
	// float64 parameters so updates can be written back after each step.
	type paramRef struct {
		f64Ptrs []*float64
		f32     []float32
	}

	// Build flat parameter references: 5 groups per layer + 2 for output proj.
	var allParams []paramRef
	for l := 0; l < numLayers; l++ {
		layer := &c.layers[l]
		inSize := len(layer.Wx)

		// Wh [hiddenSize][hiddenSize]
		whPtrs := make([]*float64, 0, hiddenSize*hiddenSize)
		whF32 := make([]float32, hiddenSize*hiddenSize)
		for i := range layer.Wh {
			for j := range layer.Wh[i] {
				whPtrs = append(whPtrs, &layer.Wh[i][j])
				whF32[i*hiddenSize+j] = float32(layer.Wh[i][j])
			}
		}
		allParams = append(allParams, paramRef{whPtrs, whF32})

		// Wx [inSize][hiddenSize]
		wxPtrs := make([]*float64, 0, inSize*hiddenSize)
		wxF32 := make([]float32, inSize*hiddenSize)
		for i := range layer.Wx {
			for j := range layer.Wx[i] {
				wxPtrs = append(wxPtrs, &layer.Wx[i][j])
				wxF32[i*hiddenSize+j] = float32(layer.Wx[i][j])
			}
		}
		allParams = append(allParams, paramRef{wxPtrs, wxF32})

		// Bh [hiddenSize]
		bhPtrs := make([]*float64, hiddenSize)
		bhF32 := make([]float32, hiddenSize)
		for j := range layer.Bh {
			bhPtrs[j] = &layer.Bh[j]
			bhF32[j] = float32(layer.Bh[j])
		}
		allParams = append(allParams, paramRef{bhPtrs, bhF32})

		// Wtau [(inSize+hiddenSize)][hiddenSize]
		tauDim := inSize + hiddenSize
		wtPtrs := make([]*float64, 0, tauDim*hiddenSize)
		wtF32 := make([]float32, tauDim*hiddenSize)
		for i := range layer.Wtau {
			for j := range layer.Wtau[i] {
				wtPtrs = append(wtPtrs, &layer.Wtau[i][j])
				wtF32[i*hiddenSize+j] = float32(layer.Wtau[i][j])
			}
		}
		allParams = append(allParams, paramRef{wtPtrs, wtF32})

		// Btau [hiddenSize]
		btPtrs := make([]*float64, hiddenSize)
		btF32 := make([]float32, hiddenSize)
		for j := range layer.Btau {
			btPtrs[j] = &layer.Btau[j]
			btF32[j] = float32(layer.Btau[j])
		}
		allParams = append(allParams, paramRef{btPtrs, btF32})
	}

	// Output projection: outW [hiddenSize][outDim], outB [outDim].
	outWPtrs := make([]*float64, 0, hiddenSize*outDim)
	outWF32 := make([]float32, hiddenSize*outDim)
	for i := range c.outW {
		for j := range c.outW[i] {
			outWPtrs = append(outWPtrs, &c.outW[i][j])
			outWF32[i*outDim+j] = float32(c.outW[i][j])
		}
	}
	allParams = append(allParams, paramRef{outWPtrs, outWF32})

	outBPtrs := make([]*float64, outDim)
	outBF32 := make([]float32, outDim)
	for j := range c.outB {
		outBPtrs[j] = &c.outB[j]
		outBF32[j] = float32(c.outB[j])
	}
	allParams = append(allParams, paramRef{outBPtrs, outBF32})

	// Wrap parameters as graph.Parameter for optimizer.AdamW.
	graphParams := make([]*graph.Parameter[float32], len(allParams))
	for i, p := range allParams {
		t, _ := tensor.New[float32]([]int{len(p.f32)}, p.f32)
		graphParams[i], _ = graph.NewParameter(fmt.Sprintf("param%d", i), t, tensor.New[float32])
	}
	opt := optimizer.NewAdamW(c.engine, float32(config.LR), float32(config.Beta1), float32(config.Beta2), float32(config.Epsilon), float32(config.WeightDecay))
	if config.GradClip > 0 {
		opt.SetMaxGradNorm(config.GradClip)
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

			allGrads := make([][]float32, len(allParams))
			for i, p := range allParams {
				allGrads[i] = make([]float32, len(p.f32))
			}

			batchLoss := float32(0)

			for s := 0; s < bs; s++ {
				seqInput := transposeWindow(windows[start+s])
				seqLen := len(seqInput)
				sampleLabels := labels[(start+s)*outDim : (start+s+1)*outDim]

				// Forward pass storing intermediates for BPTT.
				type stepState struct {
					x     []float32
					hPrev []float32
					tau   []float32
					pre   []float32
					h     []float32
				}
				states := make([][]stepState, seqLen)

				hPrev := make([][]float32, numLayers)
				for l := 0; l < numLayers; l++ {
					hPrev[l] = make([]float32, hiddenSize)
				}

				for t := 0; t < seqLen; t++ {
					states[t] = make([]stepState, numLayers)
					xF := make([]float32, len(seqInput[t]))
					for i, v := range seqInput[t] {
						xF[i] = float32(v)
					}

					for l := 0; l < numLayers; l++ {
						paramBase := l * 5
						whData := allParams[paramBase].f32
						wxData := allParams[paramBase+1].f32
						bhData := allParams[paramBase+2].f32
						wtauData := allParams[paramBase+3].f32
						btauData := allParams[paramBase+4].f32

						inSize := len(c.layers[l].Wx)
						ss := stepState{
							x:     make([]float32, len(xF)),
							hPrev: make([]float32, hiddenSize),
							tau:   make([]float32, hiddenSize),
							pre:   make([]float32, hiddenSize),
							h:     make([]float32, hiddenSize),
						}
						copy(ss.x, xF)
						copy(ss.hPrev, hPrev[l])

						// tau = sigmoid(Wtau * [x, h] + btau)
						// Split Wtau into x-part [inSize, hiddenSize] and h-part [hiddenSize, hiddenSize].
						tauX := c.cfcEngineMatMul(ctx, xF, wtauData[:inSize*hiddenSize], inSize, hiddenSize)
						tauH := c.cfcEngineMatMul(ctx, hPrev[l], wtauData[inSize*hiddenSize:], hiddenSize, hiddenSize)
						for j := 0; j < hiddenSize; j++ {
							val := btauData[j] + tauX[j] + tauH[j]
							ss.tau[j] = 1.0 / (1.0 + float32(math.Exp(float64(-val))))
						}

						// preact = Wx*x + Wh*h + bh (before tanh)
						preX := c.cfcEngineMatMul(ctx, xF, wxData, inSize, hiddenSize)
						preH := c.cfcEngineMatMul(ctx, hPrev[l], whData, hiddenSize, hiddenSize)
						for j := 0; j < hiddenSize; j++ {
							ss.pre[j] = bhData[j] + preX[j] + preH[j]
						}

						// h_new = f * h_old + (1-f) * tanh(pre)
						for j := 0; j < hiddenSize; j++ {
							tauC := ss.tau[j]
							if tauC < 1e-6 {
								tauC = 1e-6
							}
							f := float32(math.Exp(float64(-1.0 / tauC)))
							tanhPre := float32(math.Tanh(float64(ss.pre[j])))
							ss.h[j] = f*hPrev[l][j] + (1-f)*tanhPre
						}

						hPrev[l] = ss.h
						xF = ss.h
						states[t][l] = ss
					}
				}

				// Output projection.
				finalH := hPrev[numLayers-1]
				outWData := allParams[numLayers*5].f32
				outBData := allParams[numLayers*5+1].f32

				projOut := c.cfcEngineMatMul(ctx, finalH, outWData, hiddenSize, outDim)
				pred := make([]float32, outDim)
				for j := 0; j < outDim; j++ {
					pred[j] = outBData[j] + projOut[j]
				}

				// MSE loss and output gradient for this sample.
				dOut := make([]float32, outDim)
				for j := 0; j < outDim; j++ {
					diff := pred[j] - float32(sampleLabels[j])
					batchLoss += diff * diff
					dOut[j] = 2.0 * diff / float32(bs*outDim)
				}

				// Backward: output projection gradients.
				outWGrads := allGrads[numLayers*5]
				outBGrads := allGrads[numLayers*5+1]
				dh := make([]float32, hiddenSize)
				for j := 0; j < outDim; j++ {
					outBGrads[j] += dOut[j]
					for i := 0; i < hiddenSize; i++ {
						outWGrads[i*outDim+j] += dOut[j] * finalH[i]
						dh[i] += dOut[j] * outWData[i*outDim+j]
					}
				}

				// BPTT through time and layers.
				for t := seqLen - 1; t >= 0; t-- {
					for l := numLayers - 1; l >= 0; l-- {
						ss := states[t][l]
						paramBase := l * 5
						whData := allParams[paramBase].f32
						wxData := allParams[paramBase+1].f32
						wtauData := allParams[paramBase+3].f32

						whGrads := allGrads[paramBase]
						wxGrads := allGrads[paramBase+1]
						bhGrads := allGrads[paramBase+2]
						wtauGrads := allGrads[paramBase+3]
						btauGrads := allGrads[paramBase+4]

						inSize := len(ss.x)

						tanhPreact := make([]float32, hiddenSize)
						fVals := make([]float32, hiddenSize)
						for j := 0; j < hiddenSize; j++ {
							tauC := ss.tau[j]
							if tauC < 1e-6 {
								tauC = 1e-6
							}
							fVals[j] = float32(math.Exp(float64(-1.0 / tauC)))
							tanhPreact[j] = float32(math.Tanh(float64(ss.pre[j])))
						}

						dPreact := make([]float32, hiddenSize)
						for j := 0; j < hiddenSize; j++ {
							dPreact[j] = dh[j] * (1 - fVals[j]) * (1 - tanhPreact[j]*tanhPreact[j])
						}

						dZtau := make([]float32, hiddenSize)
						for j := 0; j < hiddenSize; j++ {
							tauC := ss.tau[j]
							if tauC < 1e-6 {
								tauC = 1e-6
							}
							dfDtau := fVals[j] / (tauC * tauC)
							dhDf := ss.hPrev[j] - tanhPreact[j]
							dtauDz := ss.tau[j] * (1 - ss.tau[j])
							dZtau[j] = dh[j] * dhDf * dfDtau * dtauDz
						}

						// Wx gradients.
						for i := 0; i < inSize; i++ {
							for j := 0; j < hiddenSize; j++ {
								wxGrads[i*hiddenSize+j] += dPreact[j] * ss.x[i]
							}
						}
						// Wh gradients.
						for i := 0; i < hiddenSize; i++ {
							for j := 0; j < hiddenSize; j++ {
								whGrads[i*hiddenSize+j] += dPreact[j] * ss.hPrev[i]
							}
						}
						// Bh gradients.
						for j := 0; j < hiddenSize; j++ {
							bhGrads[j] += dPreact[j]
						}
						// Wtau gradients.
						for i := 0; i < inSize; i++ {
							for j := 0; j < hiddenSize; j++ {
								wtauGrads[i*hiddenSize+j] += dZtau[j] * ss.x[i]
							}
						}
						for i := 0; i < hiddenSize; i++ {
							for j := 0; j < hiddenSize; j++ {
								wtauGrads[(inSize+i)*hiddenSize+j] += dZtau[j] * ss.hPrev[i]
							}
						}
						// Btau gradients.
						for j := 0; j < hiddenSize; j++ {
							btauGrads[j] += dZtau[j]
						}

						// Propagate dh backward.
						dhPrev := make([]float32, hiddenSize)
						for i := 0; i < hiddenSize; i++ {
							for j := 0; j < hiddenSize; j++ {
								dhPrev[i] += dPreact[j] * whData[i*hiddenSize+j]
								dhPrev[i] += dZtau[j] * wtauData[(inSize+i)*hiddenSize+j]
							}
							dhPrev[i] += dh[i] * fVals[i]
						}

						if l > 0 {
							dx := make([]float32, inSize)
							for i := 0; i < inSize; i++ {
								for j := 0; j < hiddenSize; j++ {
									dx[i] += dPreact[j] * wxData[i*hiddenSize+j]
									dx[i] += dZtau[j] * wtauData[i*hiddenSize+j]
								}
							}
							dh = dx
						} else {
							dh = dhPrev
						}

						if l == 0 {
							dh = dhPrev
						}
					}
				}
			}

			batchLoss /= float32(bs * outDim)
			epochLoss += float64(batchLoss)
			nBatches++

			// Gradient clipping.
			if config.GradClip > 0 {
				norm := float32(0)
				for _, g := range allGrads {
					for _, gv := range g {
						norm += gv * gv
					}
				}
				norm = float32(math.Sqrt(float64(norm)))
				if norm > float32(config.GradClip) {
					scale := float32(config.GradClip) / norm
					for _, g := range allGrads {
						for i := range g {
							g[i] *= scale
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
				return nil, fmt.Errorf("cfc: adamw step: %w", err)
			}

			// Write back to float32 working copies and float64 model weights.
			for pi, p := range allParams {
				copy(p.f32, graphParams[pi].Value.Data())
				for i, ptr := range p.f64Ptrs {
					*ptr = float64(p.f32[i])
				}
			}
		}

		result.LossHistory[epoch] = epochLoss / float64(nBatches)
		result.FinalLoss = result.LossHistory[epoch]

		if !isFinite(result.FinalLoss) {
			return nil, fmt.Errorf("cfc: training diverged at epoch %d: loss=%v", epoch, result.FinalLoss)
		}
	}

	result.Metrics = map[string]float64{"mse": result.FinalLoss}
	return result, nil
}

// cfcEngineMatMul computes vec @ mat using engine.MatMul.
// vec has length rows, mat is [rows][cols] stored row-major with length rows*cols.
// Returns a slice of length cols. Falls back to scalar multiply on error.
func (c *CfC) cfcEngineMatMul(ctx context.Context, vec []float32, mat []float32, rows, cols int) []float32 {
	vT, err := tensor.New[float32]([]int{1, rows}, vec)
	if err != nil {
		return cfcScalarMatMul(vec, mat, rows, cols)
	}
	mT, err := tensor.New[float32]([]int{rows, cols}, mat)
	if err != nil {
		return cfcScalarMatMul(vec, mat, rows, cols)
	}
	out, err := c.engine.MatMul(ctx, vT, mT)
	if err != nil {
		return cfcScalarMatMul(vec, mat, rows, cols)
	}
	return out.Data()
}

// cfcScalarMatMul computes vec @ mat on the CPU as a fallback.
func cfcScalarMatMul(vec []float32, mat []float32, rows, cols int) []float32 {
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

