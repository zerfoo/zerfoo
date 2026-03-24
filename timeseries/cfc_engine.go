package timeseries

import (
	"context"
	"fmt"
	"math"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
)

// SetEngine configures an optional compute engine for GPU-accelerated training.
// When set, TrainWindowed dispatches to the engine-based path. Pass nil to
// revert to the pure-Go CPU path.
func (c *CfC) SetEngine(engine compute.Engine[float32], ops numeric.Arithmetic[float32]) {
	c.engine = engine
	c.ops = ops
}

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

	windows, _, _ = normalizeWindows(windows)
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
						for j := 0; j < hiddenSize; j++ {
							val := btauData[j]
							for i := 0; i < inSize; i++ {
								val += xF[i] * wtauData[i*hiddenSize+j]
							}
							for i := 0; i < hiddenSize; i++ {
								val += hPrev[l][i] * wtauData[(inSize+i)*hiddenSize+j]
							}
							ss.tau[j] = 1.0 / (1.0 + float32(math.Exp(float64(-val))))
						}

						// preact = Wx*x + Wh*h + bh (before tanh)
						for j := 0; j < hiddenSize; j++ {
							val := bhData[j]
							for i := 0; i < inSize; i++ {
								val += xF[i] * wxData[i*hiddenSize+j]
							}
							for i := 0; i < hiddenSize; i++ {
								val += hPrev[l][i] * whData[i*hiddenSize+j]
							}
							ss.pre[j] = val
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

				pred := make([]float32, outDim)
				for j := 0; j < outDim; j++ {
					pred[j] = outBData[j]
					for i := 0; i < hiddenSize; i++ {
						pred[j] += finalH[i] * outWData[i*outDim+j]
					}
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

			// AdamW update using engine for large parameter tensors.
			lr := float32(warmupLR(config.LR, epoch, config.WarmupEpochs))
			tStep := float32(epoch*((nSamples+batchSize-1)/batchSize) + nBatches)
			beta1 := float32(config.Beta1)
			beta2 := float32(config.Beta2)
			eps := float32(config.Epsilon)
			wd := float32(config.WeightDecay)

			for pi, p := range allParams {
				st := &adamStates[pi]
				grads := allGrads[pi]

				if len(p.f32) >= 64 {
					c.adamStepEngine(ctx, p.f32, grads, st.m, st.v, lr, tStep, beta1, beta2, eps, wd)
				} else {
					for i := range p.f32 {
						st.m[i] = beta1*st.m[i] + (1-beta1)*grads[i]
						st.v[i] = beta2*st.v[i] + (1-beta2)*grads[i]*grads[i]
						mHat := st.m[i] / (1 - float32(math.Pow(float64(beta1), float64(tStep))))
						vHat := st.v[i] / (1 - float32(math.Pow(float64(beta2), float64(tStep))))
						p.f32[i] -= lr * (mHat/(float32(math.Sqrt(float64(vHat)))+eps) + wd*p.f32[i])
					}
				}

				// Write back to float64 model weights.
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

// adamStepEngine performs a single AdamW update step using engine tensor
// operations for vectorized math (Sqrt, Div, Add, Sub, MulScalar).
func (c *CfC) adamStepEngine(ctx context.Context, params, grads, mState, vState []float32, lr, tStep, beta1, beta2, eps, wd float32) {
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
		c.adamStepScalar(params, mState, vState, lr, tStep, beta1, beta2, eps, wd)
		return
	}
	mT, err := tensor.New[float32](shape, mState)
	if err != nil {
		c.adamStepScalar(params, mState, vState, lr, tStep, beta1, beta2, eps, wd)
		return
	}
	vT, err := tensor.New[float32](shape, vState)
	if err != nil {
		c.adamStepScalar(params, mState, vState, lr, tStep, beta1, beta2, eps, wd)
		return
	}

	mHat, err := c.engine.DivScalar(ctx, mT, mHatCorr)
	if err != nil {
		c.adamStepScalar(params, mState, vState, lr, tStep, beta1, beta2, eps, wd)
		return
	}
	vHat, err := c.engine.DivScalar(ctx, vT, vHatCorr)
	if err != nil {
		c.adamStepScalar(params, mState, vState, lr, tStep, beta1, beta2, eps, wd)
		return
	}
	vSqrt, err := c.engine.Sqrt(ctx, vHat)
	if err != nil {
		c.adamStepScalar(params, mState, vState, lr, tStep, beta1, beta2, eps, wd)
		return
	}
	vSqrtEps, err := c.engine.AddScalar(ctx, vSqrt, eps)
	if err != nil {
		c.adamStepScalar(params, mState, vState, lr, tStep, beta1, beta2, eps, wd)
		return
	}
	adamUpdate, err := c.engine.Div(ctx, mHat, vSqrtEps)
	if err != nil {
		c.adamStepScalar(params, mState, vState, lr, tStep, beta1, beta2, eps, wd)
		return
	}
	wdTerm, err := c.engine.MulScalar(ctx, pT, wd)
	if err != nil {
		c.adamStepScalar(params, mState, vState, lr, tStep, beta1, beta2, eps, wd)
		return
	}
	total, err := c.engine.Add(ctx, adamUpdate, wdTerm)
	if err != nil {
		c.adamStepScalar(params, mState, vState, lr, tStep, beta1, beta2, eps, wd)
		return
	}
	scaled, err := c.engine.MulScalar(ctx, total, lr)
	if err != nil {
		c.adamStepScalar(params, mState, vState, lr, tStep, beta1, beta2, eps, wd)
		return
	}
	newP, err := c.engine.Sub(ctx, pT, scaled)
	if err != nil {
		c.adamStepScalar(params, mState, vState, lr, tStep, beta1, beta2, eps, wd)
		return
	}

	copy(params, newP.Data())
}

// adamStepScalar is the scalar fallback for AdamW when engine operations fail.
func (c *CfC) adamStepScalar(params, mState, vState []float32, lr, tStep, beta1, beta2, eps, wd float32) {
	for i := range params {
		mHat := mState[i] / (1 - float32(math.Pow(float64(beta1), float64(tStep))))
		vHat := vState[i] / (1 - float32(math.Pow(float64(beta2), float64(tStep))))
		params[i] -= lr * (mHat/(float32(math.Sqrt(float64(vHat)))+eps) + wd*params[i])
	}
}
