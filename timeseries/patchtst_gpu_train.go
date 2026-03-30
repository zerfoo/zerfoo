package timeseries

import (
	"context"
	"fmt"
	"math"

	"github.com/zerfoo/ztensor/tensor"
)

// gpuParams holds all PatchTST parameters as float32 tensors for GPU training.
type gpuParams struct {
	patchEmbW *tensor.TensorNumeric[float32] // [patchLen, dModel]
	patchEmbB *tensor.TensorNumeric[float32] // [1, dModel]
	posEmb    *tensor.TensorNumeric[float32] // [numPatches, dModel]
	layers    []gpuEncoderLayer
	headW     *tensor.TensorNumeric[float32] // [headIn, outDim]
	headB     *tensor.TensorNumeric[float32] // [1, outDim]
}

type gpuEncoderLayer struct {
	qW, qB       *tensor.TensorNumeric[float32]
	kW, kB       *tensor.TensorNumeric[float32]
	vW, vB       *tensor.TensorNumeric[float32]
	oW, oB       *tensor.TensorNumeric[float32]
	ffn1W, ffn1B *tensor.TensorNumeric[float32]
	ffn2W, ffn2B *tensor.TensorNumeric[float32]
	norm1, bias1 *tensor.TensorNumeric[float32]
	norm2, bias2 *tensor.TensorNumeric[float32]
}

// gpuGrads mirrors gpuParams for gradient accumulation.
type gpuGrads struct {
	patchEmbW *tensor.TensorNumeric[float32]
	patchEmbB *tensor.TensorNumeric[float32]
	posEmb    *tensor.TensorNumeric[float32]
	layers    []gpuEncoderLayer
	headW     *tensor.TensorNumeric[float32]
	headB     *tensor.TensorNumeric[float32]
}

// gpuAdamState holds first and second moment tensors for AdamW.
type gpuAdamState struct {
	m [](*tensor.TensorNumeric[float32])
	v [](*tensor.TensorNumeric[float32])
}

// extractGPUParams converts the PatchTST float32 model weights to gpuParams tensors.
func (m *PatchTST) extractGPUParams() (*gpuParams, error) {
	p := &gpuParams{}
	var err error
	dModel := m.config.DModel
	numPatches := m.config.NumPatches()
	ffnDim := dModel * 4
	outDim := m.config.OutputDim

	// Clone weights so we own the data.
	clone := func(src *tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
		data := make([]float32, len(src.Data()))
		copy(data, src.Data())
		return tensor.New[float32](src.Shape(), data)
	}

	// Patch embedding: weights [patchLen, dModel], bias [1, dModel].
	p.patchEmbW, err = clone(m.patchEmb.weights)
	if err != nil {
		return nil, err
	}
	biasData := make([]float32, dModel)
	copy(biasData, m.patchEmb.biases.Data())
	p.patchEmbB, err = tensor.New[float32]([]int{1, dModel}, biasData)
	if err != nil {
		return nil, err
	}

	// Positional embedding: [numPatches, dModel].
	posData := make([]float32, numPatches*dModel)
	copy(posData, m.posEmb.Data())
	p.posEmb, err = tensor.New[float32]([]int{numPatches, dModel}, posData)
	if err != nil {
		return nil, err
	}

	p.layers = make([]gpuEncoderLayer, m.config.NLayers)
	for i, l := range m.layers {
		gl := &p.layers[i]
		gl.qW, err = clone(l.qProj.weights)
		if err != nil {
			return nil, err
		}
		gl.qB, err = reshapeBias(l.qProj.biases, dModel)
		if err != nil {
			return nil, err
		}
		gl.kW, err = clone(l.kProj.weights)
		if err != nil {
			return nil, err
		}
		gl.kB, err = reshapeBias(l.kProj.biases, dModel)
		if err != nil {
			return nil, err
		}
		gl.vW, err = clone(l.vProj.weights)
		if err != nil {
			return nil, err
		}
		gl.vB, err = reshapeBias(l.vProj.biases, dModel)
		if err != nil {
			return nil, err
		}
		gl.oW, err = clone(l.oProj.weights)
		if err != nil {
			return nil, err
		}
		gl.oB, err = reshapeBias(l.oProj.biases, dModel)
		if err != nil {
			return nil, err
		}
		gl.ffn1W, err = clone(l.ffn1.weights)
		if err != nil {
			return nil, err
		}
		gl.ffn1B, err = reshapeBias(l.ffn1.biases, ffnDim)
		if err != nil {
			return nil, err
		}
		gl.ffn2W, err = clone(l.ffn2.weights)
		if err != nil {
			return nil, err
		}
		gl.ffn2B, err = reshapeBias(l.ffn2.biases, dModel)
		if err != nil {
			return nil, err
		}
		// Norm weights: [1, dModel] for broadcasting.
		gl.norm1, err = reshapeBias(l.norm1, dModel)
		if err != nil {
			return nil, err
		}
		gl.bias1, err = reshapeBias(l.bias1, dModel)
		if err != nil {
			return nil, err
		}
		gl.norm2, err = reshapeBias(l.norm2, dModel)
		if err != nil {
			return nil, err
		}
		gl.bias2, err = reshapeBias(l.bias2, dModel)
		if err != nil {
			return nil, err
		}
	}

	// Head: [headIn, outDim].
	p.headW, err = clone(m.head.weights)
	if err != nil {
		return nil, err
	}
	headBData := make([]float32, outDim)
	copy(headBData, m.head.biases.Data())
	p.headB, err = tensor.New[float32]([]int{1, outDim}, headBData)
	if err != nil {
		return nil, err
	}

	return p, nil
}

func reshapeBias(t *tensor.TensorNumeric[float32], dim int) (*tensor.TensorNumeric[float32], error) {
	data := make([]float32, dim)
	copy(data, t.Data())
	return tensor.New[float32]([]int{1, dim}, data)
}

// allParamTensors returns all parameter tensors in flat order matching flatParams.
func (p *gpuParams) allParamTensors() []*tensor.TensorNumeric[float32] {
	var ts []*tensor.TensorNumeric[float32]
	ts = append(ts, p.patchEmbW, p.patchEmbB, p.posEmb)
	for i := range p.layers {
		l := &p.layers[i]
		ts = append(ts, l.qW, l.qB, l.kW, l.kB, l.vW, l.vB, l.oW, l.oB,
			l.ffn1W, l.ffn1B, l.ffn2W, l.ffn2B, l.norm1, l.bias1, l.norm2, l.bias2)
	}
	ts = append(ts, p.headW, p.headB)
	return ts
}

// zerosLike creates a new tensor with the same shape filled with zeros.
func zerosLike(t *tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
	return tensor.New[float32](t.Shape(), make([]float32, len(t.Data())))
}

// allocGrads creates zero gradient tensors matching params.
func allocGrads(p *gpuParams) (*gpuGrads, error) {
	g := &gpuGrads{}
	var err error
	g.patchEmbW, err = zerosLike(p.patchEmbW)
	if err != nil {
		return nil, err
	}
	g.patchEmbB, err = zerosLike(p.patchEmbB)
	if err != nil {
		return nil, err
	}
	g.posEmb, err = zerosLike(p.posEmb)
	if err != nil {
		return nil, err
	}
	g.layers = make([]gpuEncoderLayer, len(p.layers))
	for i := range p.layers {
		gl := &g.layers[i]
		pl := &p.layers[i]
		gl.qW, err = zerosLike(pl.qW)
		if err != nil {
			return nil, err
		}
		gl.qB, err = zerosLike(pl.qB)
		if err != nil {
			return nil, err
		}
		gl.kW, err = zerosLike(pl.kW)
		if err != nil {
			return nil, err
		}
		gl.kB, err = zerosLike(pl.kB)
		if err != nil {
			return nil, err
		}
		gl.vW, err = zerosLike(pl.vW)
		if err != nil {
			return nil, err
		}
		gl.vB, err = zerosLike(pl.vB)
		if err != nil {
			return nil, err
		}
		gl.oW, err = zerosLike(pl.oW)
		if err != nil {
			return nil, err
		}
		gl.oB, err = zerosLike(pl.oB)
		if err != nil {
			return nil, err
		}
		gl.ffn1W, err = zerosLike(pl.ffn1W)
		if err != nil {
			return nil, err
		}
		gl.ffn1B, err = zerosLike(pl.ffn1B)
		if err != nil {
			return nil, err
		}
		gl.ffn2W, err = zerosLike(pl.ffn2W)
		if err != nil {
			return nil, err
		}
		gl.ffn2B, err = zerosLike(pl.ffn2B)
		if err != nil {
			return nil, err
		}
		gl.norm1, err = zerosLike(pl.norm1)
		if err != nil {
			return nil, err
		}
		gl.bias1, err = zerosLike(pl.bias1)
		if err != nil {
			return nil, err
		}
		gl.norm2, err = zerosLike(pl.norm2)
		if err != nil {
			return nil, err
		}
		gl.bias2, err = zerosLike(pl.bias2)
		if err != nil {
			return nil, err
		}
	}
	g.headW, err = zerosLike(p.headW)
	if err != nil {
		return nil, err
	}
	g.headB, err = zerosLike(p.headB)
	if err != nil {
		return nil, err
	}
	return g, nil
}

func (g *gpuGrads) allParamTensors() []*tensor.TensorNumeric[float32] {
	var ts []*tensor.TensorNumeric[float32]
	ts = append(ts, g.patchEmbW, g.patchEmbB, g.posEmb)
	for i := range g.layers {
		l := &g.layers[i]
		ts = append(ts, l.qW, l.qB, l.kW, l.kB, l.vW, l.vB, l.oW, l.oB,
			l.ffn1W, l.ffn1B, l.ffn2W, l.ffn2B, l.norm1, l.bias1, l.norm2, l.bias2)
	}
	ts = append(ts, g.headW, g.headB)
	return ts
}

// gpuLayerCache stores per-sample per-layer forward activations needed for backward.
type gpuLayerCache struct {
	normed1   *tensor.TensorNumeric[float32] // [numPatches, dModel]
	q, k, v   *tensor.TensorNumeric[float32] // [numPatches, dModel]
	scores    [][][]float32                   // [nHeads][seq][seq] - small, CPU
	attnOut   *tensor.TensorNumeric[float32]  // [numPatches, dModel]
	normed2   *tensor.TensorNumeric[float32]  // [numPatches, dModel]
	ffn1PreAct *tensor.TensorNumeric[float32] // [numPatches, ffnDim]
	ffn1Out   *tensor.TensorNumeric[float32]  // [numPatches, ffnDim]
	// LayerNorm cache (CPU, small).
	centered1 [][]float32 // [numPatches][dModel]
	invStd1   []float32   // [numPatches]
	centered2 [][]float32
	invStd2   []float32
}

// gpuChannelCache stores per-sample per-channel data.
type gpuChannelCache struct {
	patches    *tensor.TensorNumeric[float32] // [numPatches, patchLen]
	embedded   *tensor.TensorNumeric[float32] // [numPatches, dModel]
	flatInput  *tensor.TensorNumeric[float32] // [1, headIn]
	layerCaches []gpuLayerCache
}

// gpuSampleCache is the full per-sample cache.
type gpuSampleCache struct {
	channels []gpuChannelCache
}

// layerNormF32WithCache performs layer norm on CPU and returns cached values.
// x: [seq][dModel], scale/bias: [dModel].
func layerNormF32WithCache(x [][]float32, scale, bias []float32, dModel int) ([][]float32, [][]float32, []float32) {
	seq := len(x)
	normed := make([][]float32, seq)
	centered := make([][]float32, seq)
	invStds := make([]float32, seq)

	for s := 0; s < seq; s++ {
		mean := float32(0)
		for j := 0; j < dModel; j++ {
			mean += x[s][j]
		}
		mean /= float32(dModel)

		variance := float32(0)
		centered[s] = make([]float32, dModel)
		for j := 0; j < dModel; j++ {
			centered[s][j] = x[s][j] - mean
			variance += centered[s][j] * centered[s][j]
		}
		variance /= float32(dModel)
		invStd := float32(1.0 / math.Sqrt(float64(variance)+1e-5))
		invStds[s] = invStd

		normed[s] = make([]float32, dModel)
		for j := 0; j < dModel; j++ {
			normed[s][j] = centered[s][j]*invStd*scale[j] + bias[j]
		}
	}
	return normed, centered, invStds
}

// layerNormBackwardF32 computes backward pass through layer norm on CPU.
// Accumulates into dScale, dBias.
func layerNormBackwardF32(dOut, centered [][]float32, invStd []float32, scale, dScale, dBias []float32, dModel int) [][]float32 {
	seq := len(dOut)
	dInput := make([][]float32, seq)
	d := float32(dModel)

	for s := 0; s < seq; s++ {
		dInput[s] = make([]float32, dModel)
		// dScale += dOut * centered * invStd
		// dBias += dOut
		dotScaleGrad := float32(0)
		dotMeanGrad := float32(0)
		for j := 0; j < dModel; j++ {
			normVal := centered[s][j] * invStd[s]
			dScale[j] += dOut[s][j] * normVal
			dBias[j] += dOut[s][j]
			// dNorm = dOut * scale
			dNorm := dOut[s][j] * scale[j]
			dotScaleGrad += dNorm * centered[s][j]
			dotMeanGrad += dNorm
		}
		for j := 0; j < dModel; j++ {
			dNorm := dOut[s][j] * scale[j]
			dInput[s][j] = invStd[s] * (dNorm - (dotMeanGrad+centered[s][j]*invStd[s]*invStd[s]*dotScaleGrad)/d)
		}
	}
	return dInput
}

// trainWindowedGPU runs the full GPU training loop for PatchTST.
// All parameters, gradients, and optimizer moments are kept as float32 tensors.
// Forward and backward linear operations use engine.MatMul.
// Softmax, GELU, and layer norm backward run on CPU (small tensors).
func (m *PatchTST) trainWindowedGPU(windows [][][]float64, labels []float64, config TrainConfig) (*TrainResult, error) {
	ctx := context.Background()
	nSamples := len(windows)
	outDim := m.config.OutputDim
	numPatches := m.config.NumPatches()
	dModel := m.config.DModel
	nHeads := m.config.NHeads
	headDim := dModel / nHeads
	ffnDim := dModel * 4
	headIn := numPatches * dModel

	// Extract params as float32 tensors.
	params, err := m.extractGPUParams()
	if err != nil {
		return nil, fmt.Errorf("patchtst gpu: extract params: %w", err)
	}

	// Allocate gradient tensors (zeroed each batch).
	grads, err := allocGrads(params)
	if err != nil {
		return nil, fmt.Errorf("patchtst gpu: alloc grads: %w", err)
	}

	// Allocate AdamW moment tensors.
	paramTs := params.allParamTensors()
	nParamTensors := len(paramTs)
	adamM := make([]*tensor.TensorNumeric[float32], nParamTensors)
	adamV := make([]*tensor.TensorNumeric[float32], nParamTensors)
	for i, pt := range paramTs {
		adamM[i], err = zerosLike(pt)
		if err != nil {
			return nil, err
		}
		adamV[i], err = zerosLike(pt)
		if err != nil {
			return nil, err
		}
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

			// Zero all gradients.
			gradTs := grads.allParamTensors()
			for _, gt := range gradTs {
				if err := m.engine.Zero(ctx, gt); err != nil {
					return nil, fmt.Errorf("patchtst gpu: zero grad: %w", err)
				}
			}

			channels := len(windows[0])
			chanScale := float32(1.0 / float64(channels))

			// Forward pass + backward for each sample in batch.
			batchLoss := 0.0
			for s := 0; s < bs; s++ {
				sIdx := start + s
				sampleLabels := labels[sIdx*outDim : (sIdx+1)*outDim]

				// Convert input to float32.
				sampleWindows := make([][]float32, channels)
				for ch := 0; ch < channels; ch++ {
					sampleWindows[ch] = make([]float32, m.config.InputLength)
					for j := range sampleWindows[ch] {
						sampleWindows[ch][j] = float32(windows[sIdx][ch][j])
					}
				}

				// Forward pass per channel, accumulating output.
				chanOutputs := make([][]float32, channels)
				caches := make([]gpuChannelCache, channels)

				for ch := 0; ch < channels; ch++ {
					cc := &caches[ch]

					// Extract patches.
					patchData := make([]float32, numPatches*m.config.PatchLength)
					for p := 0; p < numPatches; p++ {
						startP := p * m.config.Stride
						copy(patchData[p*m.config.PatchLength:(p+1)*m.config.PatchLength],
							sampleWindows[ch][startP:startP+m.config.PatchLength])
					}
					cc.patches, err = tensor.New[float32]([]int{numPatches, m.config.PatchLength}, patchData)
					if err != nil {
						return nil, err
					}

					// Patch embedding: patches @ patchEmbW + patchEmbB.
					cc.embedded, err = m.engine.MatMul(ctx, cc.patches, params.patchEmbW)
					if err != nil {
						return nil, fmt.Errorf("gpu fwd patch emb: %w", err)
					}
					cc.embedded, err = m.engine.Add(ctx, cc.embedded, params.patchEmbB)
					if err != nil {
						return nil, err
					}

					// Add positional embedding.
					x, err := m.engine.Add(ctx, cc.embedded, params.posEmb)
					if err != nil {
						return nil, err
					}

					// Encoder layers.
					cc.layerCaches = make([]gpuLayerCache, m.config.NLayers)
					for li := 0; li < m.config.NLayers; li++ {
						layer := &params.layers[li]
						lc := &cc.layerCaches[li]

						// Layer norm 1 on CPU.
						xData := matFromTensor(x, numPatches, dModel)
						normed1, cent1, invStd1 := layerNormF32WithCache(xData, layer.norm1.Data(), layer.bias1.Data(), dModel)
						lc.centered1 = cent1
						lc.invStd1 = invStd1
						lc.normed1, err = tensorFromMat(normed1, numPatches, dModel)
						if err != nil {
							return nil, err
						}

						// Q/K/V projections via engine.
						lc.q, err = m.engine.MatMul(ctx, lc.normed1, layer.qW)
						if err != nil {
							return nil, err
						}
						lc.q, err = m.engine.Add(ctx, lc.q, layer.qB)
						if err != nil {
							return nil, err
						}
						lc.k, err = m.engine.MatMul(ctx, lc.normed1, layer.kW)
						if err != nil {
							return nil, err
						}
						lc.k, err = m.engine.Add(ctx, lc.k, layer.kB)
						if err != nil {
							return nil, err
						}
						lc.v, err = m.engine.MatMul(ctx, lc.normed1, layer.vW)
						if err != nil {
							return nil, err
						}
						lc.v, err = m.engine.Add(ctx, lc.v, layer.vB)
						if err != nil {
							return nil, err
						}

						// Attention on CPU (small seq x seq matrices).
						qData := lc.q.Data()
						kData := lc.k.Data()
						vData := lc.v.Data()
						seq := numPatches
						lc.scores = make([][][]float32, nHeads)
						attnOutData := make([]float32, seq*dModel)

						scale := float32(1.0 / math.Sqrt(float64(headDim)))
						for h := 0; h < nHeads; h++ {
							hOff := h * headDim
							lc.scores[h] = make([][]float32, seq)
							for i := 0; i < seq; i++ {
								lc.scores[h][i] = make([]float32, seq)
								for j := 0; j < seq; j++ {
									dot := float32(0)
									for d := 0; d < headDim; d++ {
										dot += qData[i*dModel+hOff+d] * kData[j*dModel+hOff+d]
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
								sumExp := float32(0)
								for j := 0; j < seq; j++ {
									lc.scores[h][i][j] = float32(math.Exp(float64(lc.scores[h][i][j] - maxS)))
									sumExp += lc.scores[h][i][j]
								}
								for j := 0; j < seq; j++ {
									lc.scores[h][i][j] /= sumExp
								}
							}
							// Weighted sum.
							for i := 0; i < seq; i++ {
								for d := 0; d < headDim; d++ {
									val := float32(0)
									for j := 0; j < seq; j++ {
										val += lc.scores[h][i][j] * vData[j*dModel+hOff+d]
									}
									attnOutData[i*dModel+hOff+d] = val
								}
							}
						}
						lc.attnOut, err = tensor.New[float32]([]int{seq, dModel}, attnOutData)
						if err != nil {
							return nil, err
						}

						// Output projection.
						attnProj, err := m.engine.MatMul(ctx, lc.attnOut, layer.oW)
						if err != nil {
							return nil, err
						}
						attnProj, err = m.engine.Add(ctx, attnProj, layer.oB)
						if err != nil {
							return nil, err
						}

						// Residual 1.
						x, err = m.engine.Add(ctx, x, attnProj)
						if err != nil {
							return nil, err
						}

						// Layer norm 2 on CPU.
						xData = matFromTensor(x, numPatches, dModel)
						normed2, cent2, invStd2 := layerNormF32WithCache(xData, layer.norm2.Data(), layer.bias2.Data(), dModel)
						lc.centered2 = cent2
						lc.invStd2 = invStd2
						lc.normed2, err = tensorFromMat(normed2, numPatches, dModel)
						if err != nil {
							return nil, err
						}

						// FFN1 via engine.
						lc.ffn1PreAct, err = m.engine.MatMul(ctx, lc.normed2, layer.ffn1W)
						if err != nil {
							return nil, err
						}
						lc.ffn1PreAct, err = m.engine.Add(ctx, lc.ffn1PreAct, layer.ffn1B)
						if err != nil {
							return nil, err
						}

						// GELU on CPU.
						ffn1Data := lc.ffn1PreAct.Data()
						ffn1OutData := make([]float32, len(ffn1Data))
						for j := range ffn1Data {
							ffn1OutData[j] = geluScalar(ffn1Data[j])
						}
						lc.ffn1Out, err = tensor.New[float32]([]int{numPatches, ffnDim}, ffn1OutData)
						if err != nil {
							return nil, err
						}

						// FFN2 via engine.
						ffn2Out, err := m.engine.MatMul(ctx, lc.ffn1Out, layer.ffn2W)
						if err != nil {
							return nil, err
						}
						ffn2Out, err = m.engine.Add(ctx, ffn2Out, layer.ffn2B)
						if err != nil {
							return nil, err
						}

						// Residual 2.
						x, err = m.engine.Add(ctx, x, ffn2Out)
						if err != nil {
							return nil, err
						}
					}

					// Flatten for output head.
					flatData := make([]float32, headIn)
					copy(flatData, matFromTensor(x, numPatches, dModel)[0][:0]) // noop; use x.Data()
					xd := x.Data()
					copy(flatData, xd)
					cc.flatInput, err = tensor.New[float32]([]int{1, headIn}, flatData)
					if err != nil {
						return nil, err
					}

					// Output head.
					headOut, err := m.engine.MatMul(ctx, cc.flatInput, params.headW)
					if err != nil {
						return nil, err
					}
					headOut, err = m.engine.Add(ctx, headOut, params.headB)
					if err != nil {
						return nil, err
					}
					chanOutputs[ch] = headOut.Data()
				}

				// Average channel outputs.
				pred := make([]float32, outDim)
				for ch := 0; ch < channels; ch++ {
					for j := 0; j < outDim; j++ {
						pred[j] += chanOutputs[ch][j]
					}
				}
				for j := range pred {
					pred[j] /= float32(channels)
				}

				// Compute MSE loss.
				for j := 0; j < outDim; j++ {
					diff := float64(pred[j]) - sampleLabels[j]
					batchLoss += diff * diff
				}

				// --- Backward pass ---
				// dL/dPred for this sample.
				dPred := make([]float32, outDim)
				for j := 0; j < outDim; j++ {
					dPred[j] = 2.0 * (pred[j] - float32(sampleLabels[j])) / float32(bs*outDim)
				}

				for ch := 0; ch < channels; ch++ {
					cc := &caches[ch]

					// Scale by 1/channels.
					dChanOut := make([]float32, outDim)
					for j := range dChanOut {
						dChanOut[j] = dPred[j] * chanScale
					}
					dChanOutT, err := tensor.New[float32]([]int{1, outDim}, dChanOut)
					if err != nil {
						return nil, err
					}

					// Head backward: out = flatInput @ headW + headB.
					// dHeadW += flatInput^T @ dChanOut.
					flatInputT, err := m.engine.Transpose(ctx, cc.flatInput, []int{1, 0})
					if err != nil {
						return nil, err
					}
					dHW, err := m.engine.MatMul(ctx, flatInputT, dChanOutT)
					if err != nil {
						return nil, err
					}
					grads.headW, err = m.engine.Add(ctx, grads.headW, dHW)
					if err != nil {
						return nil, err
					}
					// dHeadB += dChanOut.
					grads.headB, err = m.engine.Add(ctx, grads.headB, dChanOutT)
					if err != nil {
						return nil, err
					}
					// dFlat = dChanOut @ headW^T.
					headWT, err := m.engine.Transpose(ctx, params.headW, []int{1, 0})
					if err != nil {
						return nil, err
					}
					dFlat, err := m.engine.MatMul(ctx, dChanOutT, headWT)
					if err != nil {
						return nil, err
					}

					// Unflatten dFlat to [numPatches, dModel].
					dX, err := m.engine.Reshape(ctx, dFlat, []int{numPatches, dModel})
					if err != nil {
						return nil, err
					}

					// Backward through encoder layers in reverse.
					for li := m.config.NLayers - 1; li >= 0; li-- {
						layer := &params.layers[li]
						lc := &cc.layerCaches[li]
						dg := &grads.layers[li]

						dXData := dX.Data()

						// FFN2 backward: ffn2Out = ffn1Out @ ffn2W + ffn2B.
						dFFN2Out := make([]float32, numPatches*dModel)
						copy(dFFN2Out, dXData)
						dFFN2OutT, err := tensor.New[float32]([]int{numPatches, dModel}, dFFN2Out)
						if err != nil {
							return nil, err
						}

						// dFFN2W += ffn1Out^T @ dFFN2Out.
						ffn1OutT, err := m.engine.Transpose(ctx, lc.ffn1Out, []int{1, 0})
						if err != nil {
							return nil, err
						}
						dFW, err := m.engine.MatMul(ctx, ffn1OutT, dFFN2OutT)
						if err != nil {
							return nil, err
						}
						dg.ffn2W, err = m.engine.Add(ctx, dg.ffn2W, dFW)
						if err != nil {
							return nil, err
						}
						// dFFN2B += sum(dFFN2Out, axis=0).
						dFB, err := m.engine.Sum(ctx, dFFN2OutT, 0, false)
						if err != nil {
							return nil, err
						}
						dFBR, err := m.engine.Reshape(ctx, dFB, []int{1, dModel})
						if err != nil {
							return nil, err
						}
						dg.ffn2B, err = m.engine.Add(ctx, dg.ffn2B, dFBR)
						if err != nil {
							return nil, err
						}
						// dFFN1Out = dFFN2Out @ ffn2W^T.
						ffn2WT, err := m.engine.Transpose(ctx, layer.ffn2W, []int{1, 0})
						if err != nil {
							return nil, err
						}
						dFFN1Out, err := m.engine.MatMul(ctx, dFFN2OutT, ffn2WT)
						if err != nil {
							return nil, err
						}

						// GELU backward on CPU.
						dFFN1OutData := dFFN1Out.Data()
						ffn1PreActData := lc.ffn1PreAct.Data()
						dFFN1PreActData := make([]float32, len(dFFN1OutData))
						for j := range dFFN1OutData {
							dFFN1PreActData[j] = dFFN1OutData[j] * geluDerivF32(ffn1PreActData[j])
						}
						dFFN1PreAct, err := tensor.New[float32]([]int{numPatches, ffnDim}, dFFN1PreActData)
						if err != nil {
							return nil, err
						}

						// FFN1 backward: ffn1PreAct = normed2 @ ffn1W + ffn1B.
						normed2T, err := m.engine.Transpose(ctx, lc.normed2, []int{1, 0})
						if err != nil {
							return nil, err
						}
						dF1W, err := m.engine.MatMul(ctx, normed2T, dFFN1PreAct)
						if err != nil {
							return nil, err
						}
						dg.ffn1W, err = m.engine.Add(ctx, dg.ffn1W, dF1W)
						if err != nil {
							return nil, err
						}
						dF1B, err := m.engine.Sum(ctx, dFFN1PreAct, 0, false)
						if err != nil {
							return nil, err
						}
						dF1BR, err := m.engine.Reshape(ctx, dF1B, []int{1, ffnDim})
						if err != nil {
							return nil, err
						}
						dg.ffn1B, err = m.engine.Add(ctx, dg.ffn1B, dF1BR)
						if err != nil {
							return nil, err
						}
						ffn1WT, err := m.engine.Transpose(ctx, layer.ffn1W, []int{1, 0})
						if err != nil {
							return nil, err
						}
						dNormed2, err := m.engine.MatMul(ctx, dFFN1PreAct, ffn1WT)
						if err != nil {
							return nil, err
						}

						// LayerNorm2 backward on CPU.
						dNormed2Data := matFromTensor(dNormed2, numPatches, dModel)
						dXAfterAttn := layerNormBackwardF32(dNormed2Data, lc.centered2, lc.invStd2,
							layer.norm2.Data(), dg.norm2.Data(), dg.bias2.Data(), dModel)
						// Add residual from FFN path (dX from residual2).
						for p := 0; p < numPatches; p++ {
							for j := 0; j < dModel; j++ {
								dXAfterAttn[p][j] += dXData[p*dModel+j]
							}
						}

						// oProj backward: attnProjOut = attnOut @ oW + oB.
						dAttnProjOutT, err := tensorFromMat(dXAfterAttn, numPatches, dModel)
						if err != nil {
							return nil, err
						}
						attnOutT, err := m.engine.Transpose(ctx, lc.attnOut, []int{1, 0})
						if err != nil {
							return nil, err
						}
						dOW, err := m.engine.MatMul(ctx, attnOutT, dAttnProjOutT)
						if err != nil {
							return nil, err
						}
						dg.oW, err = m.engine.Add(ctx, dg.oW, dOW)
						if err != nil {
							return nil, err
						}
						dOB, err := m.engine.Sum(ctx, dAttnProjOutT, 0, false)
						if err != nil {
							return nil, err
						}
						dOBR, err := m.engine.Reshape(ctx, dOB, []int{1, dModel})
						if err != nil {
							return nil, err
						}
						dg.oB, err = m.engine.Add(ctx, dg.oB, dOBR)
						if err != nil {
							return nil, err
						}
						oWT, err := m.engine.Transpose(ctx, layer.oW, []int{1, 0})
						if err != nil {
							return nil, err
						}
						dAttnOut, err := m.engine.MatMul(ctx, dAttnProjOutT, oWT)
						if err != nil {
							return nil, err
						}

						// Attention backward on CPU.
						dAttnOutData := dAttnOut.Data()
						qData := lc.q.Data()
						kData := lc.k.Data()
						vData := lc.v.Data()
						seq := numPatches
						dQData := make([]float32, seq*dModel)
						dKData := make([]float32, seq*dModel)
						dVData := make([]float32, seq*dModel)

						attnScale := float32(1.0 / math.Sqrt(float64(headDim)))
						for h := 0; h < nHeads; h++ {
							hOff := h * headDim

							dScores := make([][]float32, seq)
							for i := 0; i < seq; i++ {
								dScores[i] = make([]float32, seq)
								for j := 0; j < seq; j++ {
									for d := 0; d < headDim; d++ {
										dScores[i][j] += dAttnOutData[i*dModel+hOff+d] * vData[j*dModel+hOff+d]
										dVData[j*dModel+hOff+d] += lc.scores[h][i][j] * dAttnOutData[i*dModel+hOff+d]
									}
								}
							}

							// Softmax backward.
							for i := 0; i < seq; i++ {
								dot := float32(0)
								for j := 0; j < seq; j++ {
									dot += lc.scores[h][i][j] * dScores[i][j]
								}
								for j := 0; j < seq; j++ {
									dLogit := lc.scores[h][i][j] * (dScores[i][j] - dot) * attnScale
									for d := 0; d < headDim; d++ {
										dQData[i*dModel+hOff+d] += dLogit * kData[j*dModel+hOff+d]
										dKData[j*dModel+hOff+d] += dLogit * qData[i*dModel+hOff+d]
									}
								}
							}
						}

						// Q/K/V projection backward via engine.
						dQT, err := tensor.New[float32]([]int{numPatches, dModel}, dQData)
						if err != nil {
							return nil, err
						}
						dKT, err := tensor.New[float32]([]int{numPatches, dModel}, dKData)
						if err != nil {
							return nil, err
						}
						dVT, err := tensor.New[float32]([]int{numPatches, dModel}, dVData)
						if err != nil {
							return nil, err
						}
						normed1T, err := m.engine.Transpose(ctx, lc.normed1, []int{1, 0})
						if err != nil {
							return nil, err
						}

						// dQW, dKW, dVW.
						dQW, err := m.engine.MatMul(ctx, normed1T, dQT)
						if err != nil {
							return nil, err
						}
						dg.qW, err = m.engine.Add(ctx, dg.qW, dQW)
						if err != nil {
							return nil, err
						}
						dQB, err := m.engine.Sum(ctx, dQT, 0, false)
						if err != nil {
							return nil, err
						}
						dQBR, err := m.engine.Reshape(ctx, dQB, []int{1, dModel})
						if err != nil {
							return nil, err
						}
						dg.qB, err = m.engine.Add(ctx, dg.qB, dQBR)
						if err != nil {
							return nil, err
						}

						dKW, err := m.engine.MatMul(ctx, normed1T, dKT)
						if err != nil {
							return nil, err
						}
						dg.kW, err = m.engine.Add(ctx, dg.kW, dKW)
						if err != nil {
							return nil, err
						}
						dKB, err := m.engine.Sum(ctx, dKT, 0, false)
						if err != nil {
							return nil, err
						}
						dKBR, err := m.engine.Reshape(ctx, dKB, []int{1, dModel})
						if err != nil {
							return nil, err
						}
						dg.kB, err = m.engine.Add(ctx, dg.kB, dKBR)
						if err != nil {
							return nil, err
						}

						dVW, err := m.engine.MatMul(ctx, normed1T, dVT)
						if err != nil {
							return nil, err
						}
						dg.vW, err = m.engine.Add(ctx, dg.vW, dVW)
						if err != nil {
							return nil, err
						}
						dVB, err := m.engine.Sum(ctx, dVT, 0, false)
						if err != nil {
							return nil, err
						}
						dVBR, err := m.engine.Reshape(ctx, dVB, []int{1, dModel})
						if err != nil {
							return nil, err
						}
						dg.vB, err = m.engine.Add(ctx, dg.vB, dVBR)
						if err != nil {
							return nil, err
						}

						// dNormed1 = dQ @ qW^T + dK @ kW^T + dV @ vW^T.
						qWT, err := m.engine.Transpose(ctx, layer.qW, []int{1, 0})
						if err != nil {
							return nil, err
						}
						dN1q, err := m.engine.MatMul(ctx, dQT, qWT)
						if err != nil {
							return nil, err
						}
						kWT, err := m.engine.Transpose(ctx, layer.kW, []int{1, 0})
						if err != nil {
							return nil, err
						}
						dN1k, err := m.engine.MatMul(ctx, dKT, kWT)
						if err != nil {
							return nil, err
						}
						vWT, err := m.engine.Transpose(ctx, layer.vW, []int{1, 0})
						if err != nil {
							return nil, err
						}
						dN1v, err := m.engine.MatMul(ctx, dVT, vWT)
						if err != nil {
							return nil, err
						}
						dNormed1, err := m.engine.Add(ctx, dN1q, dN1k)
						if err != nil {
							return nil, err
						}
						dNormed1, err = m.engine.Add(ctx, dNormed1, dN1v)
						if err != nil {
							return nil, err
						}

						// LayerNorm1 backward on CPU.
						dNormed1Data := matFromTensor(dNormed1, numPatches, dModel)
						dLayerInput := layerNormBackwardF32(dNormed1Data, lc.centered1, lc.invStd1,
							layer.norm1.Data(), dg.norm1.Data(), dg.bias1.Data(), dModel)
						// Add residual from attention path.
						for p := 0; p < numPatches; p++ {
							for j := 0; j < dModel; j++ {
								dLayerInput[p][j] += dXAfterAttn[p][j]
							}
						}

						dX, err = tensorFromMat(dLayerInput, numPatches, dModel)
						if err != nil {
							return nil, err
						}
					}

					// Positional embedding gradient.
					dPosData := grads.posEmb.Data()
					dXData := dX.Data()
					for j := range dPosData {
						dPosData[j] += dXData[j]
					}

					// Patch embedding backward: embedded = patches @ patchEmbW + patchEmbB.
					patchesT, err := m.engine.Transpose(ctx, cc.patches, []int{1, 0})
					if err != nil {
						return nil, err
					}
					dPEW, err := m.engine.MatMul(ctx, patchesT, dX)
					if err != nil {
						return nil, err
					}
					grads.patchEmbW, err = m.engine.Add(ctx, grads.patchEmbW, dPEW)
					if err != nil {
						return nil, err
					}
					dPEB, err := m.engine.Sum(ctx, dX, 0, false)
					if err != nil {
						return nil, err
					}
					dPEBR, err := m.engine.Reshape(ctx, dPEB, []int{1, dModel})
					if err != nil {
						return nil, err
					}
					grads.patchEmbB, err = m.engine.Add(ctx, grads.patchEmbB, dPEBR)
					if err != nil {
						return nil, err
					}
				}
			}

			batchLoss /= float64(bs * outDim)
			epochLoss += batchLoss
			nBatches++

			// Gradient clipping.
			if config.GradClip > 0 {
				gradTs := grads.allParamTensors()
				norm := float64(0)
				for _, gt := range gradTs {
					for _, v := range gt.Data() {
						norm += float64(v) * float64(v)
					}
				}
				norm = math.Sqrt(norm)
				if norm > config.GradClip {
					scale := float32(config.GradClip / norm)
					for _, gt := range gradTs {
						_, err := m.engine.MulScalar(ctx, gt, scale, gt)
						if err != nil {
							return nil, fmt.Errorf("patchtst gpu: grad clip: %w", err)
						}
					}
				}
			}

			// AdamW update.
			lr := warmupLR(config.LR, epoch, config.WarmupEpochs)
			t := float64(epoch*((nSamples+batchSize-1)/batchSize) + nBatches)
			beta1 := float32(config.Beta1)
			beta2 := float32(config.Beta2)
			eps := float32(config.Epsilon)
			lrF := float32(lr)
			mCorr := float32(1.0 / (1.0 - math.Pow(config.Beta1, t)))
			vCorr := float32(1.0 / (1.0 - math.Pow(config.Beta2, t)))
			wdF := float32(config.WeightDecay)

			paramTs := params.allParamTensors()
			gradTs = grads.allParamTensors()
			for i := range paramTs {
				// AdamW step on CPU for simplicity and correctness.
				pData := paramTs[i].Data()
				gData := gradTs[i].Data()
				mData := adamM[i].Data()
				vData := adamV[i].Data()
				for j := range pData {
					mData[j] = beta1*mData[j] + (1-beta1)*gData[j]
					vData[j] = beta2*vData[j] + (1-beta2)*gData[j]*gData[j]
					mHat := mData[j] * mCorr
					vHat := vData[j] * vCorr
					pData[j] -= lrF * (mHat/(float32(math.Sqrt(float64(vHat)))+eps) + wdF*pData[j])
				}
			}
		}

		result.LossHistory[epoch] = epochLoss / float64(nBatches)
		result.FinalLoss = result.LossHistory[epoch]

		if !isFinite(result.FinalLoss) {
			return nil, fmt.Errorf("patchtst: training diverged at epoch %d: loss=%v", epoch, result.FinalLoss)
		}
	}

	// Write optimized params back to model tensors.
	m.writeBackF32FromGPU(params)

	result.Metrics = map[string]float64{"mse": result.FinalLoss}
	return result, nil
}

// matFromTensor extracts a [][]float32 from a 2D tensor.
func matFromTensor(t *tensor.TensorNumeric[float32], rows, cols int) [][]float32 {
	data := t.Data()
	result := make([][]float32, rows)
	for i := 0; i < rows; i++ {
		result[i] = data[i*cols : (i+1)*cols]
	}
	return result
}

// tensorFromMat creates a 2D tensor from [][]float32.
func tensorFromMat(m [][]float32, rows, cols int) (*tensor.TensorNumeric[float32], error) {
	data := make([]float32, rows*cols)
	for i := 0; i < rows; i++ {
		copy(data[i*cols:(i+1)*cols], m[i])
	}
	return tensor.New[float32]([]int{rows, cols}, data)
}

// geluDerivF32 computes the GELU derivative in float32.
func geluDerivF32(x float32) float32 {
	xf := float64(x)
	c := math.Sqrt(2.0 / math.Pi)
	inner := c * (xf + 0.044715*xf*xf*xf)
	tanh := math.Tanh(inner)
	dInner := c * (1 + 3*0.044715*xf*xf)
	return float32(0.5 * (1 + tanh) + 0.5*xf*(1-tanh*tanh)*dInner)
}

// writeBackF32FromGPU writes GPU params back to model float32 tensors.
func (m *PatchTST) writeBackF32FromGPU(p *gpuParams) {
	copy(m.patchEmb.weights.Data(), p.patchEmbW.Data())
	copy(m.patchEmb.biases.Data(), p.patchEmbB.Data())
	copy(m.posEmb.Data(), p.posEmb.Data())

	for i := range m.layers {
		copy(m.layers[i].qProj.weights.Data(), p.layers[i].qW.Data())
		copy(m.layers[i].qProj.biases.Data(), p.layers[i].qB.Data())
		copy(m.layers[i].kProj.weights.Data(), p.layers[i].kW.Data())
		copy(m.layers[i].kProj.biases.Data(), p.layers[i].kB.Data())
		copy(m.layers[i].vProj.weights.Data(), p.layers[i].vW.Data())
		copy(m.layers[i].vProj.biases.Data(), p.layers[i].vB.Data())
		copy(m.layers[i].oProj.weights.Data(), p.layers[i].oW.Data())
		copy(m.layers[i].oProj.biases.Data(), p.layers[i].oB.Data())
		copy(m.layers[i].ffn1.weights.Data(), p.layers[i].ffn1W.Data())
		copy(m.layers[i].ffn1.biases.Data(), p.layers[i].ffn1B.Data())
		copy(m.layers[i].ffn2.weights.Data(), p.layers[i].ffn2W.Data())
		copy(m.layers[i].ffn2.biases.Data(), p.layers[i].ffn2B.Data())
		copy(m.layers[i].norm1.Data(), p.layers[i].norm1.Data())
		copy(m.layers[i].bias1.Data(), p.layers[i].bias1.Data())
		copy(m.layers[i].norm2.Data(), p.layers[i].norm2.Data())
		copy(m.layers[i].bias2.Data(), p.layers[i].bias2.Data())
	}

	copy(m.head.weights.Data(), p.headW.Data())
	copy(m.head.biases.Data(), p.headB.Data())
}
