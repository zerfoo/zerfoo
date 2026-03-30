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

// gpuBatchLayerCache stores per-layer forward activations for the entire batch.
type gpuBatchLayerCache struct {
	normed1    *tensor.TensorNumeric[float32] // [bs*numPatches, dModel]
	q, k, v    *tensor.TensorNumeric[float32] // [bs*numPatches, dModel]
	scoresTensor *tensor.TensorNumeric[float32] // [bs*nHeads, seq, seq] - batched attention scores
	attnOut    *tensor.TensorNumeric[float32]  // [bs*numPatches, dModel]
	normed2    *tensor.TensorNumeric[float32]  // [bs*numPatches, dModel]
	ffn1PreAct *tensor.TensorNumeric[float32]  // [bs*numPatches, ffnDim]
	ffn1Out    *tensor.TensorNumeric[float32]  // [bs*numPatches, ffnDim]
	centered1  *tensor.TensorNumeric[float32]   // [bs*numPatches, dModel]
	invStd1    *tensor.TensorNumeric[float32]  // [bs*numPatches, 1]
	centered2  *tensor.TensorNumeric[float32]  // [bs*numPatches, dModel]
	invStd2    *tensor.TensorNumeric[float32]  // [bs*numPatches, 1]
	xResidual   *tensor.TensorNumeric[float32] // [bs*numPatches, dModel] input to layer
	xAfterAttn  *tensor.TensorNumeric[float32] // [bs*numPatches, dModel] after residual 1
	geluTanhVal *tensor.TensorNumeric[float32] // [bs*numPatches, ffnDim] cached for backward
}

// gpuBatchForwardCache stores batched forward data across all channels.
type gpuBatchForwardCache struct {
	patches     *tensor.TensorNumeric[float32] // [bsC*numPatches, patchLen]
	flatInput   *tensor.TensorNumeric[float32] // [bsC, headIn]
	layerCaches []gpuBatchLayerCache
}

// copyMatToTensor copies a [][]float32 matrix into an existing tensor's data buffer.
func copyMatToTensor(mat [][]float32, dst *tensor.TensorNumeric[float32]) {
	data := dst.Data()
	cols := len(mat[0])
	for i, row := range mat {
		copy(data[i*cols:(i+1)*cols], row)
	}
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

// layerNormForwardEngine delegates to the standalone layerNormForwardWithEngine.
func (m *PatchTST) layerNormForwardEngine(ctx context.Context, x, scale, bias *tensor.TensorNumeric[float32], rows, dModel int) (normed, centered, invStd *tensor.TensorNumeric[float32], err error) {
	return layerNormForwardWithEngine(ctx, m.engine, x, scale, bias, rows, dModel)
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
// Forward and backward linear operations use engine.MatMul with batched samples
// concatenated into single large matrices, making ONE MatMul call per linear layer
// per channel regardless of batch size.
// Softmax, GELU, and layer norm run on CPU (small tensors, element-wise).
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

	channels := len(windows[0])

	result := &TrainResult{
		LossHistory: make([]float64, config.Epochs),
	}

	batchSize := nSamples
	if config.BatchSize > 0 && config.BatchSize < nSamples {
		batchSize = config.BatchSize
	}

	// Pre-allocate tensor workspace outside the training loop.
	// All batch dimensions are fixed after partial-batch dropping (T51.1.1),
	// so these buffers can be reused across every batch and epoch.
	// This eliminates per-batch allocations that would cause cudaMalloc
	// during CUDA graph capture (error 901).
	bsC := batchSize * channels          // batch dimension includes all channels
	totalRows := bsC * numPatches        // total rows for encoder
	chanScale := float32(1.0 / float64(channels))

	// Forward cache: pre-allocate struct and layer cache slice once.
	fc := gpuBatchForwardCache{
		layerCaches: make([]gpuBatchLayerCache, m.config.NLayers),
	}

	// Pre-allocate reusable slices for patch building, flatten, loss, and backward.
	patchData := make([]float32, totalRows*m.config.PatchLength)
	flatData := make([]float32, bsC*headIn)
	predData := make([]float32, batchSize*outDim)
	dPredData := make([]float32, batchSize*outDim)
	dChanOutData := make([]float32, bsC*outDim)

	// Pre-allocate tensors with fixed shapes that are overwritten each batch.
	fc.patches, err = tensor.New[float32]([]int{totalRows, m.config.PatchLength}, patchData)
	if err != nil {
		return nil, fmt.Errorf("gpu workspace patches: %w", err)
	}
	fc.flatInput, err = tensor.New[float32]([]int{bsC, headIn}, flatData)
	if err != nil {
		return nil, fmt.Errorf("gpu workspace flatInput: %w", err)
	}
	dChanOut, err := tensor.New[float32]([]int{bsC, outDim}, dChanOutData)
	if err != nil {
		return nil, fmt.Errorf("gpu workspace dChanOut: %w", err)
	}

	// Cache gradient tensor list once (slice of pointers is stable across batches).
	gradTs := grads.allParamTensors()

	// Drop partial final batch for consistent tensor shapes (required for CUDA graph capture).
	fullBatches := nSamples - (nSamples % batchSize)

	for epoch := 0; epoch < config.Epochs; epoch++ {
		epochLoss := 0.0
		nBatches := 0

		for start := 0; start < fullBatches; start += batchSize {
			bs := batchSize

			// Zero all gradients.
			for _, gt := range gradTs {
				if err := m.engine.Zero(ctx, gt); err != nil {
					return nil, fmt.Errorf("patchtst gpu: zero grad: %w", err)
				}
			}

			// Pre-compute weight transposes (used in backward, constant within batch).
			headWT, err := m.engine.Transpose(ctx, params.headW, []int{1, 0})
			if err != nil {
				return nil, err
			}
			// layerTransposes is defined at package level in patchtst_encoder.go.
			layerWTs := make([]layerTransposes, m.config.NLayers)
			for li := 0; li < m.config.NLayers; li++ {
				layer := &params.layers[li]
				lt := &layerWTs[li]
				lt.qWT, err = m.engine.Transpose(ctx, layer.qW, []int{1, 0})
				if err != nil {
					return nil, err
				}
				lt.kWT, err = m.engine.Transpose(ctx, layer.kW, []int{1, 0})
				if err != nil {
					return nil, err
				}
				lt.vWT, err = m.engine.Transpose(ctx, layer.vW, []int{1, 0})
				if err != nil {
					return nil, err
				}
				lt.oWT, err = m.engine.Transpose(ctx, layer.oW, []int{1, 0})
				if err != nil {
					return nil, err
				}
				lt.ffn1WT, err = m.engine.Transpose(ctx, layer.ffn1W, []int{1, 0})
				if err != nil {
					return nil, err
				}
				lt.ffn2WT, err = m.engine.Transpose(ctx, layer.ffn2W, []int{1, 0})
				if err != nil {
					return nil, err
				}
			}

			// Build ALL patches at once into pre-allocated buffer: [bsC*numPatches, patchLen].
			// Interleave: sample0-ch0, sample0-ch1, ..., sample0-chN, sample1-ch0, ...
			for s := 0; s < bs; s++ {
				sIdx := start + s
				for ch := 0; ch < channels; ch++ {
					for p := 0; p < numPatches; p++ {
						startP := p * m.config.Stride
						off := ((s*channels+ch)*numPatches + p) * m.config.PatchLength
						for j := 0; j < m.config.PatchLength; j++ {
							patchData[off+j] = float32(windows[sIdx][ch][startP+j])
						}
					}
				}
			}

			// Patch embedding: [bsC*numPatches, patchLen] @ [patchLen, dModel] = [bsC*numPatches, dModel].
			// ONE MatMul for all samples and channels.
			embedded, err := m.engine.MatMul(ctx, fc.patches, params.patchEmbW)
			if err != nil {
				return nil, fmt.Errorf("gpu fwd patch emb: %w", err)
			}
			// Add bias: [bsC*numPatches, dModel] + [1, dModel] (broadcast).
			embedded, err = m.engine.Add(ctx, embedded, params.patchEmbB)
			if err != nil {
				return nil, err
			}

			// Add positional embedding via broadcast:
			// embedded [bsC*numPatches, dModel] -> [bsC, numPatches, dModel]
			// posEmb [numPatches, dModel] -> [1, numPatches, dModel] (implicit broadcast)
			emb3d, err := m.engine.Reshape(ctx, embedded, []int{bsC, numPatches, dModel})
			if err != nil {
				return nil, err
			}
			posEmb3d, err := m.engine.Reshape(ctx, params.posEmb, []int{1, numPatches, dModel})
			if err != nil {
				return nil, err
			}
			emb3d, err = m.engine.Add(ctx, emb3d, posEmb3d)
			if err != nil {
				return nil, err
			}
			x, err := m.engine.Reshape(ctx, emb3d, []int{totalRows, dModel})
			if err != nil {
				return nil, err
			}

			// Encoder forward (one pass for all samples x channels).
			x, fc.layerCaches, err = encoderForward(ctx, m.engine, x, params.layers,
				bsC, numPatches, totalRows, dModel, nHeads, headDim, ffnDim)
			if err != nil {
				return nil, fmt.Errorf("gpu encoder fwd: %w", err)
			}

			// Flatten for output head: [bsC*numPatches, dModel] -> [bsC, headIn].
			// Overwrite pre-allocated flatData buffer (backing fc.flatInput).
			copy(flatData, x.Data())

			// Output head: ONE MatMul for all samples x channels.
			// [bsC, headIn] @ [headIn, outDim] = [bsC, outDim]
			headOut, err := m.engine.MatMul(ctx, fc.flatInput, params.headW)
			if err != nil {
				return nil, err
			}
			headOut, err = m.engine.Add(ctx, headOut, params.headB)
			if err != nil {
				return nil, err
			}

			// Average across channels: reshape [bsC, outDim] -> [bs, channels, outDim],
			// sum axis=1 -> [bs, outDim], scale by 1/channels.
			headOutData := headOut.Data()
			for i := range predData {
				predData[i] = 0
			}
			for s := 0; s < bs; s++ {
				for ch := 0; ch < channels; ch++ {
					row := s*channels + ch
					for j := 0; j < outDim; j++ {
						predData[s*outDim+j] += headOutData[row*outDim+j]
					}
				}
				for j := 0; j < outDim; j++ {
					predData[s*outDim+j] *= chanScale
				}
			}

			// Compute loss and dPred on CPU: pred [bs, outDim] vs labels [bs, outDim].
			batchLoss := 0.0
			for s := 0; s < bs; s++ {
				sIdx := start + s
				sampleLabels := labels[sIdx*outDim : (sIdx+1)*outDim]
				for j := 0; j < outDim; j++ {
					diff := float64(predData[s*outDim+j]) - sampleLabels[j]
					batchLoss += diff * diff
					dPredData[s*outDim+j] = 2.0 * (predData[s*outDim+j] - float32(sampleLabels[j])) / float32(bs*outDim)
				}
			}

			// --- Backward pass (single pass for all channels) ---
			// Broadcast dPred [bs, outDim] -> [bsC, outDim], scaled by 1/channels.
			// Overwrite pre-allocated dChanOutData buffer (backing dChanOut).
			for s := 0; s < bs; s++ {
				for ch := 0; ch < channels; ch++ {
					row := s*channels + ch
					for j := 0; j < outDim; j++ {
						dChanOutData[row*outDim+j] = dPredData[s*outDim+j] * chanScale
					}
				}
			}

			// Head backward: ONE MatMul for dW, ONE for dX.
			// dHeadW += flatInput^T @ dChanOut : [headIn, bsC] @ [bsC, outDim] = [headIn, outDim]
			flatInputT, err := m.engine.Transpose(ctx, fc.flatInput, []int{1, 0})
			if err != nil {
				return nil, err
			}
			dHW, err := m.engine.MatMul(ctx, flatInputT, dChanOut)
			if err != nil {
				return nil, err
			}
			grads.headW, err = m.engine.Add(ctx, grads.headW, dHW)
			if err != nil {
				return nil, err
			}
			// dHeadB += sum(dChanOut, axis=0) : [1, outDim]
			dHB, err := m.engine.Sum(ctx, dChanOut, 0, false)
			if err != nil {
				return nil, err
			}
			dHBR, err := m.engine.Reshape(ctx, dHB, []int{1, outDim})
			if err != nil {
				return nil, err
			}
			grads.headB, err = m.engine.Add(ctx, grads.headB, dHBR)
			if err != nil {
				return nil, err
			}

			// dFlat = dChanOut @ headW^T : [bsC, outDim] @ [outDim, headIn] = [bsC, headIn]
			dFlat, err := m.engine.MatMul(ctx, dChanOut, headWT)
			if err != nil {
				return nil, err
			}

			// Reshape dFlat from [bsC, headIn] to [bsC*numPatches, dModel].
			dX, err := m.engine.Reshape(ctx, dFlat, []int{totalRows, dModel})
			if err != nil {
				return nil, err
			}

			// Backward through encoder layers in reverse (single pass).
			dX, err = encoderBackward(ctx, m.engine, dX, params.layers, grads.layers,
				fc.layerCaches, layerWTs, bsC, numPatches, totalRows, dModel, nHeads, headDim, ffnDim)
			if err != nil {
				return nil, fmt.Errorf("gpu encoder bwd: %w", err)
			}

			// Positional embedding gradient: sum across all bsC samples.
			dPosData := grads.posEmb.Data()
			dXData := dX.Data()
			for s := 0; s < bsC; s++ {
				sOff := s * numPatches * dModel
				for j := 0; j < numPatches*dModel; j++ {
					dPosData[j] += dXData[sOff+j]
				}
			}

			// Patch embedding backward: ONE MatMul each.
			// dW += patches^T @ dX : [patchLen, totalRows] @ [totalRows, dModel]
			patchesT, err := m.engine.Transpose(ctx, fc.patches, []int{1, 0})
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

			batchLoss /= float64(bs * outDim)
			epochLoss += batchLoss
			nBatches++

			// Gradient clipping.
			if config.GradClip > 0 {
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
			t := float64(epoch*(fullBatches/batchSize) + nBatches)
			beta1 := float32(config.Beta1)
			beta2 := float32(config.Beta2)
			eps := float32(config.Epsilon)
			lrF := float32(lr)
			mCorr := float32(1.0 / (1.0 - math.Pow(config.Beta1, t)))
			vCorr := float32(1.0 / (1.0 - math.Pow(config.Beta2, t)))
			wdF := float32(config.WeightDecay)

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
