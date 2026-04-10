package timeseries

import (
	"context"
	"fmt"
	"math"

	"github.com/zerfoo/zerfoo/training/scheduler"
	"github.com/zerfoo/ztensor/compute"
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
	normed1      *tensor.TensorNumeric[float32] // [bs*numPatches, dModel]
	q, k, v      *tensor.TensorNumeric[float32] // [bs*numPatches, dModel]
	scoresTensor *tensor.TensorNumeric[float32] // [bs*nHeads, seq, seq] - batched attention scores
	attnOut      *tensor.TensorNumeric[float32] // [bs*numPatches, dModel]
	normed2      *tensor.TensorNumeric[float32] // [bs*numPatches, dModel]
	ffn1PreAct   *tensor.TensorNumeric[float32] // [bs*numPatches, ffnDim]
	ffn1Out      *tensor.TensorNumeric[float32] // [bs*numPatches, ffnDim]
	centered1    *tensor.TensorNumeric[float32] // [bs*numPatches, dModel]
	invStd1      *tensor.TensorNumeric[float32] // [bs*numPatches, 1]
	centered2    *tensor.TensorNumeric[float32] // [bs*numPatches, dModel]
	invStd2      *tensor.TensorNumeric[float32] // [bs*numPatches, 1]
	xResidual    *tensor.TensorNumeric[float32] // [bs*numPatches, dModel] input to layer
	xAfterAttn   *tensor.TensorNumeric[float32] // [bs*numPatches, dModel] after residual 1
	geluTanhVal  *tensor.TensorNumeric[float32] // [bs*numPatches, ffnDim] cached for backward

	// --- T85.2.4 pre-allocated encoder fwd/bwd scratch buffers ---
	// Allocated lazily on the first encoderForward call and reused for all
	// subsequent batches. All dimensions are fixed after partial-batch
	// dropping so these buffers can be written into in-place via the
	// dst-param variants of the engine ops.

	// Layer norm 1 (forward) intermediates.
	ln1Mean    *tensor.TensorNumeric[float32] // [totalRows, 1]
	ln1CentSq  *tensor.TensorNumeric[float32] // [totalRows, dModel]
	ln1Var     *tensor.TensorNumeric[float32] // [totalRows, 1]
	ln1VarEps  *tensor.TensorNumeric[float32] // [totalRows, 1]
	ln1Stddev  *tensor.TensorNumeric[float32] // [totalRows, 1]
	ln1Ones    *tensor.TensorNumeric[float32] // [totalRows, 1]
	ln1Scaled  *tensor.TensorNumeric[float32] // [totalRows, dModel] centered*invStd
	ln1NormMul *tensor.TensorNumeric[float32] // [totalRows, dModel] normed*scale pre-bias
	// Layer norm 2 (forward) intermediates.
	ln2Mean    *tensor.TensorNumeric[float32]
	ln2CentSq  *tensor.TensorNumeric[float32]
	ln2Var     *tensor.TensorNumeric[float32]
	ln2VarEps  *tensor.TensorNumeric[float32]
	ln2Stddev  *tensor.TensorNumeric[float32]
	ln2Ones    *tensor.TensorNumeric[float32]
	ln2Scaled  *tensor.TensorNumeric[float32]
	ln2NormMul *tensor.TensorNumeric[float32]

	// Attention forward intermediates (4D reshape/transpose + matmul scratch).
	qBiased *tensor.TensorNumeric[float32] // [totalRows, dModel] q after bias add
	kBiased *tensor.TensorNumeric[float32]
	vBiased *tensor.TensorNumeric[float32]

	q4d          *tensor.TensorNumeric[float32] // [bsC, seq, nHeads, headDim]
	k4d          *tensor.TensorNumeric[float32]
	v4d          *tensor.TensorNumeric[float32]
	q4dT         *tensor.TensorNumeric[float32] // [bsC, nHeads, seq, headDim]
	k4dT         *tensor.TensorNumeric[float32]
	v4dT         *tensor.TensorNumeric[float32]
	qH           *tensor.TensorNumeric[float32] // [bnh, seq, headDim]
	kH           *tensor.TensorNumeric[float32]
	vH           *tensor.TensorNumeric[float32]
	kHT          *tensor.TensorNumeric[float32] // [bnh, headDim, seq]
	logits       *tensor.TensorNumeric[float32] // [bnh, seq, seq]
	logitsScaled *tensor.TensorNumeric[float32]
	attnH        *tensor.TensorNumeric[float32] // [bnh, seq, headDim]
	attnH4d      *tensor.TensorNumeric[float32] // [bsC, nHeads, seq, headDim]
	attnH4dT     *tensor.TensorNumeric[float32] // [bsC, seq, nHeads, headDim]

	attnProj     *tensor.TensorNumeric[float32] // [totalRows, dModel]
	attnProjBias *tensor.TensorNumeric[float32]
	xAfterRes1   *tensor.TensorNumeric[float32] // [totalRows, dModel]

	// FFN forward intermediates.
	ffn1Matmul      *tensor.TensorNumeric[float32] // [totalRows, ffnDim]
	geluX3          *tensor.TensorNumeric[float32]
	geluInner1      *tensor.TensorNumeric[float32]
	geluInner2      *tensor.TensorNumeric[float32]
	geluInner3      *tensor.TensorNumeric[float32]
	geluOnePlusTanh *tensor.TensorNumeric[float32]
	geluXTimes      *tensor.TensorNumeric[float32]
	ffn2Matmul      *tensor.TensorNumeric[float32] // [totalRows, dModel]
	ffn2Out         *tensor.TensorNumeric[float32]
	xAfterRes2      *tensor.TensorNumeric[float32] // [totalRows, dModel] (layer output)

	// --- Backward scratch ---
	// FFN backward.
	ffn1OutT    *tensor.TensorNumeric[float32] // [ffnDim, totalRows]
	dFfn2W      *tensor.TensorNumeric[float32] // [ffnDim, dModel]
	dFfn2BSum   *tensor.TensorNumeric[float32] // [dModel]
	dFfn2BR     *tensor.TensorNumeric[float32] // [1, dModel]
	dFfn1Out    *tensor.TensorNumeric[float32] // [totalRows, ffnDim]
	gTerm1      *tensor.TensorNumeric[float32]
	gTanhSq     *tensor.TensorNumeric[float32]
	gSechSq     *tensor.TensorNumeric[float32]
	gXSq        *tensor.TensorNumeric[float32]
	gDudx       *tensor.TensorNumeric[float32]
	gTerm2      *tensor.TensorNumeric[float32]
	gDeriv      *tensor.TensorNumeric[float32]
	dFfn1PreAct *tensor.TensorNumeric[float32]
	normed2T    *tensor.TensorNumeric[float32] // [dModel, totalRows]
	dFfn1W      *tensor.TensorNumeric[float32] // [dModel, ffnDim]
	dFfn1BSum   *tensor.TensorNumeric[float32]
	dFfn1BR     *tensor.TensorNumeric[float32]
	dNormed2    *tensor.TensorNumeric[float32] // [totalRows, dModel]

	// LayerNorm2 backward.
	dAttnProjOut *tensor.TensorNumeric[float32]

	// Output proj backward.
	attnOutT *tensor.TensorNumeric[float32] // [dModel, totalRows]
	dOW      *tensor.TensorNumeric[float32]
	dOBSum   *tensor.TensorNumeric[float32]
	dOBR     *tensor.TensorNumeric[float32]
	dAttnOut *tensor.TensorNumeric[float32] // [totalRows, dModel]

	// Attention backward.
	dAO4d     *tensor.TensorNumeric[float32]
	dAO4dT    *tensor.TensorNumeric[float32]
	dAttnOutH *tensor.TensorNumeric[float32] // [bnh, seq, headDim]
	bwdQ4d    *tensor.TensorNumeric[float32]
	bwdQ4dT   *tensor.TensorNumeric[float32]
	bwdQH     *tensor.TensorNumeric[float32]
	bwdK4d    *tensor.TensorNumeric[float32]
	bwdK4dT   *tensor.TensorNumeric[float32]
	bwdKH     *tensor.TensorNumeric[float32]
	bwdV4d    *tensor.TensorNumeric[float32]
	bwdV4dT   *tensor.TensorNumeric[float32]
	bwdVH     *tensor.TensorNumeric[float32]
	vHT       *tensor.TensorNumeric[float32] // [bnh, headDim, seq]
	dScores   *tensor.TensorNumeric[float32] // [bnh, seq, seq]
	scoresT   *tensor.TensorNumeric[float32]
	dVH       *tensor.TensorNumeric[float32]
	sDScores  *tensor.TensorNumeric[float32]
	rowSum    *tensor.TensorNumeric[float32] // [bnh, seq, 1]
	dLogits1  *tensor.TensorNumeric[float32]
	dLogits2  *tensor.TensorNumeric[float32]
	dLogits   *tensor.TensorNumeric[float32]
	dQH       *tensor.TensorNumeric[float32]
	dLogitsT  *tensor.TensorNumeric[float32]
	dKH       *tensor.TensorNumeric[float32]
	dQH4d     *tensor.TensorNumeric[float32]
	dQH4dT    *tensor.TensorNumeric[float32]
	dQT       *tensor.TensorNumeric[float32] // [totalRows, dModel]
	dKH4d     *tensor.TensorNumeric[float32]
	dKH4dT    *tensor.TensorNumeric[float32]
	dKT       *tensor.TensorNumeric[float32]
	dVH4d     *tensor.TensorNumeric[float32]
	dVH4dT    *tensor.TensorNumeric[float32]
	dVT       *tensor.TensorNumeric[float32]

	// Q/K/V projection backward.
	normed1T *tensor.TensorNumeric[float32]
	dQW      *tensor.TensorNumeric[float32]
	dQBSum   *tensor.TensorNumeric[float32]
	dQBR     *tensor.TensorNumeric[float32]
	dKW      *tensor.TensorNumeric[float32]
	dKBSum   *tensor.TensorNumeric[float32]
	dKBR     *tensor.TensorNumeric[float32]
	dVW      *tensor.TensorNumeric[float32]
	dVBSum   *tensor.TensorNumeric[float32]
	dVBR     *tensor.TensorNumeric[float32]
	dN1q     *tensor.TensorNumeric[float32]
	dN1k     *tensor.TensorNumeric[float32]
	dN1v     *tensor.TensorNumeric[float32]
	dN1Sum1  *tensor.TensorNumeric[float32]
	dNormed1 *tensor.TensorNumeric[float32]

	// LayerNorm backward scratch (shared between ln1/ln2 backward calls).
	lnbNormVal      *tensor.TensorNumeric[float32] // [totalRows, dModel]
	lnbDScaleBatch  *tensor.TensorNumeric[float32] // [totalRows, dModel]
	lnbDScaleSum    *tensor.TensorNumeric[float32] // [dModel]
	lnbDScaleSumR   *tensor.TensorNumeric[float32] // [1, dModel]
	lnbDBiasSum     *tensor.TensorNumeric[float32]
	lnbDBiasSumR    *tensor.TensorNumeric[float32]
	lnbDNorm        *tensor.TensorNumeric[float32]
	lnbDNormCent    *tensor.TensorNumeric[float32]
	lnbDotScaleGrad *tensor.TensorNumeric[float32]
	lnbDotMeanGrad  *tensor.TensorNumeric[float32]
	lnbInvStdSq     *tensor.TensorNumeric[float32]
	lnbTerm         *tensor.TensorNumeric[float32]
	lnbCorrection   *tensor.TensorNumeric[float32]
	lnbInner        *tensor.TensorNumeric[float32]
	lnbDInput       *tensor.TensorNumeric[float32]

	// Second set for ln2 backward (since both ln1 and ln2 backward run in the
	// same iteration and need separate buffers).
	ln2bNormVal      *tensor.TensorNumeric[float32]
	ln2bDScaleBatch  *tensor.TensorNumeric[float32]
	ln2bDScaleSum    *tensor.TensorNumeric[float32]
	ln2bDScaleSumR   *tensor.TensorNumeric[float32]
	ln2bDBiasSum     *tensor.TensorNumeric[float32]
	ln2bDBiasSumR    *tensor.TensorNumeric[float32]
	ln2bDNorm        *tensor.TensorNumeric[float32]
	ln2bDNormCent    *tensor.TensorNumeric[float32]
	ln2bDotScaleGrad *tensor.TensorNumeric[float32]
	ln2bDotMeanGrad  *tensor.TensorNumeric[float32]
	ln2bInvStdSq     *tensor.TensorNumeric[float32]
	ln2bTerm         *tensor.TensorNumeric[float32]
	ln2bCorrection   *tensor.TensorNumeric[float32]
	ln2bInner        *tensor.TensorNumeric[float32]
	ln2bDInput       *tensor.TensorNumeric[float32]

	// dXOut is the gradient w.r.t. this layer's input, computed by
	// encoderBackward and consumed as the dX input by the previous layer's
	// backward iteration. [totalRows, dModel]
	dXOut *tensor.TensorNumeric[float32]

	// buffersAllocated is set true after the one-time lazy allocation.
	buffersAllocated bool
}

// gpuBatchForwardCache stores batched forward data across all channels.
// All tensors are pre-allocated ONCE before the epoch loop and reused every
// batch via engine dst-param variants to avoid per-batch cudaMalloc (E85).
type gpuBatchForwardCache struct {
	patches     *tensor.TensorNumeric[float32] // [bsC*numPatches, patchLen]
	flatInput   *tensor.TensorNumeric[float32] // [bsC, headIn]
	layerCaches []gpuBatchLayerCache
	fusedGPU    fusedEncoderGPU // persistent GPU buffers for fused encoder path

	// Pre-allocated weight-transpose buffers (T85.2.1).
	headWT   *tensor.TensorNumeric[float32] // [outDim, headIn]
	layerWTs []layerTransposes              // per-layer qWT/kWT/vWT/oWT/ffn1WT/ffn2WT

	// Pre-allocated forward-prefix buffers (T85.2.2).
	embedded *tensor.TensorNumeric[float32] // [bsC*numPatches, dModel]
	emb3d    *tensor.TensorNumeric[float32] // [bsC, numPatches, dModel]
	posEmb3d *tensor.TensorNumeric[float32] // [1, numPatches, dModel]
	xForward *tensor.TensorNumeric[float32] // [totalRows, dModel]
	headOut  *tensor.TensorNumeric[float32] // [bsC, outDim]

	// Pre-allocated backward intermediates (T85.2.3).
	flatInputT *tensor.TensorNumeric[float32] // [headIn, bsC]
	dHW        *tensor.TensorNumeric[float32] // [headIn, outDim]
	dHB        *tensor.TensorNumeric[float32] // [outDim]
	dHBR       *tensor.TensorNumeric[float32] // [1, outDim]
	dFlat      *tensor.TensorNumeric[float32] // [bsC, headIn]
	dX         *tensor.TensorNumeric[float32] // [totalRows, dModel]
	patchesT   *tensor.TensorNumeric[float32] // [patchLen, totalRows]
	dPEW       *tensor.TensorNumeric[float32] // [patchLen, dModel]
	dPEB       *tensor.TensorNumeric[float32] // [dModel]
	dPEBR      *tensor.TensorNumeric[float32] // [1, dModel]
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
	bsC := batchSize * channels   // batch dimension includes all channels
	totalRows := bsC * numPatches // total rows for encoder
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

	// --- E85 T85.2.1: pre-allocate transpose buffers ------------------------
	// headW is [headIn, outDim]; headWT is [outDim, headIn].
	fc.headWT, err = tensor.New[float32]([]int{outDim, headIn}, make([]float32, outDim*headIn))
	if err != nil {
		return nil, fmt.Errorf("gpu workspace headWT: %w", err)
	}
	fc.layerWTs = make([]layerTransposes, m.config.NLayers)
	for li := 0; li < m.config.NLayers; li++ {
		lt := &fc.layerWTs[li]
		// qW/kW/vW/oW are [dModel, dModel]; transpose preserves shape.
		lt.qWT, err = tensor.New[float32]([]int{dModel, dModel}, make([]float32, dModel*dModel))
		if err != nil {
			return nil, fmt.Errorf("gpu workspace qWT: %w", err)
		}
		lt.kWT, err = tensor.New[float32]([]int{dModel, dModel}, make([]float32, dModel*dModel))
		if err != nil {
			return nil, fmt.Errorf("gpu workspace kWT: %w", err)
		}
		lt.vWT, err = tensor.New[float32]([]int{dModel, dModel}, make([]float32, dModel*dModel))
		if err != nil {
			return nil, fmt.Errorf("gpu workspace vWT: %w", err)
		}
		lt.oWT, err = tensor.New[float32]([]int{dModel, dModel}, make([]float32, dModel*dModel))
		if err != nil {
			return nil, fmt.Errorf("gpu workspace oWT: %w", err)
		}
		// ffn1W is [dModel, ffnDim]; ffn1WT is [ffnDim, dModel].
		lt.ffn1WT, err = tensor.New[float32]([]int{ffnDim, dModel}, make([]float32, ffnDim*dModel))
		if err != nil {
			return nil, fmt.Errorf("gpu workspace ffn1WT: %w", err)
		}
		// ffn2W is [ffnDim, dModel]; ffn2WT is [dModel, ffnDim].
		lt.ffn2WT, err = tensor.New[float32]([]int{dModel, ffnDim}, make([]float32, dModel*ffnDim))
		if err != nil {
			return nil, fmt.Errorf("gpu workspace ffn2WT: %w", err)
		}
	}

	// --- E85 T85.2.2: pre-allocate forward-prefix buffers --------------------
	fc.embedded, err = tensor.New[float32]([]int{bsC * numPatches, dModel}, make([]float32, bsC*numPatches*dModel))
	if err != nil {
		return nil, fmt.Errorf("gpu workspace embedded: %w", err)
	}
	fc.emb3d, err = tensor.New[float32]([]int{bsC, numPatches, dModel}, make([]float32, bsC*numPatches*dModel))
	if err != nil {
		return nil, fmt.Errorf("gpu workspace emb3d: %w", err)
	}
	fc.posEmb3d, err = tensor.New[float32]([]int{1, numPatches, dModel}, make([]float32, numPatches*dModel))
	if err != nil {
		return nil, fmt.Errorf("gpu workspace posEmb3d: %w", err)
	}
	fc.xForward, err = tensor.New[float32]([]int{totalRows, dModel}, make([]float32, totalRows*dModel))
	if err != nil {
		return nil, fmt.Errorf("gpu workspace xForward: %w", err)
	}
	fc.headOut, err = tensor.New[float32]([]int{bsC, outDim}, make([]float32, bsC*outDim))
	if err != nil {
		return nil, fmt.Errorf("gpu workspace headOut: %w", err)
	}

	// --- E85 T85.2.3: pre-allocate backward intermediate buffers -------------
	fc.flatInputT, err = tensor.New[float32]([]int{headIn, bsC}, make([]float32, headIn*bsC))
	if err != nil {
		return nil, fmt.Errorf("gpu workspace flatInputT: %w", err)
	}
	fc.dHW, err = tensor.New[float32]([]int{headIn, outDim}, make([]float32, headIn*outDim))
	if err != nil {
		return nil, fmt.Errorf("gpu workspace dHW: %w", err)
	}
	// Sum over axis=0 with keepDims=false collapses that axis, producing shape [outDim].
	fc.dHB, err = tensor.New[float32]([]int{outDim}, make([]float32, outDim))
	if err != nil {
		return nil, fmt.Errorf("gpu workspace dHB: %w", err)
	}
	fc.dHBR, err = tensor.New[float32]([]int{1, outDim}, make([]float32, outDim))
	if err != nil {
		return nil, fmt.Errorf("gpu workspace dHBR: %w", err)
	}
	fc.dFlat, err = tensor.New[float32]([]int{bsC, headIn}, make([]float32, bsC*headIn))
	if err != nil {
		return nil, fmt.Errorf("gpu workspace dFlat: %w", err)
	}
	fc.dX, err = tensor.New[float32]([]int{totalRows, dModel}, make([]float32, totalRows*dModel))
	if err != nil {
		return nil, fmt.Errorf("gpu workspace dX: %w", err)
	}
	fc.patchesT, err = tensor.New[float32]([]int{m.config.PatchLength, totalRows}, make([]float32, m.config.PatchLength*totalRows))
	if err != nil {
		return nil, fmt.Errorf("gpu workspace patchesT: %w", err)
	}
	fc.dPEW, err = tensor.New[float32]([]int{m.config.PatchLength, dModel}, make([]float32, m.config.PatchLength*dModel))
	if err != nil {
		return nil, fmt.Errorf("gpu workspace dPEW: %w", err)
	}
	fc.dPEB, err = tensor.New[float32]([]int{dModel}, make([]float32, dModel))
	if err != nil {
		return nil, fmt.Errorf("gpu workspace dPEB: %w", err)
	}
	fc.dPEBR, err = tensor.New[float32]([]int{1, dModel}, make([]float32, dModel))
	if err != nil {
		return nil, fmt.Errorf("gpu workspace dPEBR: %w", err)
	}

	// paramTs and gradTs are (re)built per batch inside the loop below.
	// See plan T1.2 and .claude/scratch/{e1-gradts-sites,t3.1b-grad-check-result}.md:
	// params.X / grads.X may be reassigned to different backing storage during
	// forward/backward; a once-per-training snapshot pointed at stale buffers,
	// so AdamW read zero gradients and weights never updated.
	var gradTs []*tensor.TensorNumeric[float32]

	// Drop partial final batch for consistent tensor shapes (required for CUDA graph capture).
	fullBatches := nSamples - (nSamples % batchSize)

	// CUDA graph capture state for the forward-prefix block.
	// The forward prefix (zero grads + weight transposes + patch embed + pos embed)
	// is a contiguous sequence of engine ops with no .Data() calls, making it safe
	// to capture into a CUDA graph. On replay, all recorded kernels execute in a
	// single GPU submission with zero intermediate synchronisation.
	//
	// Batch 0 = warmup (normal execution, allocates output buffers on GPU).
	// Batch 1 = capture (BeginCapture, run ops, EndCapture).
	// Batch 2+ = replay (ReplayGraph reuses same GPU memory addresses).
	// ztensor's GPU memory pool is now capture-aware (ztensor PR #48):
	// during BeginCapture, the pool switches to cudaMallocAsync on the
	// capture stream, so allocations are recorded as graph nodes.
	// Forward-prefix capture (~78 ops) is slower than no capture (20.9s vs
	// 12.9s/epoch) because the graph is too small for replay savings to
	// exceed launch+sync cost. Disabled until the full ~500-op encoder
	// can be captured (requires fused encoder kernel -- see E55).
	gc, canCapture := m.engine.(compute.GraphCapturer)
	_ = gc
	canCapture = false
	var fwdGraph compute.GraphHandle
	fwdCaptured := false
	batchIter := 0

	for epoch := 0; epoch < config.Epochs; epoch++ {
		epochLoss := 0.0
		nBatches := 0

		for start := 0; start < fullBatches; start += batchSize {
			bs := batchSize

			// Rebuild paramTs/gradTs from the live struct fields each batch.
			// A once-per-training snapshot went stale whenever params.X / grads.X
			// were reassigned, causing AdamW to read zero-filled buffers and
			// freezing PatchTST GPU training loss at 0.268357 (plan T1.2).
			paramTs = params.allParamTensors()
			gradTs = grads.allParamTensors()

			// Build ALL patches at once into pre-allocated buffer: [bsC*numPatches, patchLen].
			// Interleave: sample0-ch0, sample0-ch1, ..., sample0-chN, sample1-ch0, ...
			// This writes to patchData which backs fc.patches. Must happen BEFORE
			// graph replay so the captured kernels read updated input data.
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

			// ----- Forward-prefix: zero grads + transposes + embed -----
			// This block is captured into a CUDA graph on batchIter==1 and
			// replayed for all subsequent batches.
			if fwdCaptured {
				// Replay: re-executes all captured GPU kernels in one shot.
				// Input tensors (params, gradTs, fc.patches) are at fixed addresses;
				// AdamW and patch-fill updated their contents in-place so the
				// replayed kernels see fresh data.
				if err := gc.ReplayGraph(fwdGraph); err != nil {
					return nil, fmt.Errorf("patchtst gpu: replay fwd graph: %w", err)
				}
			} else {
				// batchIter==1: begin capture (batchIter==0 was the warmup).
				if canCapture && batchIter == 1 {
					if err := gc.BeginCapture(); err != nil {
						return nil, fmt.Errorf("patchtst gpu: begin capture: %w", err)
					}
				}

				// Zero all gradients.
				for _, gt := range gradTs {
					if err := m.engine.Zero(ctx, gt); err != nil {
						return nil, fmt.Errorf("patchtst gpu: zero grad: %w", err)
					}
				}

				// Pre-compute weight transposes (used in backward, constant within batch).
				// Writes into pre-allocated fc.headWT / fc.layerWTs buffers (E85 T85.2.1).
				if _, err = m.engine.Transpose(ctx, params.headW, []int{1, 0}, fc.headWT); err != nil {
					return nil, err
				}
				for li := 0; li < m.config.NLayers; li++ {
					layer := &params.layers[li]
					lt := &fc.layerWTs[li]
					if _, err = m.engine.Transpose(ctx, layer.qW, []int{1, 0}, lt.qWT); err != nil {
						return nil, err
					}
					if _, err = m.engine.Transpose(ctx, layer.kW, []int{1, 0}, lt.kWT); err != nil {
						return nil, err
					}
					if _, err = m.engine.Transpose(ctx, layer.vW, []int{1, 0}, lt.vWT); err != nil {
						return nil, err
					}
					if _, err = m.engine.Transpose(ctx, layer.oW, []int{1, 0}, lt.oWT); err != nil {
						return nil, err
					}
					if _, err = m.engine.Transpose(ctx, layer.ffn1W, []int{1, 0}, lt.ffn1WT); err != nil {
						return nil, err
					}
					if _, err = m.engine.Transpose(ctx, layer.ffn2W, []int{1, 0}, lt.ffn2WT); err != nil {
						return nil, err
					}
				}

				// Patch embedding: [bsC*numPatches, patchLen] @ [patchLen, dModel] = [bsC*numPatches, dModel].
				// Pre-zero fc.embedded because the CPU SGEMM kernel accumulates into C (T85.2.2).
				if err = m.engine.Zero(ctx, fc.embedded); err != nil {
					return nil, err
				}
				if _, err = m.engine.MatMul(ctx, fc.patches, params.patchEmbW, fc.embedded); err != nil {
					return nil, fmt.Errorf("gpu fwd patch emb: %w", err)
				}
				if _, err = m.engine.Add(ctx, fc.embedded, params.patchEmbB, fc.embedded); err != nil {
					return nil, err
				}
				if _, err = m.engine.Reshape(ctx, fc.embedded, []int{bsC, numPatches, dModel}, fc.emb3d); err != nil {
					return nil, err
				}
				if _, err = m.engine.Reshape(ctx, params.posEmb, []int{1, numPatches, dModel}, fc.posEmb3d); err != nil {
					return nil, err
				}
				if _, err = m.engine.Add(ctx, fc.emb3d, fc.posEmb3d, fc.emb3d); err != nil {
					return nil, err
				}
				if _, err = m.engine.Reshape(ctx, fc.emb3d, []int{totalRows, dModel}, fc.xForward); err != nil {
					return nil, err
				}

				// End capture on batchIter==1.
				if canCapture && batchIter == 1 {
					fwdGraph, err = gc.EndCapture()
					if err != nil {
						return nil, fmt.Errorf("patchtst gpu: end capture: %w", err)
					}
					fwdCaptured = true
				}
			}

			// Forward-prefix outputs live in pre-allocated fc buffers. On graph
			// replay, the captured kernels write into the same GPU memory.
			x := fc.xForward
			batchIter++

			// Encoder forward (one pass for all samples x channels).
			// fc.layerCaches is pre-allocated and persistent across batches so
			// that encoder scratch buffers are reused (T85.2.4).
			x, err = encoderForward(ctx, m.engine, x, params.layers, fc.layerCaches, &fc.fusedGPU,
				bsC, numPatches, totalRows, dModel, nHeads, headDim, ffnDim)
			if err != nil {
				return nil, fmt.Errorf("gpu encoder fwd: %w", err)
			}

			// Flatten for output head: [bsC*numPatches, dModel] -> [bsC, headIn].
			// Overwrite pre-allocated flatData buffer (backing fc.flatInput).
			copy(flatData, x.Data())

			// Output head: ONE MatMul for all samples x channels. Writes fc.headOut.
			// [bsC, headIn] @ [headIn, outDim] = [bsC, outDim]
			// NOTE: the CPU SGEMM kernel accumulates into C; pre-zero fc.headOut
			// so we overwrite rather than accumulate stale data from the previous batch.
			if err = m.engine.Zero(ctx, fc.headOut); err != nil {
				return nil, err
			}
			if _, err = m.engine.MatMul(ctx, fc.flatInput, params.headW, fc.headOut); err != nil {
				return nil, err
			}
			if _, err = m.engine.Add(ctx, fc.headOut, params.headB, fc.headOut); err != nil {
				return nil, err
			}

			// Average across channels: reshape [bsC, outDim] -> [bs, channels, outDim],
			// sum axis=1 -> [bs, outDim], scale by 1/channels.
			headOutData := fc.headOut.Data()
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

			// Head backward: all intermediates pre-allocated in fc (E85 T85.2.3).
			// Gradients accumulate in-place via Add(a, b, a) so grads.headW and
			// grads.headB keep pointer identity with gradTs (T85.2.5).
			// NOTE: SGEMM accumulates into its C; dst tensors are zeroed before MatMul.
			//
			// dHeadW += flatInput^T @ dChanOut : [headIn, bsC] @ [bsC, outDim] = [headIn, outDim]
			if _, err = m.engine.Transpose(ctx, fc.flatInput, []int{1, 0}, fc.flatInputT); err != nil {
				return nil, err
			}
			if err = m.engine.Zero(ctx, fc.dHW); err != nil {
				return nil, err
			}
			if _, err = m.engine.MatMul(ctx, fc.flatInputT, dChanOut, fc.dHW); err != nil {
				return nil, err
			}
			if _, err = m.engine.Add(ctx, grads.headW, fc.dHW, grads.headW); err != nil {
				return nil, err
			}
			// dHeadB += sum(dChanOut, axis=0) : [outDim] -> [1, outDim]
			if _, err = m.engine.Sum(ctx, dChanOut, 0, false, fc.dHB); err != nil {
				return nil, err
			}
			if _, err = m.engine.Reshape(ctx, fc.dHB, []int{1, outDim}, fc.dHBR); err != nil {
				return nil, err
			}
			if _, err = m.engine.Add(ctx, grads.headB, fc.dHBR, grads.headB); err != nil {
				return nil, err
			}

			// dFlat = dChanOut @ headW^T : [bsC, outDim] @ [outDim, headIn] = [bsC, headIn]
			if err = m.engine.Zero(ctx, fc.dFlat); err != nil {
				return nil, err
			}
			if _, err = m.engine.MatMul(ctx, dChanOut, fc.headWT, fc.dFlat); err != nil {
				return nil, err
			}

			// Reshape dFlat from [bsC, headIn] to [totalRows, dModel].
			// NOTE: ztensor GPUEngine.Reshape ignores the dst arg for the
			// zero-copy GPUStorage fast-path — it returns a fresh tensor
			// aliasing fc.dFlat's storage. Callers MUST use the return
			// value; `fc.dX` here is kept only as a pre-alloc slot for its
			// shape metadata but its backing storage is stale zeros. See
			// docs/devlog.md 2026-04-09 Wave 7 root cause entry.
			dXReshaped, err := m.engine.Reshape(ctx, fc.dFlat, []int{totalRows, dModel}, fc.dX)
			if err != nil {
				return nil, err
			}

			// Backward through encoder layers in reverse (single pass).
			// encoderBackward may return a different tensor than its input; use its return value.
			dX, err := encoderBackward(ctx, m.engine, dXReshaped, params.layers, grads.layers,
				fc.layerCaches, fc.layerWTs, bsC, numPatches, totalRows, dModel, nHeads, headDim, ffnDim)
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

			// Patch embedding backward: pre-allocated buffers (E85 T85.2.3).
			if _, err = m.engine.Transpose(ctx, fc.patches, []int{1, 0}, fc.patchesT); err != nil {
				return nil, err
			}
			if err = m.engine.Zero(ctx, fc.dPEW); err != nil {
				return nil, err
			}
			if _, err = m.engine.MatMul(ctx, fc.patchesT, dX, fc.dPEW); err != nil {
				return nil, err
			}
			if _, err = m.engine.Add(ctx, grads.patchEmbW, fc.dPEW, grads.patchEmbW); err != nil {
				return nil, err
			}
			if _, err = m.engine.Sum(ctx, dX, 0, false, fc.dPEB); err != nil {
				return nil, err
			}
			if _, err = m.engine.Reshape(ctx, fc.dPEB, []int{1, dModel}, fc.dPEBR); err != nil {
				return nil, err
			}
			if _, err = m.engine.Add(ctx, grads.patchEmbB, fc.dPEBR, grads.patchEmbB); err != nil {
				return nil, err
			}

			// T2.1 (v2 plan): strengthened sentinel. Compares data-slice base
			// pointers via unsafe.Pointer so regressions of the gradTs staleness
			// class (T3.1b finding: struct-wrapper identity holds while Data()
			// arenas diverge) panic loudly at first batch. The dead struct-
			// pointer checks were removed in T1.2 now that T1.2 rebuilds gradTs
			// per batch. See .claude/scratch/e1-gradts-sites.md and
			// .claude/scratch/t3.1b-grad-check-result.md.
			if epoch == 0 && nBatches == 0 {
				verifyGradTsAliasing(grads, gradTs)
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

			// AdamW update. Step count is 1-indexed so the bias-correction terms
			// (1 - beta^t) are nonzero on the first step. Using t=0 would produce
			// mCorr/vCorr = +Inf and NaN updates. This was a latent bug masked by
			// stale gradTs pointers that kept real gradients out of AdamW (E85 T85.2.5).
			lr := scheduler.WarmupLR(config.LR, epoch, config.WarmupEpochs)
			t := float64(epoch*(fullBatches/batchSize)+nBatches) + 1
			beta1 := float32(config.Beta1)
			beta2 := float32(config.Beta2)
			eps := float32(config.Epsilon)
			lrF := float32(lr)
			mCorr := float32(1.0 / (1.0 - math.Pow(config.Beta1, t)))
			vCorr := float32(1.0 / (1.0 - math.Pow(config.Beta2, t)))
			wdF := float32(config.WeightDecay)

			for i := range paramTs {
				// AdamW step on CPU for simplicity and correctness. Since T1.2
				// rebuilds paramTs and gradTs from the live struct fields each
				// batch, paramTs[i].Data() and gradTs[i].Data() return the
				// actual live storage — direct in-place updates propagate to
				// subsequent forward passes without a SetData round trip.
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

	// Release captured CUDA graph resources.
	if fwdCaptured {
		gc.DestroyGraph(fwdGraph)
	}

	// Write optimized params back to model tensors.
	m.writeBackF32FromGPU(params)

	result.Metrics = map[string]float64{"mse": result.FinalLoss}
	return result, nil
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
