package timeseries

import (
	"context"
	"fmt"
	"math"

	"github.com/zerfoo/zerfoo/layers/functional"
	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
)

// layerTransposes holds pre-computed weight transposes for a single encoder layer.
// These are constant within a batch and shared between forward and backward.
type layerTransposes struct {
	qWT, kWT, vWT, oWT *tensor.TensorNumeric[float32]
	ffn1WT, ffn2WT     *tensor.TensorNumeric[float32]
}

// matMulInto computes dst = a @ b, zeroing dst first because the CPU SGEMM
// backend (xblas.SgemmSimd) accumulates into C rather than overwriting it.
// Without the zero, reusing a buffer across batches would produce (prev + new).
func matMulInto(
	ctx context.Context,
	engine compute.Engine[float32],
	a, b, dst *tensor.TensorNumeric[float32],
) error {
	if err := engine.Zero(ctx, dst); err != nil {
		return err
	}
	_, err := engine.MatMul(ctx, a, b, dst)
	return err
}

// layerNormForwardWithEngine performs layer norm using engine ops.
// x: [rows, dModel], scale: [1, dModel], bias: [1, dModel].
// Writes into pre-allocated buffers (normedBuf, centeredBuf, invStdBuf) plus
// scratch buffers (mean, centSq, variance, varEps, stddev, ones, scaledBuf,
// normMulBuf). All buffers must have the correct shapes pre-allocated.
func layerNormForwardWithEngine(
	ctx context.Context,
	engine compute.Engine[float32],
	x, scale, bias *tensor.TensorNumeric[float32],
	normedBuf, centeredBuf, invStdBuf *tensor.TensorNumeric[float32],
	meanBuf, centSqBuf, varBuf, varEpsBuf, stddevBuf, onesBuf, scaledBuf, normMulBuf *tensor.TensorNumeric[float32],
	rows, dModel int,
) (normed, centered, invStd *tensor.TensorNumeric[float32], err error) {
	invD := float32(1.0) / float32(dModel)

	// mean = Sum(x, axis=1, keepDims=true) * (1/dModel)  -> [rows, 1]
	mean, err := engine.Sum(ctx, x, 1, true, meanBuf)
	if err != nil {
		return nil, nil, nil, err
	}
	mean, err = engine.MulScalar(ctx, mean, invD, meanBuf)
	if err != nil {
		return nil, nil, nil, err
	}

	// centered = x - mean  (broadcast [rows,1] over [rows, dModel])
	centered, err = engine.Sub(ctx, x, mean, centeredBuf)
	if err != nil {
		return nil, nil, nil, err
	}

	// variance = Sum(centered*centered, axis=1, keepDims=true) * (1/dModel)  -> [rows, 1]
	centSq, err := engine.Mul(ctx, centered, centered, centSqBuf)
	if err != nil {
		return nil, nil, nil, err
	}
	variance, err := engine.Sum(ctx, centSq, 1, true, varBuf)
	if err != nil {
		return nil, nil, nil, err
	}
	variance, err = engine.MulScalar(ctx, variance, invD, varBuf)
	if err != nil {
		return nil, nil, nil, err
	}

	// invStd = 1 / sqrt(variance + eps)  -> [rows, 1]
	varEps, err := engine.AddScalar(ctx, variance, 1e-5, varEpsBuf)
	if err != nil {
		return nil, nil, nil, err
	}
	stddev, err := engine.Sqrt(ctx, varEps, stddevBuf)
	if err != nil {
		return nil, nil, nil, err
	}
	invStd, err = engine.Div(ctx, onesBuf, stddev, invStdBuf)
	if err != nil {
		return nil, nil, nil, err
	}

	// normed = centered * invStd * scale + bias  (broadcast [rows,1] and [1,dModel])
	scaled, err := engine.Mul(ctx, centered, invStd, scaledBuf)
	if err != nil {
		return nil, nil, nil, err
	}
	normMul, err := engine.Mul(ctx, scaled, scale, normMulBuf)
	if err != nil {
		return nil, nil, nil, err
	}
	normed, err = engine.Add(ctx, normMul, bias, normedBuf)
	if err != nil {
		return nil, nil, nil, err
	}
	_ = rows

	return normed, centered, invStd, nil
}

// lnBwdBufs groups scratch buffers used by layerNormBackwardWithEngine so
// that callers (encoderBackward) can supply distinct buffer sets for ln1 and
// ln2 backward within the same iteration.
type lnBwdBufs struct {
	normVal, dScaleBatch                      *tensor.TensorNumeric[float32]
	dScaleSum, dScaleSumR                     *tensor.TensorNumeric[float32]
	dBiasSum, dBiasSumR                       *tensor.TensorNumeric[float32]
	dNorm, dNormCent                          *tensor.TensorNumeric[float32]
	dotScaleGrad, dotMeanGrad                 *tensor.TensorNumeric[float32]
	invStdSq, term, correction, inner, dInput *tensor.TensorNumeric[float32]
}

// layerNormBackwardWithEngine computes the backward pass through layer norm using engine ops.
// dOut: [rows, dModel], centered: [rows, dModel], invStd: [rows, 1].
// scale: [1, dModel] (layer norm weight), dScale/dBias: [1, dModel] (gradient accumulators).
// Writes into pre-allocated scratch buffers in bufs. The returned newDScale
// and newDBias ARE bufs.dScaleAcc and bufs.dBiasAcc respectively; the caller
// is expected to copy these into its gradient accumulators (dg.norm*, dg.bias*)
// via a subsequent engine.Add to match the pre-fix semantics exactly.
func layerNormBackwardWithEngine(
	ctx context.Context,
	engine compute.Engine[float32],
	dOut, centered, invStd, scale, dScale, dBias *tensor.TensorNumeric[float32],
	bufs *lnBwdBufs,
	rows, dModel int,
) (dInput, newDScale, newDBias *tensor.TensorNumeric[float32], err error) {
	_ = rows

	// dScale += Sum(dOut * centered * invStd, axis=0)  -> [1, dModel]
	normVal, err := engine.Mul(ctx, centered, invStd, bufs.normVal) // [rows, dModel]
	if err != nil {
		return nil, nil, nil, err
	}
	dScaleBatch, err := engine.Mul(ctx, dOut, normVal, bufs.dScaleBatch) // [rows, dModel]
	if err != nil {
		return nil, nil, nil, err
	}
	dScaleSum, err := engine.Sum(ctx, dScaleBatch, 0, false, bufs.dScaleSum) // [dModel]
	if err != nil {
		return nil, nil, nil, err
	}
	dScaleSumR, err := engine.Reshape(ctx, dScaleSum, []int{1, dModel}, bufs.dScaleSumR)
	if err != nil {
		return nil, nil, nil, err
	}
	// Accumulate in-place into the caller's dScale tensor.
	newDScale, err = engine.Add(ctx, dScale, dScaleSumR, dScale)
	if err != nil {
		return nil, nil, nil, err
	}

	// dBias += Sum(dOut, axis=0)  -> [1, dModel]
	dBiasSum, err := engine.Sum(ctx, dOut, 0, false, bufs.dBiasSum) // [dModel]
	if err != nil {
		return nil, nil, nil, err
	}
	dBiasSumR, err := engine.Reshape(ctx, dBiasSum, []int{1, dModel}, bufs.dBiasSumR)
	if err != nil {
		return nil, nil, nil, err
	}
	// Accumulate in-place into the caller's dBias tensor.
	newDBias, err = engine.Add(ctx, dBias, dBiasSumR, dBias)
	if err != nil {
		return nil, nil, nil, err
	}

	// dNorm = dOut * scale  (broadcast [1, dModel] over [rows, dModel])
	dNorm, err := engine.Mul(ctx, dOut, scale, bufs.dNorm) // [rows, dModel]
	if err != nil {
		return nil, nil, nil, err
	}

	// dotScaleGrad = Sum(dNorm * centered, axis=1, keepDims=true)  -> [rows, 1]
	dNormCent, err := engine.Mul(ctx, dNorm, centered, bufs.dNormCent) // [rows, dModel]
	if err != nil {
		return nil, nil, nil, err
	}
	dotScaleGrad, err := engine.Sum(ctx, dNormCent, 1, true, bufs.dotScaleGrad) // [rows, 1]
	if err != nil {
		return nil, nil, nil, err
	}

	// dotMeanGrad = Sum(dNorm, axis=1, keepDims=true)  -> [rows, 1]
	dotMeanGrad, err := engine.Sum(ctx, dNorm, 1, true, bufs.dotMeanGrad) // [rows, 1]
	if err != nil {
		return nil, nil, nil, err
	}

	// dInput = invStd * (dNorm - (dotMeanGrad + centered * invStd^2 * dotScaleGrad) / dModel)
	invStdSq, err := engine.Mul(ctx, invStd, invStd, bufs.invStdSq) // [rows, 1]
	if err != nil {
		return nil, nil, nil, err
	}
	term, err := engine.Mul(ctx, centered, invStdSq, bufs.term) // [rows, dModel] (broadcast [rows,1])
	if err != nil {
		return nil, nil, nil, err
	}
	term, err = engine.Mul(ctx, term, dotScaleGrad, bufs.term) // [rows, dModel] (broadcast [rows,1])
	if err != nil {
		return nil, nil, nil, err
	}
	correction, err := engine.Add(ctx, dotMeanGrad, term, bufs.correction) // [rows, dModel] (broadcast)
	if err != nil {
		return nil, nil, nil, err
	}
	correction, err = engine.MulScalar(ctx, correction, 1.0/float32(dModel), bufs.correction)
	if err != nil {
		return nil, nil, nil, err
	}
	inner, err := engine.Sub(ctx, dNorm, correction, bufs.inner) // [rows, dModel]
	if err != nil {
		return nil, nil, nil, err
	}
	dInput, err = engine.Mul(ctx, invStd, inner, bufs.dInput) // [rows, dModel] (broadcast [rows,1])
	if err != nil {
		return nil, nil, nil, err
	}

	return dInput, newDScale, newDBias, nil
}

// allocLayerCacheBuffers lazily allocates all scratch buffers on a
// gpuBatchLayerCache the first time it is touched. All dimensions come from
// the caller and are fixed for the lifetime of the training/inference loop.
// Subsequent calls on the same cache are no-ops.
func allocLayerCacheBuffers(lc *gpuBatchLayerCache, bsC, seq, totalRows, dModel, nHeads, headDim, ffnDim int) error {
	if lc.buffersAllocated {
		return nil
	}
	bnh := bsC * nHeads

	mk2 := func(rows, cols int) (*tensor.TensorNumeric[float32], error) {
		return tensor.New[float32]([]int{rows, cols}, make([]float32, rows*cols))
	}
	mk1 := func(n int) (*tensor.TensorNumeric[float32], error) {
		return tensor.New[float32]([]int{n}, make([]float32, n))
	}
	mk3 := func(a, b, c int) (*tensor.TensorNumeric[float32], error) {
		return tensor.New[float32]([]int{a, b, c}, make([]float32, a*b*c))
	}
	mk4 := func(a, b, c, d int) (*tensor.TensorNumeric[float32], error) {
		return tensor.New[float32]([]int{a, b, c, d}, make([]float32, a*b*c*d))
	}

	var err error
	// Cached forward activations (previously allocated inside encoderForward).
	if lc.normed1, err = mk2(totalRows, dModel); err != nil {
		return err
	}
	if lc.centered1, err = mk2(totalRows, dModel); err != nil {
		return err
	}
	if lc.invStd1, err = mk2(totalRows, 1); err != nil {
		return err
	}
	if lc.normed2, err = mk2(totalRows, dModel); err != nil {
		return err
	}
	if lc.centered2, err = mk2(totalRows, dModel); err != nil {
		return err
	}
	if lc.invStd2, err = mk2(totalRows, 1); err != nil {
		return err
	}
	if lc.q, err = mk2(totalRows, dModel); err != nil {
		return err
	}
	if lc.k, err = mk2(totalRows, dModel); err != nil {
		return err
	}
	if lc.v, err = mk2(totalRows, dModel); err != nil {
		return err
	}
	if lc.scoresTensor, err = mk3(bnh, seq, seq); err != nil {
		return err
	}
	if lc.attnOut, err = mk2(totalRows, dModel); err != nil {
		return err
	}
	if lc.ffn1PreAct, err = mk2(totalRows, ffnDim); err != nil {
		return err
	}
	if lc.ffn1Out, err = mk2(totalRows, ffnDim); err != nil {
		return err
	}
	if lc.xResidual, err = mk2(totalRows, dModel); err != nil {
		return err
	}
	if lc.xAfterAttn, err = mk2(totalRows, dModel); err != nil {
		return err
	}
	if lc.geluTanhVal, err = mk2(totalRows, ffnDim); err != nil {
		return err
	}

	// Layer norm forward scratch.
	if lc.ln1Mean, err = mk2(totalRows, 1); err != nil {
		return err
	}
	if lc.ln1CentSq, err = mk2(totalRows, dModel); err != nil {
		return err
	}
	if lc.ln1Var, err = mk2(totalRows, 1); err != nil {
		return err
	}
	if lc.ln1VarEps, err = mk2(totalRows, 1); err != nil {
		return err
	}
	if lc.ln1Stddev, err = mk2(totalRows, 1); err != nil {
		return err
	}
	onesData1 := make([]float32, totalRows)
	for i := range onesData1 {
		onesData1[i] = 1.0
	}
	if lc.ln1Ones, err = tensor.New[float32]([]int{totalRows, 1}, onesData1); err != nil {
		return err
	}
	if lc.ln1Scaled, err = mk2(totalRows, dModel); err != nil {
		return err
	}
	if lc.ln1NormMul, err = mk2(totalRows, dModel); err != nil {
		return err
	}
	if lc.ln2Mean, err = mk2(totalRows, 1); err != nil {
		return err
	}
	if lc.ln2CentSq, err = mk2(totalRows, dModel); err != nil {
		return err
	}
	if lc.ln2Var, err = mk2(totalRows, 1); err != nil {
		return err
	}
	if lc.ln2VarEps, err = mk2(totalRows, 1); err != nil {
		return err
	}
	if lc.ln2Stddev, err = mk2(totalRows, 1); err != nil {
		return err
	}
	onesData2 := make([]float32, totalRows)
	for i := range onesData2 {
		onesData2[i] = 1.0
	}
	if lc.ln2Ones, err = tensor.New[float32]([]int{totalRows, 1}, onesData2); err != nil {
		return err
	}
	if lc.ln2Scaled, err = mk2(totalRows, dModel); err != nil {
		return err
	}
	if lc.ln2NormMul, err = mk2(totalRows, dModel); err != nil {
		return err
	}

	// Attention forward scratch.
	if lc.qBiased, err = mk2(totalRows, dModel); err != nil {
		return err
	}
	if lc.kBiased, err = mk2(totalRows, dModel); err != nil {
		return err
	}
	if lc.vBiased, err = mk2(totalRows, dModel); err != nil {
		return err
	}
	if lc.q4d, err = mk4(bsC, seq, nHeads, headDim); err != nil {
		return err
	}
	if lc.k4d, err = mk4(bsC, seq, nHeads, headDim); err != nil {
		return err
	}
	if lc.v4d, err = mk4(bsC, seq, nHeads, headDim); err != nil {
		return err
	}
	if lc.q4dT, err = mk4(bsC, nHeads, seq, headDim); err != nil {
		return err
	}
	if lc.k4dT, err = mk4(bsC, nHeads, seq, headDim); err != nil {
		return err
	}
	if lc.v4dT, err = mk4(bsC, nHeads, seq, headDim); err != nil {
		return err
	}
	if lc.qH, err = mk3(bnh, seq, headDim); err != nil {
		return err
	}
	if lc.kH, err = mk3(bnh, seq, headDim); err != nil {
		return err
	}
	if lc.vH, err = mk3(bnh, seq, headDim); err != nil {
		return err
	}
	if lc.kHT, err = mk3(bnh, headDim, seq); err != nil {
		return err
	}
	if lc.logits, err = mk3(bnh, seq, seq); err != nil {
		return err
	}
	if lc.logitsScaled, err = mk3(bnh, seq, seq); err != nil {
		return err
	}
	if lc.attnH, err = mk3(bnh, seq, headDim); err != nil {
		return err
	}
	if lc.attnH4d, err = mk4(bsC, nHeads, seq, headDim); err != nil {
		return err
	}
	if lc.attnH4dT, err = mk4(bsC, seq, nHeads, headDim); err != nil {
		return err
	}
	if lc.attnProj, err = mk2(totalRows, dModel); err != nil {
		return err
	}
	if lc.attnProjBias, err = mk2(totalRows, dModel); err != nil {
		return err
	}
	if lc.xAfterRes1, err = mk2(totalRows, dModel); err != nil {
		return err
	}

	// FFN forward scratch.
	if lc.ffn1Matmul, err = mk2(totalRows, ffnDim); err != nil {
		return err
	}
	if lc.geluX3, err = mk2(totalRows, ffnDim); err != nil {
		return err
	}
	if lc.geluInner1, err = mk2(totalRows, ffnDim); err != nil {
		return err
	}
	if lc.geluInner2, err = mk2(totalRows, ffnDim); err != nil {
		return err
	}
	if lc.geluInner3, err = mk2(totalRows, ffnDim); err != nil {
		return err
	}
	if lc.geluOnePlusTanh, err = mk2(totalRows, ffnDim); err != nil {
		return err
	}
	if lc.geluXTimes, err = mk2(totalRows, ffnDim); err != nil {
		return err
	}
	if lc.ffn2Matmul, err = mk2(totalRows, dModel); err != nil {
		return err
	}
	if lc.ffn2Out, err = mk2(totalRows, dModel); err != nil {
		return err
	}
	if lc.xAfterRes2, err = mk2(totalRows, dModel); err != nil {
		return err
	}

	// Backward scratch.
	if lc.ffn1OutT, err = mk2(ffnDim, totalRows); err != nil {
		return err
	}
	if lc.dFfn2W, err = mk2(ffnDim, dModel); err != nil {
		return err
	}
	if lc.dFfn2BSum, err = mk1(dModel); err != nil {
		return err
	}
	if lc.dFfn2BR, err = mk2(1, dModel); err != nil {
		return err
	}
	if lc.dFfn1Out, err = mk2(totalRows, ffnDim); err != nil {
		return err
	}
	if lc.gTerm1, err = mk2(totalRows, ffnDim); err != nil {
		return err
	}
	if lc.gTanhSq, err = mk2(totalRows, ffnDim); err != nil {
		return err
	}
	if lc.gSechSq, err = mk2(totalRows, ffnDim); err != nil {
		return err
	}
	if lc.gXSq, err = mk2(totalRows, ffnDim); err != nil {
		return err
	}
	if lc.gDudx, err = mk2(totalRows, ffnDim); err != nil {
		return err
	}
	if lc.gTerm2, err = mk2(totalRows, ffnDim); err != nil {
		return err
	}
	if lc.gDeriv, err = mk2(totalRows, ffnDim); err != nil {
		return err
	}
	if lc.dFfn1PreAct, err = mk2(totalRows, ffnDim); err != nil {
		return err
	}
	if lc.normed2T, err = mk2(dModel, totalRows); err != nil {
		return err
	}
	if lc.dFfn1W, err = mk2(dModel, ffnDim); err != nil {
		return err
	}
	if lc.dFfn1BSum, err = mk1(ffnDim); err != nil {
		return err
	}
	if lc.dFfn1BR, err = mk2(1, ffnDim); err != nil {
		return err
	}
	if lc.dNormed2, err = mk2(totalRows, dModel); err != nil {
		return err
	}
	if lc.dAttnProjOut, err = mk2(totalRows, dModel); err != nil {
		return err
	}
	if lc.attnOutT, err = mk2(dModel, totalRows); err != nil {
		return err
	}
	if lc.dOW, err = mk2(dModel, dModel); err != nil {
		return err
	}
	if lc.dOBSum, err = mk1(dModel); err != nil {
		return err
	}
	if lc.dOBR, err = mk2(1, dModel); err != nil {
		return err
	}
	if lc.dAttnOut, err = mk2(totalRows, dModel); err != nil {
		return err
	}
	if lc.dAO4d, err = mk4(bsC, seq, nHeads, headDim); err != nil {
		return err
	}
	if lc.dAO4dT, err = mk4(bsC, nHeads, seq, headDim); err != nil {
		return err
	}
	if lc.dAttnOutH, err = mk3(bnh, seq, headDim); err != nil {
		return err
	}
	if lc.bwdQ4d, err = mk4(bsC, seq, nHeads, headDim); err != nil {
		return err
	}
	if lc.bwdQ4dT, err = mk4(bsC, nHeads, seq, headDim); err != nil {
		return err
	}
	if lc.bwdQH, err = mk3(bnh, seq, headDim); err != nil {
		return err
	}
	if lc.bwdK4d, err = mk4(bsC, seq, nHeads, headDim); err != nil {
		return err
	}
	if lc.bwdK4dT, err = mk4(bsC, nHeads, seq, headDim); err != nil {
		return err
	}
	if lc.bwdKH, err = mk3(bnh, seq, headDim); err != nil {
		return err
	}
	if lc.bwdV4d, err = mk4(bsC, seq, nHeads, headDim); err != nil {
		return err
	}
	if lc.bwdV4dT, err = mk4(bsC, nHeads, seq, headDim); err != nil {
		return err
	}
	if lc.bwdVH, err = mk3(bnh, seq, headDim); err != nil {
		return err
	}
	if lc.vHT, err = mk3(bnh, headDim, seq); err != nil {
		return err
	}
	if lc.dScores, err = mk3(bnh, seq, seq); err != nil {
		return err
	}
	if lc.scoresT, err = mk3(bnh, seq, seq); err != nil {
		return err
	}
	if lc.dVH, err = mk3(bnh, seq, headDim); err != nil {
		return err
	}
	if lc.sDScores, err = mk3(bnh, seq, seq); err != nil {
		return err
	}
	if lc.rowSum, err = mk3(bnh, seq, 1); err != nil {
		return err
	}
	if lc.dLogits1, err = mk3(bnh, seq, seq); err != nil {
		return err
	}
	if lc.dLogits2, err = mk3(bnh, seq, seq); err != nil {
		return err
	}
	if lc.dLogits, err = mk3(bnh, seq, seq); err != nil {
		return err
	}
	if lc.dQH, err = mk3(bnh, seq, headDim); err != nil {
		return err
	}
	if lc.dLogitsT, err = mk3(bnh, seq, seq); err != nil {
		return err
	}
	if lc.dKH, err = mk3(bnh, seq, headDim); err != nil {
		return err
	}
	if lc.dQH4d, err = mk4(bsC, nHeads, seq, headDim); err != nil {
		return err
	}
	if lc.dQH4dT, err = mk4(bsC, seq, nHeads, headDim); err != nil {
		return err
	}
	if lc.dQT, err = mk2(totalRows, dModel); err != nil {
		return err
	}
	if lc.dKH4d, err = mk4(bsC, nHeads, seq, headDim); err != nil {
		return err
	}
	if lc.dKH4dT, err = mk4(bsC, seq, nHeads, headDim); err != nil {
		return err
	}
	if lc.dKT, err = mk2(totalRows, dModel); err != nil {
		return err
	}
	if lc.dVH4d, err = mk4(bsC, nHeads, seq, headDim); err != nil {
		return err
	}
	if lc.dVH4dT, err = mk4(bsC, seq, nHeads, headDim); err != nil {
		return err
	}
	if lc.dVT, err = mk2(totalRows, dModel); err != nil {
		return err
	}
	if lc.normed1T, err = mk2(dModel, totalRows); err != nil {
		return err
	}
	if lc.dQW, err = mk2(dModel, dModel); err != nil {
		return err
	}
	if lc.dQBSum, err = mk1(dModel); err != nil {
		return err
	}
	if lc.dQBR, err = mk2(1, dModel); err != nil {
		return err
	}
	if lc.dKW, err = mk2(dModel, dModel); err != nil {
		return err
	}
	if lc.dKBSum, err = mk1(dModel); err != nil {
		return err
	}
	if lc.dKBR, err = mk2(1, dModel); err != nil {
		return err
	}
	if lc.dVW, err = mk2(dModel, dModel); err != nil {
		return err
	}
	if lc.dVBSum, err = mk1(dModel); err != nil {
		return err
	}
	if lc.dVBR, err = mk2(1, dModel); err != nil {
		return err
	}
	if lc.dN1q, err = mk2(totalRows, dModel); err != nil {
		return err
	}
	if lc.dN1k, err = mk2(totalRows, dModel); err != nil {
		return err
	}
	if lc.dN1v, err = mk2(totalRows, dModel); err != nil {
		return err
	}
	if lc.dN1Sum1, err = mk2(totalRows, dModel); err != nil {
		return err
	}
	if lc.dNormed1, err = mk2(totalRows, dModel); err != nil {
		return err
	}
	if lc.dXOut, err = mk2(totalRows, dModel); err != nil {
		return err
	}

	// Layer norm backward scratch (two sets to support ln1 and ln2 in same iter).
	if lc.lnbNormVal, err = mk2(totalRows, dModel); err != nil {
		return err
	}
	if lc.lnbDScaleBatch, err = mk2(totalRows, dModel); err != nil {
		return err
	}
	if lc.lnbDScaleSum, err = mk1(dModel); err != nil {
		return err
	}
	if lc.lnbDScaleSumR, err = mk2(1, dModel); err != nil {
		return err
	}
	if lc.lnbDBiasSum, err = mk1(dModel); err != nil {
		return err
	}
	if lc.lnbDBiasSumR, err = mk2(1, dModel); err != nil {
		return err
	}
	if lc.lnbDNorm, err = mk2(totalRows, dModel); err != nil {
		return err
	}
	if lc.lnbDNormCent, err = mk2(totalRows, dModel); err != nil {
		return err
	}
	if lc.lnbDotScaleGrad, err = mk2(totalRows, 1); err != nil {
		return err
	}
	if lc.lnbDotMeanGrad, err = mk2(totalRows, 1); err != nil {
		return err
	}
	if lc.lnbInvStdSq, err = mk2(totalRows, 1); err != nil {
		return err
	}
	if lc.lnbTerm, err = mk2(totalRows, dModel); err != nil {
		return err
	}
	if lc.lnbCorrection, err = mk2(totalRows, dModel); err != nil {
		return err
	}
	if lc.lnbInner, err = mk2(totalRows, dModel); err != nil {
		return err
	}
	if lc.lnbDInput, err = mk2(totalRows, dModel); err != nil {
		return err
	}
	if lc.ln2bNormVal, err = mk2(totalRows, dModel); err != nil {
		return err
	}
	if lc.ln2bDScaleBatch, err = mk2(totalRows, dModel); err != nil {
		return err
	}
	if lc.ln2bDScaleSum, err = mk1(dModel); err != nil {
		return err
	}
	if lc.ln2bDScaleSumR, err = mk2(1, dModel); err != nil {
		return err
	}
	if lc.ln2bDBiasSum, err = mk1(dModel); err != nil {
		return err
	}
	if lc.ln2bDBiasSumR, err = mk2(1, dModel); err != nil {
		return err
	}
	if lc.ln2bDNorm, err = mk2(totalRows, dModel); err != nil {
		return err
	}
	if lc.ln2bDNormCent, err = mk2(totalRows, dModel); err != nil {
		return err
	}
	if lc.ln2bDotScaleGrad, err = mk2(totalRows, 1); err != nil {
		return err
	}
	if lc.ln2bDotMeanGrad, err = mk2(totalRows, 1); err != nil {
		return err
	}
	if lc.ln2bInvStdSq, err = mk2(totalRows, 1); err != nil {
		return err
	}
	if lc.ln2bTerm, err = mk2(totalRows, dModel); err != nil {
		return err
	}
	if lc.ln2bCorrection, err = mk2(totalRows, dModel); err != nil {
		return err
	}
	if lc.ln2bInner, err = mk2(totalRows, dModel); err != nil {
		return err
	}
	if lc.ln2bDInput, err = mk2(totalRows, dModel); err != nil {
		return err
	}

	lc.buffersAllocated = true
	return nil
}

// encoderLayersToGPU converts a slice of encoderLayer (linearLayer-based) to
// gpuEncoderLayer (flat tensor pointers) so that the shared encoderForward can
// be used by both the inference Forward path and the GPU training path.
// The returned slice shares the underlying tensor data with the originals.
func encoderLayersToGPU(layers []encoderLayer) []gpuEncoderLayer {
	out := make([]gpuEncoderLayer, len(layers))
	for i, l := range layers {
		out[i] = gpuEncoderLayer{
			qW: l.qProj.weights, qB: l.qProj.biases,
			kW: l.kProj.weights, kB: l.kProj.biases,
			vW: l.vProj.weights, vB: l.vProj.biases,
			oW: l.oProj.weights, oB: l.oProj.biases,
			ffn1W: l.ffn1.weights, ffn1B: l.ffn1.biases,
			ffn2W: l.ffn2.weights, ffn2B: l.ffn2.biases,
			norm1: l.norm1, bias1: l.bias1,
			norm2: l.norm2, bias2: l.bias2,
		}
	}
	return out
}

// encoderForward runs the PatchTST transformer encoder layers using engine ops.
// x: [totalRows, dModel] input tensor (after patch embedding + pos embedding).
// bsC is batch*channels, numPatches is the sequence length per sample-channel.
// totalRows must equal bsC * numPatches.
//
// layerCaches is an IN parameter: on the first call, all per-layer scratch
// buffers are lazily allocated and cached; subsequent calls reuse these
// buffers via the dst-param variants of the engine ops. This eliminates
// per-batch allocations in the hot training path.
//
// Returns: output [totalRows, dModel] (the final layer's xAfterRes2 buffer).
func encoderForward(
	ctx context.Context,
	engine compute.Engine[float32],
	x *tensor.TensorNumeric[float32],
	layers []gpuEncoderLayer,
	layerCaches []gpuBatchLayerCache,
	fusedGPU *fusedEncoderGPU,
	bsC, numPatches, totalRows, dModel, nHeads, headDim, ffnDim int,
) (*tensor.TensorNumeric[float32], error) {
	nLayers := len(layers)
	if len(layerCaches) != nLayers {
		return nil, fmt.Errorf("encoderForward: layerCaches len %d != nLayers %d", len(layerCaches), nLayers)
	}

	// Try fused encoder kernel path (replaces ~78 ops/layer with 1 call).
	if result, used, err := fusedEncoderForward(ctx, engine, x, layers, layerCaches, fusedGPU,
		bsC, numPatches, totalRows, dModel, nHeads, headDim, ffnDim); err != nil {
		return nil, fmt.Errorf("encoderForward fused: %w", err)
	} else if used {
		return result, nil
	}

	seq := numPatches
	bnh := bsC * nHeads

	var err error
	for li := 0; li < nLayers; li++ {
		layer := &layers[li]
		lc := &layerCaches[li]

		if err = allocLayerCacheBuffers(lc, bsC, seq, totalRows, dModel, nHeads, headDim, ffnDim); err != nil {
			return nil, err
		}

		// Layer norm 1 via engine ops (over all bsC*numPatches rows).
		_, _, _, err = layerNormForwardWithEngine(ctx, engine, x, layer.norm1, layer.bias1,
			lc.normed1, lc.centered1, lc.invStd1,
			lc.ln1Mean, lc.ln1CentSq, lc.ln1Var, lc.ln1VarEps, lc.ln1Stddev, lc.ln1Ones, lc.ln1Scaled, lc.ln1NormMul,
			totalRows, dModel)
		if err != nil {
			return nil, err
		}

		// Q/K/V projections.
		if err := matMulInto(ctx, engine, lc.normed1, layer.qW, lc.q); err != nil {
			return nil, err
		}
		if _, err = engine.Add(ctx, lc.q, layer.qB, lc.q); err != nil {
			return nil, err
		}
		if err := matMulInto(ctx, engine, lc.normed1, layer.kW, lc.k); err != nil {
			return nil, err
		}
		if _, err = engine.Add(ctx, lc.k, layer.kB, lc.k); err != nil {
			return nil, err
		}
		if err := matMulInto(ctx, engine, lc.normed1, layer.vW, lc.v); err != nil {
			return nil, err
		}
		if _, err = engine.Add(ctx, lc.v, layer.vB, lc.v); err != nil {
			return nil, err
		}

		// Batched attention.
		scale := float32(1.0 / math.Sqrt(float64(headDim)))

		// Q: reshape -> transpose -> reshape to [bnh, seq, headDim].
		if _, err = engine.Reshape(ctx, lc.q, []int{bsC, seq, nHeads, headDim}, lc.q4d); err != nil {
			return nil, err
		}
		if _, err = engine.Transpose(ctx, lc.q4d, []int{0, 2, 1, 3}, lc.q4dT); err != nil {
			return nil, err
		}
		if _, err = engine.Reshape(ctx, lc.q4dT, []int{bnh, seq, headDim}, lc.qH); err != nil {
			return nil, err
		}
		// K.
		if _, err = engine.Reshape(ctx, lc.k, []int{bsC, seq, nHeads, headDim}, lc.k4d); err != nil {
			return nil, err
		}
		if _, err = engine.Transpose(ctx, lc.k4d, []int{0, 2, 1, 3}, lc.k4dT); err != nil {
			return nil, err
		}
		if _, err = engine.Reshape(ctx, lc.k4dT, []int{bnh, seq, headDim}, lc.kH); err != nil {
			return nil, err
		}
		// V.
		if _, err = engine.Reshape(ctx, lc.v, []int{bsC, seq, nHeads, headDim}, lc.v4d); err != nil {
			return nil, err
		}
		if _, err = engine.Transpose(ctx, lc.v4d, []int{0, 2, 1, 3}, lc.v4dT); err != nil {
			return nil, err
		}
		if _, err = engine.Reshape(ctx, lc.v4dT, []int{bnh, seq, headDim}, lc.vH); err != nil {
			return nil, err
		}

		// scores = Q @ K^T * scale.
		if _, err = engine.Transpose(ctx, lc.kH, []int{0, 2, 1}, lc.kHT); err != nil {
			return nil, err
		}
		if err := matMulInto(ctx, engine, lc.qH, lc.kHT, lc.logits); err != nil {
			return nil, err
		}
		if _, err = engine.MulScalar(ctx, lc.logits, scale, lc.logitsScaled); err != nil {
			return nil, err
		}

		// softmax along last axis.
		if _, err = engine.Softmax(ctx, lc.logitsScaled, -1, lc.scoresTensor); err != nil {
			return nil, err
		}

		// attnOut = scores @ V -> [bnh, seq, headDim] -> reshape -> transpose -> reshape.
		if err := matMulInto(ctx, engine, lc.scoresTensor, lc.vH, lc.attnH); err != nil {
			return nil, err
		}
		if _, err = engine.Reshape(ctx, lc.attnH, []int{bsC, nHeads, seq, headDim}, lc.attnH4d); err != nil {
			return nil, err
		}
		if _, err = engine.Transpose(ctx, lc.attnH4d, []int{0, 2, 1, 3}, lc.attnH4dT); err != nil {
			return nil, err
		}
		if _, err = engine.Reshape(ctx, lc.attnH4dT, []int{totalRows, dModel}, lc.attnOut); err != nil {
			return nil, err
		}

		// Output projection.
		if err := matMulInto(ctx, engine, lc.attnOut, layer.oW, lc.attnProj); err != nil {
			return nil, err
		}
		if _, err = engine.Add(ctx, lc.attnProj, layer.oB, lc.attnProjBias); err != nil {
			return nil, err
		}

		// Residual 1: xAfterRes1 = x + attnProjBias.
		if _, err = engine.Add(ctx, x, lc.attnProjBias, lc.xAfterRes1); err != nil {
			return nil, err
		}

		// Layer norm 2: reads xAfterRes1.
		_, _, _, err = layerNormForwardWithEngine(ctx, engine, lc.xAfterRes1, layer.norm2, layer.bias2,
			lc.normed2, lc.centered2, lc.invStd2,
			lc.ln2Mean, lc.ln2CentSq, lc.ln2Var, lc.ln2VarEps, lc.ln2Stddev, lc.ln2Ones, lc.ln2Scaled, lc.ln2NormMul,
			totalRows, dModel)
		if err != nil {
			return nil, err
		}

		// FFN1: [totalRows, dModel] @ [dModel, ffnDim] = [totalRows, ffnDim].
		if err := matMulInto(ctx, engine, lc.normed2, layer.ffn1W, lc.ffn1Matmul); err != nil {
			return nil, err
		}
		if _, err = engine.Add(ctx, lc.ffn1Matmul, layer.ffn1B, lc.ffn1PreAct); err != nil {
			return nil, err
		}

		// GELU via engine ops: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
		geluIn := lc.ffn1PreAct
		if _, err = engine.Mul(ctx, geluIn, geluIn, lc.geluX3); err != nil {
			return nil, err
		}
		if _, err = engine.Mul(ctx, lc.geluX3, geluIn, lc.geluX3); err != nil {
			return nil, err
		}
		if _, err = engine.MulScalar(ctx, lc.geluX3, float32(0.044715), lc.geluInner1); err != nil {
			return nil, err
		}
		if _, err = engine.Add(ctx, geluIn, lc.geluInner1, lc.geluInner2); err != nil {
			return nil, err
		}
		if _, err = engine.MulScalar(ctx, lc.geluInner2, float32(math.Sqrt(2.0/math.Pi)), lc.geluInner3); err != nil {
			return nil, err
		}
		if _, err = engine.Tanh(ctx, lc.geluInner3, lc.geluTanhVal); err != nil {
			return nil, err
		}
		if _, err = engine.AddScalar(ctx, lc.geluTanhVal, float32(1.0), lc.geluOnePlusTanh); err != nil {
			return nil, err
		}
		if _, err = engine.Mul(ctx, geluIn, lc.geluOnePlusTanh, lc.geluXTimes); err != nil {
			return nil, err
		}
		if _, err = engine.MulScalar(ctx, lc.geluXTimes, float32(0.5), lc.ffn1Out); err != nil {
			return nil, err
		}

		// FFN2.
		if err := matMulInto(ctx, engine, lc.ffn1Out, layer.ffn2W, lc.ffn2Matmul); err != nil {
			return nil, err
		}
		if _, err = engine.Add(ctx, lc.ffn2Matmul, layer.ffn2B, lc.ffn2Out); err != nil {
			return nil, err
		}

		// Residual 2: xAfterRes2 = xAfterRes1 + ffn2Out.
		if _, err = engine.Add(ctx, lc.xAfterRes1, lc.ffn2Out, lc.xAfterRes2); err != nil {
			return nil, err
		}

		// Next layer input is this layer's output.
		x = lc.xAfterRes2
	}

	return x, nil
}

// encoderBackward runs the PatchTST transformer encoder backward pass.
// dX: [totalRows, dModel] upstream gradient.
// Returns: gradient with respect to encoder input.
func encoderBackward(
	ctx context.Context,
	engine compute.Engine[float32],
	dX *tensor.TensorNumeric[float32],
	layers []gpuEncoderLayer,
	grads []gpuEncoderLayer,
	layerCaches []gpuBatchLayerCache,
	lwts []layerTransposes,
	bsC, numPatches, totalRows, dModel, nHeads, headDim, ffnDim int,
) (*tensor.TensorNumeric[float32], error) {
	seq := numPatches
	bnh := bsC * nHeads

	for li := len(layers) - 1; li >= 0; li-- {
		layer := &layers[li]
		lc := &layerCaches[li]
		dg := &grads[li]
		lt := &lwts[li]

		// FFN2 backward.
		// dFFN2Out = dX (from residual 2).
		// dW += ffn1Out^T @ dFFN2Out : [ffnDim, totalRows] @ [totalRows, dModel]
		if _, err := engine.Transpose(ctx, lc.ffn1Out, []int{1, 0}, lc.ffn1OutT); err != nil {
			return nil, err
		}
		if err := matMulInto(ctx, engine, lc.ffn1OutT, dX, lc.dFfn2W); err != nil {
			return nil, err
		}
		if _, err := engine.Add(ctx, dg.ffn2W, lc.dFfn2W, dg.ffn2W); err != nil {
			return nil, err
		}
		if _, err := engine.Sum(ctx, dX, 0, false, lc.dFfn2BSum); err != nil {
			return nil, err
		}
		if _, err := engine.Reshape(ctx, lc.dFfn2BSum, []int{1, dModel}, lc.dFfn2BR); err != nil {
			return nil, err
		}
		if _, err := engine.Add(ctx, dg.ffn2B, lc.dFfn2BR, dg.ffn2B); err != nil {
			return nil, err
		}
		// dFFN1Out = dFFN2Out @ ffn2W^T
		if err := matMulInto(ctx, engine, dX, lt.ffn2WT, lc.dFfn1Out); err != nil {
			return nil, err
		}

		// GELU backward: d/dx = 0.5*(1+tanh) + 0.5*x*(1-tanh^2)*sqrt(2/pi)*(1+3*0.044715*x^2)
		geluX := lc.ffn1PreAct
		tanhVal := lc.geluTanhVal

		// term1 = 0.5 * (1 + tanh)
		if _, err := engine.AddScalar(ctx, tanhVal, float32(1.0), lc.gTerm1); err != nil {
			return nil, err
		}
		if _, err := engine.MulScalar(ctx, lc.gTerm1, float32(0.5), lc.gTerm1); err != nil {
			return nil, err
		}
		// sech^2 = 1 - tanh^2
		if _, err := engine.Mul(ctx, tanhVal, tanhVal, lc.gTanhSq); err != nil {
			return nil, err
		}
		if _, err := engine.MulScalar(ctx, lc.gTanhSq, float32(-1.0), lc.gSechSq); err != nil {
			return nil, err
		}
		if _, err := engine.AddScalar(ctx, lc.gSechSq, float32(1.0), lc.gSechSq); err != nil {
			return nil, err
		}
		// du/dx = sqrt(2/pi) * (1 + 3*0.044715*x^2)
		if _, err := engine.Mul(ctx, geluX, geluX, lc.gXSq); err != nil {
			return nil, err
		}
		if _, err := engine.MulScalar(ctx, lc.gXSq, float32(3*0.044715), lc.gDudx); err != nil {
			return nil, err
		}
		if _, err := engine.AddScalar(ctx, lc.gDudx, float32(1.0), lc.gDudx); err != nil {
			return nil, err
		}
		if _, err := engine.MulScalar(ctx, lc.gDudx, float32(math.Sqrt(2.0/math.Pi)), lc.gDudx); err != nil {
			return nil, err
		}
		// term2 = 0.5 * x * sech^2 * du/dx
		if _, err := engine.Mul(ctx, geluX, lc.gSechSq, lc.gTerm2); err != nil {
			return nil, err
		}
		if _, err := engine.Mul(ctx, lc.gTerm2, lc.gDudx, lc.gTerm2); err != nil {
			return nil, err
		}
		if _, err := engine.MulScalar(ctx, lc.gTerm2, float32(0.5), lc.gTerm2); err != nil {
			return nil, err
		}
		// geluDeriv = term1 + term2
		if _, err := engine.Add(ctx, lc.gTerm1, lc.gTerm2, lc.gDeriv); err != nil {
			return nil, err
		}
		if _, err := engine.Mul(ctx, lc.dFfn1Out, lc.gDeriv, lc.dFfn1PreAct); err != nil {
			return nil, err
		}

		// FFN1 backward.
		if _, err := engine.Transpose(ctx, lc.normed2, []int{1, 0}, lc.normed2T); err != nil {
			return nil, err
		}
		if err := matMulInto(ctx, engine, lc.normed2T, lc.dFfn1PreAct, lc.dFfn1W); err != nil {
			return nil, err
		}
		if _, err := engine.Add(ctx, dg.ffn1W, lc.dFfn1W, dg.ffn1W); err != nil {
			return nil, err
		}
		if _, err := engine.Sum(ctx, lc.dFfn1PreAct, 0, false, lc.dFfn1BSum); err != nil {
			return nil, err
		}
		if _, err := engine.Reshape(ctx, lc.dFfn1BSum, []int{1, ffnDim}, lc.dFfn1BR); err != nil {
			return nil, err
		}
		if _, err := engine.Add(ctx, dg.ffn1B, lc.dFfn1BR, dg.ffn1B); err != nil {
			return nil, err
		}
		// dNormed2 = dFFN1PreAct @ ffn1W^T
		if err := matMulInto(ctx, engine, lc.dFfn1PreAct, lt.ffn1WT, lc.dNormed2); err != nil {
			return nil, err
		}

		// LayerNorm2 backward.
		ln2Bufs := &lnBwdBufs{
			normVal:      lc.ln2bNormVal,
			dScaleBatch:  lc.ln2bDScaleBatch,
			dScaleSum:    lc.ln2bDScaleSum,
			dScaleSumR:   lc.ln2bDScaleSumR,
			dBiasSum:     lc.ln2bDBiasSum,
			dBiasSumR:    lc.ln2bDBiasSumR,
			dNorm:        lc.ln2bDNorm,
			dNormCent:    lc.ln2bDNormCent,
			dotScaleGrad: lc.ln2bDotScaleGrad,
			dotMeanGrad:  lc.ln2bDotMeanGrad,
			invStdSq:     lc.ln2bInvStdSq,
			term:         lc.ln2bTerm,
			correction:   lc.ln2bCorrection,
			inner:        lc.ln2bInner,
			dInput:       lc.ln2bDInput,
		}
		dLN2Input, _, _, err := layerNormBackwardWithEngine(ctx, engine, lc.dNormed2, lc.centered2, lc.invStd2,
			layer.norm2, dg.norm2, dg.bias2, ln2Bufs, totalRows, dModel)
		if err != nil {
			return nil, err
		}
		// Add residual gradient from FFN path: dAttnProjOut = dLN2Input + dX.
		if _, err := engine.Add(ctx, dLN2Input, dX, lc.dAttnProjOut); err != nil {
			return nil, err
		}
		// dW += attnOut^T @ dAttnProjOut
		if _, err := engine.Transpose(ctx, lc.attnOut, []int{1, 0}, lc.attnOutT); err != nil {
			return nil, err
		}
		if err := matMulInto(ctx, engine, lc.attnOutT, lc.dAttnProjOut, lc.dOW); err != nil {
			return nil, err
		}
		if _, err := engine.Add(ctx, dg.oW, lc.dOW, dg.oW); err != nil {
			return nil, err
		}
		if _, err := engine.Sum(ctx, lc.dAttnProjOut, 0, false, lc.dOBSum); err != nil {
			return nil, err
		}
		if _, err := engine.Reshape(ctx, lc.dOBSum, []int{1, dModel}, lc.dOBR); err != nil {
			return nil, err
		}
		if _, err := engine.Add(ctx, dg.oB, lc.dOBR, dg.oB); err != nil {
			return nil, err
		}
		// dAttnOut = dAttnProjOut @ oW^T
		if err := matMulInto(ctx, engine, lc.dAttnProjOut, lt.oWT, lc.dAttnOut); err != nil {
			return nil, err
		}

		// Batched attention backward.
		attnScale := float32(1.0 / math.Sqrt(float64(headDim)))

		if _, err := engine.Reshape(ctx, lc.dAttnOut, []int{bsC, seq, nHeads, headDim}, lc.dAO4d); err != nil {
			return nil, err
		}
		if _, err := engine.Transpose(ctx, lc.dAO4d, []int{0, 2, 1, 3}, lc.dAO4dT); err != nil {
			return nil, err
		}
		if _, err := engine.Reshape(ctx, lc.dAO4dT, []int{bnh, seq, headDim}, lc.dAttnOutH); err != nil {
			return nil, err
		}

		// Reshape Q/K/V to [bnh, seq, headDim] using bwd scratch buffers.
		if _, err := engine.Reshape(ctx, lc.q, []int{bsC, seq, nHeads, headDim}, lc.bwdQ4d); err != nil {
			return nil, err
		}
		if _, err := engine.Transpose(ctx, lc.bwdQ4d, []int{0, 2, 1, 3}, lc.bwdQ4dT); err != nil {
			return nil, err
		}
		if _, err := engine.Reshape(ctx, lc.bwdQ4dT, []int{bnh, seq, headDim}, lc.bwdQH); err != nil {
			return nil, err
		}
		if _, err := engine.Reshape(ctx, lc.k, []int{bsC, seq, nHeads, headDim}, lc.bwdK4d); err != nil {
			return nil, err
		}
		if _, err := engine.Transpose(ctx, lc.bwdK4d, []int{0, 2, 1, 3}, lc.bwdK4dT); err != nil {
			return nil, err
		}
		if _, err := engine.Reshape(ctx, lc.bwdK4dT, []int{bnh, seq, headDim}, lc.bwdKH); err != nil {
			return nil, err
		}
		if _, err := engine.Reshape(ctx, lc.v, []int{bsC, seq, nHeads, headDim}, lc.bwdV4d); err != nil {
			return nil, err
		}
		if _, err := engine.Transpose(ctx, lc.bwdV4d, []int{0, 2, 1, 3}, lc.bwdV4dT); err != nil {
			return nil, err
		}
		if _, err := engine.Reshape(ctx, lc.bwdV4dT, []int{bnh, seq, headDim}, lc.bwdVH); err != nil {
			return nil, err
		}

		// dScores = dAttnOut @ V^T.
		if _, err := engine.Transpose(ctx, lc.bwdVH, []int{0, 2, 1}, lc.vHT); err != nil {
			return nil, err
		}
		if err := matMulInto(ctx, engine, lc.dAttnOutH, lc.vHT, lc.dScores); err != nil {
			return nil, err
		}
		// dV = scores^T @ dAttnOut.
		if _, err := engine.Transpose(ctx, lc.scoresTensor, []int{0, 2, 1}, lc.scoresT); err != nil {
			return nil, err
		}
		if err := matMulInto(ctx, engine, lc.scoresT, lc.dAttnOutH, lc.dVH); err != nil {
			return nil, err
		}

		// Softmax backward.
		if _, err := engine.Mul(ctx, lc.scoresTensor, lc.dScores, lc.sDScores); err != nil {
			return nil, err
		}
		if _, err := engine.Sum(ctx, lc.sDScores, 2, true, lc.rowSum); err != nil {
			return nil, err
		}
		if _, err := engine.Sub(ctx, lc.dScores, lc.rowSum, lc.dLogits1); err != nil {
			return nil, err
		}
		if _, err := engine.Mul(ctx, lc.scoresTensor, lc.dLogits1, lc.dLogits2); err != nil {
			return nil, err
		}
		if _, err := engine.MulScalar(ctx, lc.dLogits2, attnScale, lc.dLogits); err != nil {
			return nil, err
		}

		// dQ = dLogits @ K.
		if err := matMulInto(ctx, engine, lc.dLogits, lc.bwdKH, lc.dQH); err != nil {
			return nil, err
		}
		// dK = dLogits^T @ Q.
		if _, err := engine.Transpose(ctx, lc.dLogits, []int{0, 2, 1}, lc.dLogitsT); err != nil {
			return nil, err
		}
		if err := matMulInto(ctx, engine, lc.dLogitsT, lc.bwdQH, lc.dKH); err != nil {
			return nil, err
		}

		// Reshape dQ/dK/dV to [totalRows, dModel].
		if _, err := engine.Reshape(ctx, lc.dQH, []int{bsC, nHeads, seq, headDim}, lc.dQH4d); err != nil {
			return nil, err
		}
		if _, err := engine.Transpose(ctx, lc.dQH4d, []int{0, 2, 1, 3}, lc.dQH4dT); err != nil {
			return nil, err
		}
		if _, err := engine.Reshape(ctx, lc.dQH4dT, []int{totalRows, dModel}, lc.dQT); err != nil {
			return nil, err
		}
		if _, err := engine.Reshape(ctx, lc.dKH, []int{bsC, nHeads, seq, headDim}, lc.dKH4d); err != nil {
			return nil, err
		}
		if _, err := engine.Transpose(ctx, lc.dKH4d, []int{0, 2, 1, 3}, lc.dKH4dT); err != nil {
			return nil, err
		}
		if _, err := engine.Reshape(ctx, lc.dKH4dT, []int{totalRows, dModel}, lc.dKT); err != nil {
			return nil, err
		}
		if _, err := engine.Reshape(ctx, lc.dVH, []int{bsC, nHeads, seq, headDim}, lc.dVH4d); err != nil {
			return nil, err
		}
		if _, err := engine.Transpose(ctx, lc.dVH4d, []int{0, 2, 1, 3}, lc.dVH4dT); err != nil {
			return nil, err
		}
		if _, err := engine.Reshape(ctx, lc.dVH4dT, []int{totalRows, dModel}, lc.dVT); err != nil {
			return nil, err
		}

		if _, err := engine.Transpose(ctx, lc.normed1, []int{1, 0}, lc.normed1T); err != nil {
			return nil, err
		}

		// Q/K/V weight gradients.
		if err := matMulInto(ctx, engine, lc.normed1T, lc.dQT, lc.dQW); err != nil {
			return nil, err
		}
		if _, err := engine.Add(ctx, dg.qW, lc.dQW, dg.qW); err != nil {
			return nil, err
		}
		if _, err := engine.Sum(ctx, lc.dQT, 0, false, lc.dQBSum); err != nil {
			return nil, err
		}
		if _, err := engine.Reshape(ctx, lc.dQBSum, []int{1, dModel}, lc.dQBR); err != nil {
			return nil, err
		}
		if _, err := engine.Add(ctx, dg.qB, lc.dQBR, dg.qB); err != nil {
			return nil, err
		}
		if err := matMulInto(ctx, engine, lc.normed1T, lc.dKT, lc.dKW); err != nil {
			return nil, err
		}
		if _, err := engine.Add(ctx, dg.kW, lc.dKW, dg.kW); err != nil {
			return nil, err
		}
		if _, err := engine.Sum(ctx, lc.dKT, 0, false, lc.dKBSum); err != nil {
			return nil, err
		}
		if _, err := engine.Reshape(ctx, lc.dKBSum, []int{1, dModel}, lc.dKBR); err != nil {
			return nil, err
		}
		if _, err := engine.Add(ctx, dg.kB, lc.dKBR, dg.kB); err != nil {
			return nil, err
		}
		if err := matMulInto(ctx, engine, lc.normed1T, lc.dVT, lc.dVW); err != nil {
			return nil, err
		}
		if _, err := engine.Add(ctx, dg.vW, lc.dVW, dg.vW); err != nil {
			return nil, err
		}
		if _, err := engine.Sum(ctx, lc.dVT, 0, false, lc.dVBSum); err != nil {
			return nil, err
		}
		if _, err := engine.Reshape(ctx, lc.dVBSum, []int{1, dModel}, lc.dVBR); err != nil {
			return nil, err
		}
		if _, err := engine.Add(ctx, dg.vB, lc.dVBR, dg.vB); err != nil {
			return nil, err
		}

		// dNormed1 = dQ @ qW^T + dK @ kW^T + dV @ vW^T
		if err := matMulInto(ctx, engine, lc.dQT, lt.qWT, lc.dN1q); err != nil {
			return nil, err
		}
		if err := matMulInto(ctx, engine, lc.dKT, lt.kWT, lc.dN1k); err != nil {
			return nil, err
		}
		if err := matMulInto(ctx, engine, lc.dVT, lt.vWT, lc.dN1v); err != nil {
			return nil, err
		}
		if _, err := engine.Add(ctx, lc.dN1q, lc.dN1k, lc.dN1Sum1); err != nil {
			return nil, err
		}
		if _, err := engine.Add(ctx, lc.dN1Sum1, lc.dN1v, lc.dNormed1); err != nil {
			return nil, err
		}

		// LayerNorm1 backward.
		ln1Bufs := &lnBwdBufs{
			normVal:      lc.lnbNormVal,
			dScaleBatch:  lc.lnbDScaleBatch,
			dScaleSum:    lc.lnbDScaleSum,
			dScaleSumR:   lc.lnbDScaleSumR,
			dBiasSum:     lc.lnbDBiasSum,
			dBiasSumR:    lc.lnbDBiasSumR,
			dNorm:        lc.lnbDNorm,
			dNormCent:    lc.lnbDNormCent,
			dotScaleGrad: lc.lnbDotScaleGrad,
			dotMeanGrad:  lc.lnbDotMeanGrad,
			invStdSq:     lc.lnbInvStdSq,
			term:         lc.lnbTerm,
			correction:   lc.lnbCorrection,
			inner:        lc.lnbInner,
			dInput:       lc.lnbDInput,
		}
		dLN1Input, _, _, err := layerNormBackwardWithEngine(ctx, engine, lc.dNormed1, lc.centered1, lc.invStd1,
			layer.norm1, dg.norm1, dg.bias1, ln1Bufs, totalRows, dModel)
		if err != nil {
			return nil, err
		}
		// Final dX for this layer = dLN1Input + dAttnProjOut, written into lc.dXOut.
		if _, err := engine.Add(ctx, dLN1Input, lc.dAttnProjOut, lc.dXOut); err != nil {
			return nil, err
		}
		dX = lc.dXOut
	}

	return dX, nil
}

// encoderBackwardF64 runs the PatchTST transformer encoder backward pass
// using functional backward ops. It mirrors encoderBackward but operates on
// float64 slices, enabling the CPU training path (BackwardSample) to share
// a single implementation with any future float64-based callers.
//
// dX: [numPatches][dModel] upstream gradient flowing into the encoder output.
// layers: encoder layer parameters.
// dLayers: gradient accumulators (same layout as layers).
// layerCaches: per-layer cached activations from the forward pass.
// numPatches, dModel, nHeads, headDim, ffnDim: architecture constants.
// Returns: dX gradient with respect to encoder input [numPatches][dModel].
func encoderBackwardF64(
	dX [][]float64,
	layers []encoderLayerF64,
	dLayers []encoderLayerF64Grad,
	layerCaches []encoderLayerCache,
	numPatches, dModel, nHeads, headDim, ffnDim int,
) [][]float64 {
	ctx := context.Background()
	eng := cpuEngine64
	ops := numeric.Float64Ops{}

	for li := len(layers) - 1; li >= 0; li-- {
		layer := &layers[li]
		lc := &layerCaches[li]
		dg := &dLayers[li]

		// Residual 2 backward: dX flows to both ffn output and the input after attn.
		dFFN2Out := copyMatrix(dX)

		// FFN backward via functional.MLPBackward.
		// Weight layout: existing flat weights are [in, out] row-major,
		// but LinearBackward convention is y = x @ W^T + b with W = [out, in].
		// So we create the tensor as [in, out] and transpose to [out, in].
		dFFN2OutT := mat64ToTensor(dFFN2Out, numPatches, dModel)
		normed2T := mat64ToTensor(lc.normed2, numPatches, dModel)
		ffn1PreActT := mat64ToTensor(lc.ffn1PreAct, numPatches, ffnDim)
		ffn1OutT := mat64ToTensor(lc.ffn1Out, numPatches, ffnDim)

		// ffn1W is [dModel*ffnDim] stored as [dModel, ffnDim] row-major.
		// LinearBackward wants [out=ffnDim, in=dModel], so transpose.
		w1Raw, _ := tensor.New[float64]([]int{dModel, ffnDim}, layer.ffn1W)
		w1T, _ := eng.Transpose(ctx, w1Raw, []int{1, 0})
		b1T, _ := tensor.New[float64]([]int{ffnDim}, layer.ffn1B)

		// ffn2W is [ffnDim*dModel] stored as [ffnDim, dModel] row-major.
		// LinearBackward wants [out=dModel, in=ffnDim], so transpose.
		w2Raw, _ := tensor.New[float64]([]int{ffnDim, dModel}, layer.ffn2W)
		w2T, _ := eng.Transpose(ctx, w2Raw, []int{1, 0})
		b2T, _ := tensor.New[float64]([]int{dModel}, layer.ffn2B)

		dNormed2T, dW1T, dB1T, dW2T, dB2T, _ := functional.MLPBackward(
			ctx, eng, ops, dFFN2OutT, normed2T, w1T, b1T, w2T, b2T,
			ffn1PreActT, ffn1OutT, "gelu")

		// Transpose weight gradients back to original layout and accumulate.
		// dW1T is [ffnDim, dModel], need [dModel, ffnDim] for accumulation.
		dW1Raw, _ := eng.Transpose(ctx, dW1T, []int{1, 0})
		accumFlat(dg.ffn1W, dW1Raw.Data())
		accumFlat(dg.ffn1B, dB1T.Data())
		// dW2T is [dModel, ffnDim], need [ffnDim, dModel] for accumulation.
		dW2Raw, _ := eng.Transpose(ctx, dW2T, []int{1, 0})
		accumFlat(dg.ffn2W, dW2Raw.Data())
		accumFlat(dg.ffn2B, dB2T.Data())

		// LayerNorm2 backward via functional.LayerNormBackward.
		xBeforeNorm2T := mat64ToTensor(lc.xBeforeNorm2, numPatches, dModel)
		norm2T, _ := tensor.New[float64]([]int{1, dModel}, layer.norm2)
		dXAfterAttnT, dScale2T, dBias2T, _ := functional.LayerNormBackward(
			ctx, eng, dNormed2T, xBeforeNorm2T, norm2T, 1e-5)
		accumFlat(dg.norm2, dScale2T.Data())
		accumFlat(dg.bias2, dBias2T.Data())

		// Add residual gradient from FFN path.
		dXResidT := mat64ToTensor(dX, numPatches, dModel)
		dXAfterAttnT, _ = eng.Add(ctx, dXAfterAttnT, dXResidT)

		// Residual 1 backward: dXAfterAttn flows to both attention output and layer input.
		dXAfterAttn := tensorToMat64(dXAfterAttnT, numPatches, dModel)
		dAttnProjOut := copyMatrix(dXAfterAttn)

		// oProj backward via functional.LinearBackward.
		dAttnProjOutT := mat64ToTensor(dAttnProjOut, numPatches, dModel)
		attnOutT := mat64ToTensor(lc.attnOut, numPatches, dModel)
		// oW is [dModel*dModel] stored as [dModel, dModel] row-major.
		// LinearBackward wants [out=dModel, in=dModel] — same shape, but transpose for correctness.
		oWRaw, _ := tensor.New[float64]([]int{dModel, dModel}, layer.oW)
		oWT, _ := eng.Transpose(ctx, oWRaw, []int{1, 0})
		dAttnOutT, dOWT, dOBT, _ := functional.LinearBackward(ctx, eng, dAttnProjOutT, attnOutT, oWT)
		// Transpose dOW back to original layout.
		dOWRaw, _ := eng.Transpose(ctx, dOWT, []int{1, 0})
		accumFlat(dg.oW, dOWRaw.Data())
		accumFlat(dg.oB, dOBT.Data())

		// Attention backward via functional.MultiHeadAttentionBackward.
		qT := mat64ToTensor(lc.q, numPatches, dModel)
		kT := mat64ToTensor(lc.k, numPatches, dModel)
		vT := mat64ToTensor(lc.v, numPatches, dModel)
		dQT, dKT, dVT, _ := functional.MultiHeadAttentionBackward(ctx, eng, ops, dAttnOutT, qT, kT, vT, nHeads)

		// Q/K/V projection backward via functional.LinearBackward.
		normed1T := mat64ToTensor(lc.normed1, numPatches, dModel)

		// qW is [dModel*dModel] stored as [dModel, dModel] row-major → transpose to [dModel, dModel].
		qWRaw, _ := tensor.New[float64]([]int{dModel, dModel}, layer.qW)
		qWTensor, _ := eng.Transpose(ctx, qWRaw, []int{1, 0})
		dNormed1Q, dQW, dQB, _ := functional.LinearBackward(ctx, eng, dQT, normed1T, qWTensor)

		kWRaw, _ := tensor.New[float64]([]int{dModel, dModel}, layer.kW)
		kWTensor, _ := eng.Transpose(ctx, kWRaw, []int{1, 0})
		dNormed1K, dKW, dKB, _ := functional.LinearBackward(ctx, eng, dKT, normed1T, kWTensor)

		vWRaw, _ := tensor.New[float64]([]int{dModel, dModel}, layer.vW)
		vWTensor, _ := eng.Transpose(ctx, vWRaw, []int{1, 0})
		dNormed1V, dVW, dVB, _ := functional.LinearBackward(ctx, eng, dVT, normed1T, vWTensor)

		// Accumulate Q/K/V weight/bias gradients.
		dQWRaw, _ := eng.Transpose(ctx, dQW, []int{1, 0})
		accumFlat(dg.qW, dQWRaw.Data())
		accumFlat(dg.qB, dQB.Data())
		dKWRaw, _ := eng.Transpose(ctx, dKW, []int{1, 0})
		accumFlat(dg.kW, dKWRaw.Data())
		accumFlat(dg.kB, dKB.Data())
		dVWRaw, _ := eng.Transpose(ctx, dVW, []int{1, 0})
		accumFlat(dg.vW, dVWRaw.Data())
		accumFlat(dg.vB, dVB.Data())

		// Sum dNormed1 from Q/K/V paths.
		dNormed1T, _ := eng.Add(ctx, dNormed1Q, dNormed1K)
		dNormed1T, _ = eng.Add(ctx, dNormed1T, dNormed1V)

		// LayerNorm1 backward via functional.LayerNormBackward.
		xBeforeNorm1T := mat64ToTensor(lc.xBeforeNorm1, numPatches, dModel)
		norm1T, _ := tensor.New[float64]([]int{1, dModel}, layer.norm1)
		dLayerInputT, dScale1T, dBias1T, _ := functional.LayerNormBackward(
			ctx, eng, dNormed1T, xBeforeNorm1T, norm1T, 1e-5)
		accumFlat(dg.norm1, dScale1T.Data())
		accumFlat(dg.bias1, dBias1T.Data())

		// Add residual gradient from attention path.
		dXAfterAttnT2 := mat64ToTensor(dXAfterAttn, numPatches, dModel)
		dLayerInputT, _ = eng.Add(ctx, dLayerInputT, dXAfterAttnT2)

		dX = tensorToMat64(dLayerInputT, numPatches, dModel)
	}

	return dX
}

// mat64ToTensor converts a [][]float64 matrix to a 2D tensor [rows, cols].
func mat64ToTensor(m [][]float64, rows, cols int) *tensor.TensorNumeric[float64] {
	flat := make([]float64, rows*cols)
	for i := 0; i < rows; i++ {
		copy(flat[i*cols:], m[i])
	}
	t, _ := tensor.New[float64]([]int{rows, cols}, flat)
	return t
}

// tensorToMat64 converts a 2D tensor [rows, cols] to a [][]float64 matrix.
func tensorToMat64(t *tensor.TensorNumeric[float64], rows, cols int) [][]float64 {
	data := t.Data()
	m := make([][]float64, rows)
	for i := 0; i < rows; i++ {
		m[i] = make([]float64, cols)
		copy(m[i], data[i*cols:(i+1)*cols])
	}
	return m
}

// accumFlat adds src element-wise into dst.
func accumFlat(dst, src []float64) {
	for i := range src {
		dst[i] += src[i]
	}
}
