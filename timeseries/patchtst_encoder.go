package timeseries

import (
	"context"
	"math"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/tensor"
)

// layerTransposes holds pre-computed weight transposes for a single encoder layer.
// These are constant within a batch and shared between forward and backward.
type layerTransposes struct {
	qWT, kWT, vWT, oWT *tensor.TensorNumeric[float32]
	ffn1WT, ffn2WT     *tensor.TensorNumeric[float32]
}

// layerNormForwardWithEngine performs layer norm using engine ops.
// x: [rows, dModel], scale: [1, dModel], bias: [1, dModel].
// Returns normed [rows, dModel], centered [rows, dModel], invStd [rows, 1].
func layerNormForwardWithEngine(ctx context.Context, engine compute.Engine[float32], x, scale, bias *tensor.TensorNumeric[float32], rows, dModel int) (normed, centered, invStd *tensor.TensorNumeric[float32], err error) {
	invD := float32(1.0) / float32(dModel)

	// mean = Sum(x, axis=1, keepDims=true) * (1/dModel)  -> [rows, 1]
	mean, err := engine.Sum(ctx, x, 1, true)
	if err != nil {
		return nil, nil, nil, err
	}
	mean, err = engine.MulScalar(ctx, mean, invD)
	if err != nil {
		return nil, nil, nil, err
	}

	// centered = x - mean  (broadcast [rows,1] over [rows, dModel])
	centered, err = engine.Sub(ctx, x, mean)
	if err != nil {
		return nil, nil, nil, err
	}

	// variance = Sum(centered*centered, axis=1, keepDims=true) * (1/dModel)  -> [rows, 1]
	centSq, err := engine.Mul(ctx, centered, centered)
	if err != nil {
		return nil, nil, nil, err
	}
	variance, err := engine.Sum(ctx, centSq, 1, true)
	if err != nil {
		return nil, nil, nil, err
	}
	variance, err = engine.MulScalar(ctx, variance, invD)
	if err != nil {
		return nil, nil, nil, err
	}

	// invStd = 1 / sqrt(variance + eps)  -> [rows, 1]
	varEps, err := engine.AddScalar(ctx, variance, 1e-5)
	if err != nil {
		return nil, nil, nil, err
	}
	stddev, err := engine.Sqrt(ctx, varEps)
	if err != nil {
		return nil, nil, nil, err
	}
	onesData := make([]float32, rows)
	for i := range onesData {
		onesData[i] = 1.0
	}
	ones, err := tensor.New[float32]([]int{rows, 1}, onesData)
	if err != nil {
		return nil, nil, nil, err
	}
	invStd, err = engine.Div(ctx, ones, stddev)
	if err != nil {
		return nil, nil, nil, err
	}

	// normed = centered * invStd * scale + bias  (broadcast [rows,1] and [1,dModel])
	normed, err = engine.Mul(ctx, centered, invStd)
	if err != nil {
		return nil, nil, nil, err
	}
	normed, err = engine.Mul(ctx, normed, scale)
	if err != nil {
		return nil, nil, nil, err
	}
	normed, err = engine.Add(ctx, normed, bias)
	if err != nil {
		return nil, nil, nil, err
	}

	return normed, centered, invStd, nil
}

// encoderForward runs the PatchTST transformer encoder layers using engine ops.
// x: [totalRows, dModel] input tensor (after patch embedding + pos embedding).
// bsC is batch*channels, numPatches is the sequence length per sample-channel.
// totalRows must equal bsC * numPatches.
// Returns: output [totalRows, dModel] and per-layer caches for backward.
func encoderForward(
	ctx context.Context,
	engine compute.Engine[float32],
	x *tensor.TensorNumeric[float32],
	layers []gpuEncoderLayer,
	bsC, numPatches, totalRows, dModel, nHeads, headDim, ffnDim int,
) (*tensor.TensorNumeric[float32], []gpuBatchLayerCache, error) {
	nLayers := len(layers)
	layerCaches := make([]gpuBatchLayerCache, nLayers)
	seq := numPatches
	bnh := bsC * nHeads

	var err error
	for li := 0; li < nLayers; li++ {
		layer := &layers[li]
		lc := &layerCaches[li]

		// Save layer input for residual backward.
		xCopy := make([]float32, len(x.Data()))
		copy(xCopy, x.Data())
		lc.xResidual, err = tensor.New[float32]([]int{totalRows, dModel}, xCopy)
		if err != nil {
			return nil, nil, err
		}

		// Layer norm 1 via engine ops (over all bsC*numPatches rows).
		lc.normed1, lc.centered1, lc.invStd1, err = layerNormForwardWithEngine(ctx, engine, x, layer.norm1, layer.bias1, totalRows, dModel)
		if err != nil {
			return nil, nil, err
		}

		// Q/K/V projections: ONE MatMul each for entire batch.
		// [bsC*numPatches, dModel] @ [dModel, dModel] = [bsC*numPatches, dModel]
		lc.q, err = engine.MatMul(ctx, lc.normed1, layer.qW)
		if err != nil {
			return nil, nil, err
		}
		lc.q, err = engine.Add(ctx, lc.q, layer.qB)
		if err != nil {
			return nil, nil, err
		}
		lc.k, err = engine.MatMul(ctx, lc.normed1, layer.kW)
		if err != nil {
			return nil, nil, err
		}
		lc.k, err = engine.Add(ctx, lc.k, layer.kB)
		if err != nil {
			return nil, nil, err
		}
		lc.v, err = engine.MatMul(ctx, lc.normed1, layer.vW)
		if err != nil {
			return nil, nil, err
		}
		lc.v, err = engine.Add(ctx, lc.v, layer.vB)
		if err != nil {
			return nil, nil, err
		}

		// Batched attention via engine ops: reshape Q/K/V to
		// [bsC*nHeads, numPatches, headDim] and use 3D batched MatMul.
		scale := float32(1.0 / math.Sqrt(float64(headDim)))

		// Reshape Q/K/V: [bsC*seq, dModel] -> [bsC, seq, nHeads, headDim]
		// -> transpose [0,2,1,3] -> [bsC, nHeads, seq, headDim]
		// -> reshape [bsC*nHeads, seq, headDim].
		q4d, err := engine.Reshape(ctx, lc.q, []int{bsC, seq, nHeads, headDim})
		if err != nil {
			return nil, nil, err
		}
		q4d, err = engine.Transpose(ctx, q4d, []int{0, 2, 1, 3})
		if err != nil {
			return nil, nil, err
		}
		qH, err := engine.Reshape(ctx, q4d, []int{bnh, seq, headDim})
		if err != nil {
			return nil, nil, err
		}

		k4d, err := engine.Reshape(ctx, lc.k, []int{bsC, seq, nHeads, headDim})
		if err != nil {
			return nil, nil, err
		}
		k4d, err = engine.Transpose(ctx, k4d, []int{0, 2, 1, 3})
		if err != nil {
			return nil, nil, err
		}
		kH, err := engine.Reshape(ctx, k4d, []int{bnh, seq, headDim})
		if err != nil {
			return nil, nil, err
		}

		v4d, err := engine.Reshape(ctx, lc.v, []int{bsC, seq, nHeads, headDim})
		if err != nil {
			return nil, nil, err
		}
		v4d, err = engine.Transpose(ctx, v4d, []int{0, 2, 1, 3})
		if err != nil {
			return nil, nil, err
		}
		vH, err := engine.Reshape(ctx, v4d, []int{bnh, seq, headDim})
		if err != nil {
			return nil, nil, err
		}

		// scores = Q @ K^T * scale: [bnh, seq, seq].
		kHT, err := engine.Transpose(ctx, kH, []int{0, 2, 1})
		if err != nil {
			return nil, nil, err
		}
		logits, err := engine.MatMul(ctx, qH, kHT)
		if err != nil {
			return nil, nil, err
		}
		logits, err = engine.MulScalar(ctx, logits, scale)
		if err != nil {
			return nil, nil, err
		}

		// softmax along last axis.
		lc.scoresTensor, err = engine.Softmax(ctx, logits, -1)
		if err != nil {
			return nil, nil, err
		}

		// attnOut = scores @ V: [bnh, seq, headDim].
		attnH, err := engine.MatMul(ctx, lc.scoresTensor, vH)
		if err != nil {
			return nil, nil, err
		}

		// Reshape back: [bnh, seq, headDim] -> [bsC, nHeads, seq, headDim]
		// -> transpose [0,2,1,3] -> [bsC, seq, nHeads, headDim]
		// -> reshape [bsC*seq, dModel].
		attnH, err = engine.Reshape(ctx, attnH, []int{bsC, nHeads, seq, headDim})
		if err != nil {
			return nil, nil, err
		}
		attnH, err = engine.Transpose(ctx, attnH, []int{0, 2, 1, 3})
		if err != nil {
			return nil, nil, err
		}
		lc.attnOut, err = engine.Reshape(ctx, attnH, []int{totalRows, dModel})
		if err != nil {
			return nil, nil, err
		}

		// Output projection: ONE MatMul for entire batch.
		// [bsC*numPatches, dModel] @ [dModel, dModel] = [bsC*numPatches, dModel]
		attnProj, err := engine.MatMul(ctx, lc.attnOut, layer.oW)
		if err != nil {
			return nil, nil, err
		}
		attnProj, err = engine.Add(ctx, attnProj, layer.oB)
		if err != nil {
			return nil, nil, err
		}

		// Residual 1.
		x, err = engine.Add(ctx, x, attnProj)
		if err != nil {
			return nil, nil, err
		}

		// Save x after attention for residual backward.
		xAfterCopy := make([]float32, len(x.Data()))
		copy(xAfterCopy, x.Data())
		lc.xAfterAttn, err = tensor.New[float32]([]int{totalRows, dModel}, xAfterCopy)
		if err != nil {
			return nil, nil, err
		}

		// Layer norm 2 via engine ops.
		lc.normed2, lc.centered2, lc.invStd2, err = layerNormForwardWithEngine(ctx, engine, x, layer.norm2, layer.bias2, totalRows, dModel)
		if err != nil {
			return nil, nil, err
		}

		// FFN1: ONE MatMul for entire batch.
		// [bsC*numPatches, dModel] @ [dModel, ffnDim] = [bsC*numPatches, ffnDim]
		lc.ffn1PreAct, err = engine.MatMul(ctx, lc.normed2, layer.ffn1W)
		if err != nil {
			return nil, nil, err
		}
		lc.ffn1PreAct, err = engine.Add(ctx, lc.ffn1PreAct, layer.ffn1B)
		if err != nil {
			return nil, nil, err
		}

		// GELU via engine ops: gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
		geluIn := lc.ffn1PreAct
		geluX3, err := engine.Mul(ctx, geluIn, geluIn)
		if err != nil {
			return nil, nil, err
		}
		geluX3, err = engine.Mul(ctx, geluX3, geluIn) // x^3
		if err != nil {
			return nil, nil, err
		}
		geluInner, err := engine.MulScalar(ctx, geluX3, float32(0.044715)) // 0.044715 * x^3
		if err != nil {
			return nil, nil, err
		}
		geluInner, err = engine.Add(ctx, geluIn, geluInner) // x + 0.044715*x^3
		if err != nil {
			return nil, nil, err
		}
		geluInner, err = engine.MulScalar(ctx, geluInner, float32(math.Sqrt(2.0/math.Pi))) // sqrt(2/pi) * (...)
		if err != nil {
			return nil, nil, err
		}
		lc.geluTanhVal, err = engine.Tanh(ctx, geluInner) // tanh(...)
		if err != nil {
			return nil, nil, err
		}
		onePlusTanh, err := engine.AddScalar(ctx, lc.geluTanhVal, float32(1.0)) // 1 + tanh(...)
		if err != nil {
			return nil, nil, err
		}
		lc.ffn1Out, err = engine.Mul(ctx, geluIn, onePlusTanh) // x * (1 + tanh(...))
		if err != nil {
			return nil, nil, err
		}
		lc.ffn1Out, err = engine.MulScalar(ctx, lc.ffn1Out, float32(0.5)) // 0.5 * x * (1 + tanh(...))
		if err != nil {
			return nil, nil, err
		}

		// FFN2: ONE MatMul for entire batch.
		// [bsC*numPatches, ffnDim] @ [ffnDim, dModel] = [bsC*numPatches, dModel]
		ffn2Out, err := engine.MatMul(ctx, lc.ffn1Out, layer.ffn2W)
		if err != nil {
			return nil, nil, err
		}
		ffn2Out, err = engine.Add(ctx, ffn2Out, layer.ffn2B)
		if err != nil {
			return nil, nil, err
		}

		// Residual 2.
		x, err = engine.Add(ctx, x, ffn2Out)
		if err != nil {
			return nil, nil, err
		}
	}

	return x, layerCaches, nil
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

		// FFN2 backward: ONE MatMul each.
		// dFFN2Out = dX (from residual 2).
		// dW += ffn1Out^T @ dFFN2Out : [ffnDim, totalRows] @ [totalRows, dModel]
		ffn1OutT, err := engine.Transpose(ctx, lc.ffn1Out, []int{1, 0})
		if err != nil {
			return nil, err
		}
		dFW, err := engine.MatMul(ctx, ffn1OutT, dX)
		if err != nil {
			return nil, err
		}
		dg.ffn2W, err = engine.Add(ctx, dg.ffn2W, dFW)
		if err != nil {
			return nil, err
		}
		dFB, err := engine.Sum(ctx, dX, 0, false)
		if err != nil {
			return nil, err
		}
		dFBR, err := engine.Reshape(ctx, dFB, []int{1, dModel})
		if err != nil {
			return nil, err
		}
		dg.ffn2B, err = engine.Add(ctx, dg.ffn2B, dFBR)
		if err != nil {
			return nil, err
		}
		// dFFN1Out = dFFN2Out @ ffn2W^T : [totalRows, dModel] @ [dModel, ffnDim]
		dFFN1Out, err := engine.MatMul(ctx, dX, lt.ffn2WT)
		if err != nil {
			return nil, err
		}

		// GELU backward via engine ops.
		// d/dx[gelu] = 0.5*(1+tanh) + 0.5*x*(1-tanh^2)*sqrt(2/pi)*(1+3*0.044715*x^2)
		geluX := lc.ffn1PreAct
		tanhVal := lc.geluTanhVal

		// term1 = 0.5 * (1 + tanh)
		term1, err := engine.AddScalar(ctx, tanhVal, float32(1.0))
		if err != nil {
			return nil, err
		}
		term1, err = engine.MulScalar(ctx, term1, float32(0.5))
		if err != nil {
			return nil, err
		}

		// sech^2 = 1 - tanh^2
		tanhSq, err := engine.Mul(ctx, tanhVal, tanhVal)
		if err != nil {
			return nil, err
		}
		sechSq, err := engine.MulScalar(ctx, tanhSq, float32(-1.0))
		if err != nil {
			return nil, err
		}
		sechSq, err = engine.AddScalar(ctx, sechSq, float32(1.0))
		if err != nil {
			return nil, err
		}

		// du/dx = sqrt(2/pi) * (1 + 3*0.044715*x^2)
		xSq, err := engine.Mul(ctx, geluX, geluX)
		if err != nil {
			return nil, err
		}
		dudx, err := engine.MulScalar(ctx, xSq, float32(3*0.044715))
		if err != nil {
			return nil, err
		}
		dudx, err = engine.AddScalar(ctx, dudx, float32(1.0))
		if err != nil {
			return nil, err
		}
		dudx, err = engine.MulScalar(ctx, dudx, float32(math.Sqrt(2.0/math.Pi)))
		if err != nil {
			return nil, err
		}

		// term2 = 0.5 * x * sech^2 * du/dx
		term2, err := engine.Mul(ctx, geluX, sechSq)
		if err != nil {
			return nil, err
		}
		term2, err = engine.Mul(ctx, term2, dudx)
		if err != nil {
			return nil, err
		}
		term2, err = engine.MulScalar(ctx, term2, float32(0.5))
		if err != nil {
			return nil, err
		}

		// geluDeriv = term1 + term2
		geluDeriv, err := engine.Add(ctx, term1, term2)
		if err != nil {
			return nil, err
		}
		dFFN1PreAct, err := engine.Mul(ctx, dFFN1Out, geluDeriv)
		if err != nil {
			return nil, err
		}

		// FFN1 backward: ONE MatMul each.
		// dW += normed2^T @ dFFN1PreAct
		normed2T, err := engine.Transpose(ctx, lc.normed2, []int{1, 0})
		if err != nil {
			return nil, err
		}
		dF1W, err := engine.MatMul(ctx, normed2T, dFFN1PreAct)
		if err != nil {
			return nil, err
		}
		dg.ffn1W, err = engine.Add(ctx, dg.ffn1W, dF1W)
		if err != nil {
			return nil, err
		}
		dF1B, err := engine.Sum(ctx, dFFN1PreAct, 0, false)
		if err != nil {
			return nil, err
		}
		dF1BR, err := engine.Reshape(ctx, dF1B, []int{1, ffnDim})
		if err != nil {
			return nil, err
		}
		dg.ffn1B, err = engine.Add(ctx, dg.ffn1B, dF1BR)
		if err != nil {
			return nil, err
		}
		// dNormed2 = dFFN1PreAct @ ffn1W^T
		dNormed2, err := engine.MatMul(ctx, dFFN1PreAct, lt.ffn1WT)
		if err != nil {
			return nil, err
		}

		// LayerNorm2 backward on CPU + residual add.
		dNormed2Data := matFromTensor(dNormed2, totalRows, dModel)
		centered2Mat := matFromTensor(lc.centered2, totalRows, dModel)
		invStd2Flat := lc.invStd2.Data() // [rows*1] flat slice
		dXAfterAttn := layerNormBackwardF32(dNormed2Data, centered2Mat, invStd2Flat,
			layer.norm2.Data(), dg.norm2.Data(), dg.bias2.Data(), dModel)
		// Add residual gradient from FFN path (dX flows through residual 2).
		dXData := dX.Data()
		for r := 0; r < totalRows; r++ {
			for j := 0; j < dModel; j++ {
				dXAfterAttn[r][j] += dXData[r*dModel+j]
			}
		}

		// oProj backward: ONE MatMul each.
		dAttnProjOut, err := tensorFromMat(dXAfterAttn, totalRows, dModel)
		if err != nil {
			return nil, err
		}
		// dW += attnOut^T @ dAttnProjOut
		attnOutT, err := engine.Transpose(ctx, lc.attnOut, []int{1, 0})
		if err != nil {
			return nil, err
		}
		dOW, err := engine.MatMul(ctx, attnOutT, dAttnProjOut)
		if err != nil {
			return nil, err
		}
		dg.oW, err = engine.Add(ctx, dg.oW, dOW)
		if err != nil {
			return nil, err
		}
		dOB, err := engine.Sum(ctx, dAttnProjOut, 0, false)
		if err != nil {
			return nil, err
		}
		dOBR, err := engine.Reshape(ctx, dOB, []int{1, dModel})
		if err != nil {
			return nil, err
		}
		dg.oB, err = engine.Add(ctx, dg.oB, dOBR)
		if err != nil {
			return nil, err
		}
		// dAttnOut = dAttnProjOut @ oW^T
		dAttnOutT, err := engine.MatMul(ctx, dAttnProjOut, lt.oWT)
		if err != nil {
			return nil, err
		}

		// Batched attention backward via engine ops.
		// Reshape dAttnOut to [bsC*nHeads, seq, headDim] (same layout as forward).
		attnScale := float32(1.0 / math.Sqrt(float64(headDim)))

		dAO4d, err := engine.Reshape(ctx, dAttnOutT, []int{bsC, seq, nHeads, headDim})
		if err != nil {
			return nil, err
		}
		dAO4d, err = engine.Transpose(ctx, dAO4d, []int{0, 2, 1, 3})
		if err != nil {
			return nil, err
		}
		dAttnOutH, err := engine.Reshape(ctx, dAO4d, []int{bnh, seq, headDim})
		if err != nil {
			return nil, err
		}

		// Reshape Q/K/V to [bsC*nHeads, seq, headDim].
		q4d, err := engine.Reshape(ctx, lc.q, []int{bsC, seq, nHeads, headDim})
		if err != nil {
			return nil, err
		}
		q4d, err = engine.Transpose(ctx, q4d, []int{0, 2, 1, 3})
		if err != nil {
			return nil, err
		}
		qH, err := engine.Reshape(ctx, q4d, []int{bnh, seq, headDim})
		if err != nil {
			return nil, err
		}

		k4d, err := engine.Reshape(ctx, lc.k, []int{bsC, seq, nHeads, headDim})
		if err != nil {
			return nil, err
		}
		k4d, err = engine.Transpose(ctx, k4d, []int{0, 2, 1, 3})
		if err != nil {
			return nil, err
		}
		kH, err := engine.Reshape(ctx, k4d, []int{bnh, seq, headDim})
		if err != nil {
			return nil, err
		}

		v4d, err := engine.Reshape(ctx, lc.v, []int{bsC, seq, nHeads, headDim})
		if err != nil {
			return nil, err
		}
		v4d, err = engine.Transpose(ctx, v4d, []int{0, 2, 1, 3})
		if err != nil {
			return nil, err
		}
		vH, err := engine.Reshape(ctx, v4d, []int{bnh, seq, headDim})
		if err != nil {
			return nil, err
		}

		// dScores = dAttnOut @ V^T: [bnh, seq, seq].
		vHT, err := engine.Transpose(ctx, vH, []int{0, 2, 1})
		if err != nil {
			return nil, err
		}
		dScores, err := engine.MatMul(ctx, dAttnOutH, vHT)
		if err != nil {
			return nil, err
		}

		// dV = scores^T @ dAttnOut: [bnh, seq, headDim].
		scoresT, err := engine.Transpose(ctx, lc.scoresTensor, []int{0, 2, 1})
		if err != nil {
			return nil, err
		}
		dVH, err := engine.MatMul(ctx, scoresT, dAttnOutH)
		if err != nil {
			return nil, err
		}

		// Softmax backward: dLogit = scores * (dScores - rowSum) * scale
		// where rowSum = sum(scores * dScores, axis=-1, keepdims=true).
		sDScores, err := engine.Mul(ctx, lc.scoresTensor, dScores)
		if err != nil {
			return nil, err
		}
		rowSum, err := engine.Sum(ctx, sDScores, 2, true)
		if err != nil {
			return nil, err
		}
		dLogits, err := engine.Sub(ctx, dScores, rowSum)
		if err != nil {
			return nil, err
		}
		dLogits, err = engine.Mul(ctx, lc.scoresTensor, dLogits)
		if err != nil {
			return nil, err
		}
		dLogits, err = engine.MulScalar(ctx, dLogits, attnScale)
		if err != nil {
			return nil, err
		}

		// dQ = dLogits @ K: [bnh, seq, headDim].
		dQH, err := engine.MatMul(ctx, dLogits, kH)
		if err != nil {
			return nil, err
		}

		// dK = dLogits^T @ Q: [bnh, seq, headDim].
		dLogitsT, err := engine.Transpose(ctx, dLogits, []int{0, 2, 1})
		if err != nil {
			return nil, err
		}
		dKH, err := engine.MatMul(ctx, dLogitsT, qH)
		if err != nil {
			return nil, err
		}

		// Reshape dQ/dK/dV back to [bsC*seq, dModel].
		dQH, err = engine.Reshape(ctx, dQH, []int{bsC, nHeads, seq, headDim})
		if err != nil {
			return nil, err
		}
		dQH, err = engine.Transpose(ctx, dQH, []int{0, 2, 1, 3})
		if err != nil {
			return nil, err
		}
		dQT, err := engine.Reshape(ctx, dQH, []int{totalRows, dModel})
		if err != nil {
			return nil, err
		}

		dKH, err = engine.Reshape(ctx, dKH, []int{bsC, nHeads, seq, headDim})
		if err != nil {
			return nil, err
		}
		dKH, err = engine.Transpose(ctx, dKH, []int{0, 2, 1, 3})
		if err != nil {
			return nil, err
		}
		dKT, err := engine.Reshape(ctx, dKH, []int{totalRows, dModel})
		if err != nil {
			return nil, err
		}

		dVH, err = engine.Reshape(ctx, dVH, []int{bsC, nHeads, seq, headDim})
		if err != nil {
			return nil, err
		}
		dVH, err = engine.Transpose(ctx, dVH, []int{0, 2, 1, 3})
		if err != nil {
			return nil, err
		}
		dVT, err := engine.Reshape(ctx, dVH, []int{totalRows, dModel})
		if err != nil {
			return nil, err
		}

		normed1T, err := engine.Transpose(ctx, lc.normed1, []int{1, 0})
		if err != nil {
			return nil, err
		}

		// dQW += normed1^T @ dQ
		dQW, err := engine.MatMul(ctx, normed1T, dQT)
		if err != nil {
			return nil, err
		}
		dg.qW, err = engine.Add(ctx, dg.qW, dQW)
		if err != nil {
			return nil, err
		}
		dQB, err := engine.Sum(ctx, dQT, 0, false)
		if err != nil {
			return nil, err
		}
		dQBR, err := engine.Reshape(ctx, dQB, []int{1, dModel})
		if err != nil {
			return nil, err
		}
		dg.qB, err = engine.Add(ctx, dg.qB, dQBR)
		if err != nil {
			return nil, err
		}

		dKW, err := engine.MatMul(ctx, normed1T, dKT)
		if err != nil {
			return nil, err
		}
		dg.kW, err = engine.Add(ctx, dg.kW, dKW)
		if err != nil {
			return nil, err
		}
		dKB, err := engine.Sum(ctx, dKT, 0, false)
		if err != nil {
			return nil, err
		}
		dKBR, err := engine.Reshape(ctx, dKB, []int{1, dModel})
		if err != nil {
			return nil, err
		}
		dg.kB, err = engine.Add(ctx, dg.kB, dKBR)
		if err != nil {
			return nil, err
		}

		dVW, err := engine.MatMul(ctx, normed1T, dVT)
		if err != nil {
			return nil, err
		}
		dg.vW, err = engine.Add(ctx, dg.vW, dVW)
		if err != nil {
			return nil, err
		}
		dVB, err := engine.Sum(ctx, dVT, 0, false)
		if err != nil {
			return nil, err
		}
		dVBR, err := engine.Reshape(ctx, dVB, []int{1, dModel})
		if err != nil {
			return nil, err
		}
		dg.vB, err = engine.Add(ctx, dg.vB, dVBR)
		if err != nil {
			return nil, err
		}

		// dNormed1 = dQ @ qW^T + dK @ kW^T + dV @ vW^T : ONE MatMul each.
		dN1q, err := engine.MatMul(ctx, dQT, lt.qWT)
		if err != nil {
			return nil, err
		}
		dN1k, err := engine.MatMul(ctx, dKT, lt.kWT)
		if err != nil {
			return nil, err
		}
		dN1v, err := engine.MatMul(ctx, dVT, lt.vWT)
		if err != nil {
			return nil, err
		}
		dNormed1, err := engine.Add(ctx, dN1q, dN1k)
		if err != nil {
			return nil, err
		}
		dNormed1, err = engine.Add(ctx, dNormed1, dN1v)
		if err != nil {
			return nil, err
		}

		// LayerNorm1 backward on CPU + residual add.
		dNormed1Data := matFromTensor(dNormed1, totalRows, dModel)
		centered1Mat := matFromTensor(lc.centered1, totalRows, dModel)
		invStd1Flat := lc.invStd1.Data() // [rows*1] flat slice
		dLayerInput := layerNormBackwardF32(dNormed1Data, centered1Mat, invStd1Flat,
			layer.norm1.Data(), dg.norm1.Data(), dg.bias1.Data(), dModel)
		// Add residual gradient from attention path.
		for r := 0; r < totalRows; r++ {
			for j := 0; j < dModel; j++ {
				dLayerInput[r][j] += dXAfterAttn[r][j]
			}
		}

		dX, err = tensorFromMat(dLayerInput, totalRows, dModel)
		if err != nil {
			return nil, err
		}
	}

	return dX, nil
}
