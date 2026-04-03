package timeseries

import (
	"context"
	"math"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/zerfoo/layers/functional"
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

// layerNormBackwardWithEngine computes the backward pass through layer norm using engine ops.
// dOut: [rows, dModel], centered: [rows, dModel], invStd: [rows, 1].
// scale: [1, dModel] (layer norm weight), dScale/dBias: [1, dModel] (gradient accumulators).
// Returns dInput [rows, dModel] and updated dScale, dBias.
func layerNormBackwardWithEngine(
	ctx context.Context,
	engine compute.Engine[float32],
	dOut, centered, invStd, scale, dScale, dBias *tensor.TensorNumeric[float32],
	rows, dModel int,
) (dInput, newDScale, newDBias *tensor.TensorNumeric[float32], err error) {
	invD := float32(1.0) / float32(rows)
	_ = invD

	// dScale += Sum(dOut * centered * invStd, axis=0)  -> [1, dModel]
	normVal, err := engine.Mul(ctx, centered, invStd) // [rows, dModel]
	if err != nil {
		return nil, nil, nil, err
	}
	dScaleBatch, err := engine.Mul(ctx, dOut, normVal) // [rows, dModel]
	if err != nil {
		return nil, nil, nil, err
	}
	dScaleSum, err := engine.Sum(ctx, dScaleBatch, 0, false) // [dModel]
	if err != nil {
		return nil, nil, nil, err
	}
	dScaleSum, err = engine.Reshape(ctx, dScaleSum, []int{1, dModel})
	if err != nil {
		return nil, nil, nil, err
	}
	newDScale, err = engine.Add(ctx, dScale, dScaleSum)
	if err != nil {
		return nil, nil, nil, err
	}

	// dBias += Sum(dOut, axis=0)  -> [1, dModel]
	dBiasSum, err := engine.Sum(ctx, dOut, 0, false) // [dModel]
	if err != nil {
		return nil, nil, nil, err
	}
	dBiasSum, err = engine.Reshape(ctx, dBiasSum, []int{1, dModel})
	if err != nil {
		return nil, nil, nil, err
	}
	newDBias, err = engine.Add(ctx, dBias, dBiasSum)
	if err != nil {
		return nil, nil, nil, err
	}

	// dNorm = dOut * scale  (broadcast [1, dModel] over [rows, dModel])
	dNorm, err := engine.Mul(ctx, dOut, scale) // [rows, dModel]
	if err != nil {
		return nil, nil, nil, err
	}

	// dotScaleGrad = Sum(dNorm * centered, axis=1, keepDims=true)  -> [rows, 1]
	dNormCent, err := engine.Mul(ctx, dNorm, centered) // [rows, dModel]
	if err != nil {
		return nil, nil, nil, err
	}
	dotScaleGrad, err := engine.Sum(ctx, dNormCent, 1, true) // [rows, 1]
	if err != nil {
		return nil, nil, nil, err
	}

	// dotMeanGrad = Sum(dNorm, axis=1, keepDims=true)  -> [rows, 1]
	dotMeanGrad, err := engine.Sum(ctx, dNorm, 1, true) // [rows, 1]
	if err != nil {
		return nil, nil, nil, err
	}

	// dInput = invStd * (dNorm - (dotMeanGrad + centered * invStd^2 * dotScaleGrad) / dModel)
	invStdSq, err := engine.Mul(ctx, invStd, invStd) // [rows, 1]
	if err != nil {
		return nil, nil, nil, err
	}
	term, err := engine.Mul(ctx, centered, invStdSq) // [rows, dModel] (broadcast [rows,1])
	if err != nil {
		return nil, nil, nil, err
	}
	term, err = engine.Mul(ctx, term, dotScaleGrad) // [rows, dModel] (broadcast [rows,1])
	if err != nil {
		return nil, nil, nil, err
	}
	correction, err := engine.Add(ctx, dotMeanGrad, term) // [rows, dModel] (broadcast)
	if err != nil {
		return nil, nil, nil, err
	}
	correction, err = engine.MulScalar(ctx, correction, 1.0/float32(dModel))
	if err != nil {
		return nil, nil, nil, err
	}
	inner, err := engine.Sub(ctx, dNorm, correction) // [rows, dModel]
	if err != nil {
		return nil, nil, nil, err
	}
	dInput, err = engine.Mul(ctx, invStd, inner) // [rows, dModel] (broadcast [rows,1])
	if err != nil {
		return nil, nil, nil, err
	}

	return dInput, newDScale, newDBias, nil
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

		// LayerNorm2 backward using engine ops + residual add.
		dLN2Input, newDNorm2, newDBias2, err := layerNormBackwardWithEngine(ctx, engine, dNormed2, lc.centered2, lc.invStd2,
			layer.norm2, dg.norm2, dg.bias2, totalRows, dModel)
		if err != nil {
			return nil, err
		}
		dg.norm2 = newDNorm2
		dg.bias2 = newDBias2
		// Add residual gradient from FFN path (dX flows through residual 2).
		dAttnProjOut, err := engine.Add(ctx, dLN2Input, dX)
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

		// LayerNorm1 backward using engine ops + residual add.
		dLN1Input, newDNorm1, newDBias1, err := layerNormBackwardWithEngine(ctx, engine, dNormed1, lc.centered1, lc.invStd1,
			layer.norm1, dg.norm1, dg.bias1, totalRows, dModel)
		if err != nil {
			return nil, err
		}
		dg.norm1 = newDNorm1
		dg.bias1 = newDBias1
		// Add residual gradient from attention path.
		dX, err = engine.Add(ctx, dLN1Input, dAttnProjOut)
		if err != nil {
			return nil, err
		}
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
