package timeseries

import (
	"context"
	"fmt"
	"unsafe"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/tensor"
)

// fusedEncoderLayerBufs holds pre-allocated GPU buffers for the fused encoder
// kernel. These are indexed by the FEB_* constants from the kernel headers.
type fusedEncoderLayerBufs struct {
	ptrs      [16]unsafe.Pointer
	allocated bool
}

// tensorDevPtr extracts the raw GPU device pointer from a tensor.
// Returns nil if the tensor is nil or not GPU-backed.
func tensorDevPtr(t *tensor.TensorNumeric[float32]) unsafe.Pointer {
	if t == nil {
		return nil
	}
	gs, ok := t.GetStorage().(*tensor.GPUStorage[float32])
	if !ok {
		return nil
	}
	return gs.Ptr()
}

// buildWeightPtrs builds the FEW_* indexed weight pointer array from a gpuEncoderLayer.
func buildWeightPtrs(layer *gpuEncoderLayer) [16]unsafe.Pointer {
	return [16]unsafe.Pointer{
		/* FEW_QW     = 0  */ tensorDevPtr(layer.qW),
		/* FEW_QB     = 1  */ tensorDevPtr(layer.qB),
		/* FEW_KW     = 2  */ tensorDevPtr(layer.kW),
		/* FEW_KB     = 3  */ tensorDevPtr(layer.kB),
		/* FEW_VW     = 4  */ tensorDevPtr(layer.vW),
		/* FEW_VB     = 5  */ tensorDevPtr(layer.vB),
		/* FEW_OW     = 6  */ tensorDevPtr(layer.oW),
		/* FEW_OB     = 7  */ tensorDevPtr(layer.oB),
		/* FEW_FFN1W  = 8  */ tensorDevPtr(layer.ffn1W),
		/* FEW_FFN1B  = 9  */ tensorDevPtr(layer.ffn1B),
		/* FEW_FFN2W  = 10 */ tensorDevPtr(layer.ffn2W),
		/* FEW_FFN2B  = 11 */ tensorDevPtr(layer.ffn2B),
		/* FEW_NORM1W = 12 */ tensorDevPtr(layer.norm1),
		/* FEW_NORM1B = 13 */ tensorDevPtr(layer.bias1),
		/* FEW_NORM2W = 14 */ tensorDevPtr(layer.norm2),
		/* FEW_NORM2B = 15 */ tensorDevPtr(layer.bias2),
	}
}

// buildFwdCachePtrs maps existing gpuBatchLayerCache tensors to the FEB_* array.
func buildFwdCachePtrs(lc *gpuBatchLayerCache) [16]unsafe.Pointer {
	return [16]unsafe.Pointer{
		/* FEB_NORMED1    = 0  */ tensorDevPtr(lc.normed1),
		/* FEB_LN1_INVSTD = 1  */ tensorDevPtr(lc.invStd1),
		/* FEB_Q          = 2  */ tensorDevPtr(lc.q),
		/* FEB_K          = 3  */ tensorDevPtr(lc.k),
		/* FEB_V          = 4  */ tensorDevPtr(lc.v),
		/* FEB_QH         = 5  */ tensorDevPtr(lc.qH),
		/* FEB_KH         = 6  */ tensorDevPtr(lc.kH),
		/* FEB_VH         = 7  */ tensorDevPtr(lc.vH),
		/* FEB_ATTN_SCORES = 8 */ tensorDevPtr(lc.scoresTensor),
		/* FEB_ATTN_OUT_H = 9  */ tensorDevPtr(lc.attnH),
		/* FEB_ATTN_OUT   = 10 */ tensorDevPtr(lc.attnOut),
		/* FEB_X_RES1     = 11 */ tensorDevPtr(lc.xAfterRes1),
		/* FEB_NORMED2    = 12 */ tensorDevPtr(lc.normed2),
		/* FEB_LN2_INVSTD = 13 */ tensorDevPtr(lc.invStd2),
		/* FEB_FFN1_PRE   = 14 */ tensorDevPtr(lc.ffn1PreAct),
		/* FEB_FFN1_OUT   = 15 */ tensorDevPtr(lc.ffn1Out),
	}
}

// buildGradPtrs maps gradient accumulator tensors to the FEG_* array.
func buildGradPtrs(grad *gpuEncoderLayer) [16]unsafe.Pointer {
	return [16]unsafe.Pointer{
		/* FEG_DQW     = 0  */ tensorDevPtr(grad.qW),
		/* FEG_DQB     = 1  */ tensorDevPtr(grad.qB),
		/* FEG_DKW     = 2  */ tensorDevPtr(grad.kW),
		/* FEG_DKB     = 3  */ tensorDevPtr(grad.kB),
		/* FEG_DVW     = 4  */ tensorDevPtr(grad.vW),
		/* FEG_DVB     = 5  */ tensorDevPtr(grad.vB),
		/* FEG_DOW     = 6  */ tensorDevPtr(grad.oW),
		/* FEG_DOB     = 7  */ tensorDevPtr(grad.oB),
		/* FEG_DFFN1W  = 8  */ tensorDevPtr(grad.ffn1W),
		/* FEG_DFFN1B  = 9  */ tensorDevPtr(grad.ffn1B),
		/* FEG_DFFN2W  = 10 */ tensorDevPtr(grad.ffn2W),
		/* FEG_DFFN2B  = 11 */ tensorDevPtr(grad.ffn2B),
		/* FEG_DNORM1W = 12 */ tensorDevPtr(grad.norm1),
		/* FEG_DNORM1B = 13 */ tensorDevPtr(grad.bias1),
		/* FEG_DNORM2W = 14 */ tensorDevPtr(grad.norm2),
		/* FEG_DNORM2B = 15 */ tensorDevPtr(grad.bias2),
	}
}

// buildWeightTransposePtrs maps pre-transposed weights to the FEWT_* array.
func buildWeightTransposePtrs(lwt *layerTransposes) [6]unsafe.Pointer {
	return [6]unsafe.Pointer{
		/* FEWT_QWT    = 0 */ tensorDevPtr(lwt.qWT),
		/* FEWT_KWT    = 1 */ tensorDevPtr(lwt.kWT),
		/* FEWT_VWT    = 2 */ tensorDevPtr(lwt.vWT),
		/* FEWT_OWT    = 3 */ tensorDevPtr(lwt.oWT),
		/* FEWT_FFN1WT = 4 */ tensorDevPtr(lwt.ffn1WT),
		/* FEWT_FFN2WT = 5 */ tensorDevPtr(lwt.ffn2WT),
	}
}

// fusedEncoderForward attempts to run the fused encoder forward path.
// Returns (result, true, nil) if the fused kernel was used successfully.
// Returns (nil, false, nil) if the fused kernel is not available (caller should fall back).
// Returns (nil, false, err) on error.
func fusedEncoderForward(
	_ context.Context,
	engine compute.Engine[float32],
	x *tensor.TensorNumeric[float32],
	layers []gpuEncoderLayer,
	layerCaches []gpuBatchLayerCache,
	bsC, numPatches, totalRows, dModel, nHeads, headDim, ffnDim int,
) (*tensor.TensorNumeric[float32], bool, error) {
	fep, ok := engine.(compute.FusedEncoderProvider)
	if !ok || !fep.FusedEncoderAvailable() {
		return nil, false, nil
	}

	nLayers := len(layers)
	inputPtr := tensorDevPtr(x)
	if inputPtr == nil {
		return nil, false, nil // not GPU-backed, fall back
	}

	for li := 0; li < nLayers; li++ {
		layer := &layers[li]
		lc := &layerCaches[li]

		// Ensure cache buffers are allocated (reuse existing allocator).
		if err := allocLayerCacheBuffers(lc, bsC, numPatches, totalRows, dModel, nHeads, headDim, ffnDim); err != nil {
			return nil, false, err
		}

		weights := buildWeightPtrs(layer)
		bufs := buildFwdCachePtrs(lc)

		// Check all pointers are valid (all tensors must be GPU-backed).
		for i := 0; i < 16; i++ {
			if weights[i] == nil {
				return nil, false, fmt.Errorf("fusedEncoderForward: weight[%d] not on GPU", i)
			}
		}
		for i := 0; i < 16; i++ {
			if bufs[i] == nil {
				return nil, false, fmt.Errorf("fusedEncoderForward: buffer[%d] not on GPU", i)
			}
		}

		// Determine output pointer. For the last layer, use x's storage
		// (the caller expects the result in the returned tensor).
		// For intermediate layers, use xAfterRes2 as output, then make it
		// the next layer's input.
		var outputPtr unsafe.Pointer
		if lc.xAfterRes2 != nil {
			outputPtr = tensorDevPtr(lc.xAfterRes2)
		} else {
			// Fall back if xAfterRes2 not allocated
			return nil, false, nil
		}

		if err := fep.FusedEncoderForward(&weights, &bufs, inputPtr, outputPtr,
			totalRows, dModel, nHeads, headDim, ffnDim, bsC, numPatches); err != nil {
			return nil, false, fmt.Errorf("fusedEncoderForward layer %d: %w", li, err)
		}

		// Output becomes input for next layer.
		inputPtr = outputPtr
	}

	// Return the last layer's output tensor.
	return layerCaches[nLayers-1].xAfterRes2, true, nil
}

// fusedEncoderBackward attempts to run the fused encoder backward path.
// Returns (dInput, true, nil) if successful, (nil, false, nil) to fall back.
func fusedEncoderBackward(
	_ context.Context,
	engine compute.Engine[float32],
	dX *tensor.TensorNumeric[float32],
	layers []gpuEncoderLayer,
	grads []gpuEncoderLayer,
	layerCaches []gpuBatchLayerCache,
	lwts []layerTransposes,
	input *tensor.TensorNumeric[float32],
	bsC, numPatches, totalRows, dModel, nHeads, headDim, ffnDim int,
) (*tensor.TensorNumeric[float32], bool, error) {
	fep, ok := engine.(compute.FusedEncoderProvider)
	if !ok || !fep.FusedEncoderAvailable() {
		return nil, false, nil
	}

	nLayers := len(layers)
	dXPtr := tensorDevPtr(dX)
	if dXPtr == nil {
		return nil, false, nil
	}

	// Process layers in reverse order.
	for li := nLayers - 1; li >= 0; li-- {
		layer := &layers[li]
		grad := &grads[li]
		lc := &layerCaches[li]
		lwt := &lwts[li]

		// Backward wiring deferred to DGX validation — the fused backward
		// kernel requires dedicated scratch buffers (FEBB_*) that are not
		// yet allocated. Fall back to per-op path for now.
		_, _, _, _ = layer, grad, lc, lwt
		return nil, false, nil
	}

	return nil, false, nil
}
