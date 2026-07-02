package timeseries

import (
	"context"
	"fmt"
	"unsafe"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/tensor"
)

// fusedLayerGPU holds persistent GPU pointers for one encoder layer's
// fused kernel buffers. Allocated once per layer on first use.
type fusedLayerGPU struct {
	weights [16]unsafe.Pointer // FEW_* indexed, persistent GPU copies of weights
	bufs    [16]unsafe.Pointer // FEB_* indexed, pre-allocated GPU scratch/cache
}

// fusedEncoderGPU holds all GPU resources for the fused encoder path.
type fusedEncoderGPU struct {
	layers    []fusedLayerGPU
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

// uploadTensor allocates GPU memory and copies the tensor's float32 data.
func uploadTensor(fep compute.FusedEncoderProvider, t *tensor.TensorNumeric[float32]) (unsafe.Pointer, error) {
	data := t.Data()
	ptr, err := fep.AllocDeviceFloat32(len(data))
	if err != nil {
		return nil, err
	}
	if err := fep.CopyToDevice(ptr, data); err != nil {
		return nil, err
	}
	return ptr, nil
}

// allocFusedLayerWeights uploads all 16 weight tensors for one layer to GPU.
func allocFusedLayerWeights(fep compute.FusedEncoderProvider, layer *gpuEncoderLayer) ([16]unsafe.Pointer, error) {
	tensors := [16]*tensor.TensorNumeric[float32]{
		layer.qW, layer.qB, layer.kW, layer.kB,
		layer.vW, layer.vB, layer.oW, layer.oB,
		layer.ffn1W, layer.ffn1B, layer.ffn2W, layer.ffn2B,
		layer.norm1, layer.bias1, layer.norm2, layer.bias2,
	}
	var ptrs [16]unsafe.Pointer
	for i, t := range tensors {
		// Check if already GPU-backed (from a previous per-op run).
		if p := tensorDevPtr(t); p != nil {
			ptrs[i] = p
			continue
		}
		// Upload to GPU.
		p, err := uploadTensor(fep, t)
		if err != nil {
			return ptrs, fmt.Errorf("weight[%d]: %w", i, err)
		}
		ptrs[i] = p
	}
	return ptrs, nil
}

// allocFusedLayerBufs allocates the 16 FEB_* GPU scratch/cache buffers for one layer.
func allocFusedLayerBufs(fep compute.FusedEncoderProvider, totalRows, dModel, nHeads, headDim, ffnDim, bsC, numPatches int) ([16]unsafe.Pointer, error) {
	bnh := bsC * nHeads
	sizes := [16]int{
		/* FEB_NORMED1    */ totalRows * dModel,
		/* FEB_LN1_INVSTD */ totalRows,
		/* FEB_Q          */ totalRows * dModel,
		/* FEB_K          */ totalRows * dModel,
		/* FEB_V          */ totalRows * dModel,
		/* FEB_QH         */ bnh * numPatches * headDim,
		/* FEB_KH         */ bnh * numPatches * headDim,
		/* FEB_VH         */ bnh * numPatches * headDim,
		/* FEB_ATTN_SCORES*/ bnh * numPatches * numPatches,
		/* FEB_ATTN_OUT_H */ bnh * numPatches * headDim,
		/* FEB_ATTN_OUT   */ totalRows * dModel,
		/* FEB_X_RES1     */ totalRows * dModel,
		/* FEB_NORMED2    */ totalRows * dModel,
		/* FEB_LN2_INVSTD */ totalRows,
		/* FEB_FFN1_PRE   */ totalRows * ffnDim,
		/* FEB_FFN1_OUT   */ totalRows * ffnDim,
	}
	var ptrs [16]unsafe.Pointer
	for i, n := range sizes {
		p, err := fep.AllocDeviceFloat32(n)
		if err != nil {
			return ptrs, fmt.Errorf("buf[%d] (%d floats): %w", i, n, err)
		}
		ptrs[i] = p
	}
	return ptrs, nil
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
	fusedGPU *fusedEncoderGPU,
	bsC, numPatches, totalRows, dModel, nHeads, headDim, ffnDim int,
) (*tensor.TensorNumeric[float32], bool, error) {
	if fusedGPU == nil {
		return nil, false, nil // no persistent state (e.g., inference path)
	}
	fep, ok := engine.(compute.FusedEncoderProvider)
	if !ok || !fep.FusedEncoderAvailable() {
		return nil, false, nil
	}

	nLayers := len(layers)

	// Allocate persistent GPU resources on first call.
	if !fusedGPU.allocated {
		fusedGPU.layers = make([]fusedLayerGPU, nLayers)
		for li := 0; li < nLayers; li++ {
			var err error
			fusedGPU.layers[li].weights, err = allocFusedLayerWeights(fep, &layers[li])
			if err != nil {
				return nil, false, fmt.Errorf("layer %d weights: %w", li, err)
			}
			fusedGPU.layers[li].bufs, err = allocFusedLayerBufs(fep, totalRows, dModel, nHeads, headDim, ffnDim, bsC, numPatches)
			if err != nil {
				return nil, false, fmt.Errorf("layer %d bufs: %w", li, err)
			}
		}
		fusedGPU.allocated = true
	}

	// Ensure input is on GPU.
	inputPtr := tensorDevPtr(x)
	if inputPtr == nil {
		// Upload input to GPU.
		var err error
		inputPtr, err = uploadTensor(fep, x)
		if err != nil {
			return nil, false, fmt.Errorf("upload input: %w", err)
		}
	}

	// Allocate output buffer (reused across layers, final layer output returned).
	outputPtr, err := fep.AllocDeviceFloat32(totalRows * dModel)
	if err != nil {
		return nil, false, fmt.Errorf("alloc output: %w", err)
	}

	for li := 0; li < nLayers; li++ {
		fl := &fusedGPU.layers[li]

		if err := fep.FusedEncoderForward(&fl.weights, &fl.bufs, inputPtr, outputPtr,
			totalRows, dModel, nHeads, headDim, ffnDim, bsC, numPatches); err != nil {
			return nil, false, fmt.Errorf("fused forward layer %d: %w", li, err)
		}

		// Output becomes input for next layer.
		inputPtr, outputPtr = outputPtr, inputPtr
	}

	// After the loop, inputPtr holds the final output (due to the swap).
	// Wrap it as a non-owning GPU tensor view for the caller.
	finalPtr := inputPtr
	gs := tensor.NewGPUStorageViewFromPtr[float32](finalPtr, totalRows*dModel, 0)
	result, err := tensor.New[float32]([]int{totalRows, dModel}, make([]float32, totalRows*dModel))
	if err != nil {
		return nil, false, fmt.Errorf("wrap output: %w", err)
	}
	result.SetStorage(gs)
	return result, true, nil
}

// buildGradPtrs maps gradient accumulator tensors to the FEG_* array.
func buildGradPtrs(grad *gpuEncoderLayer) [16]unsafe.Pointer {
	return [16]unsafe.Pointer{
		tensorDevPtr(grad.qW), tensorDevPtr(grad.qB),
		tensorDevPtr(grad.kW), tensorDevPtr(grad.kB),
		tensorDevPtr(grad.vW), tensorDevPtr(grad.vB),
		tensorDevPtr(grad.oW), tensorDevPtr(grad.oB),
		tensorDevPtr(grad.ffn1W), tensorDevPtr(grad.ffn1B),
		tensorDevPtr(grad.ffn2W), tensorDevPtr(grad.ffn2B),
		tensorDevPtr(grad.norm1), tensorDevPtr(grad.bias1),
		tensorDevPtr(grad.norm2), tensorDevPtr(grad.bias2),
	}
}

// buildWeightTransposePtrs maps pre-transposed weights to the FEWT_* array.
func buildWeightTransposePtrs(lwt *layerTransposes) [6]unsafe.Pointer {
	return [6]unsafe.Pointer{
		tensorDevPtr(lwt.qWT), tensorDevPtr(lwt.kWT),
		tensorDevPtr(lwt.vWT), tensorDevPtr(lwt.oWT),
		tensorDevPtr(lwt.ffn1WT), tensorDevPtr(lwt.ffn2WT),
	}
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
	// Backward fused path deferred — needs dedicated FEBB_* scratch allocation
	// and mapping of gradient accumulators. Fall back to per-op path.
	_, _ = engine, dX
	return nil, false, nil
}
