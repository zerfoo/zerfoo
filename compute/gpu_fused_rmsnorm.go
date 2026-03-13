package compute

import (
	"fmt"
	"log"
	"unsafe"

	"github.com/zerfoo/zerfoo/tensor"
)

// FusedRMSNormGPU implements the FusedRMSNormer interface for GPUEngine.
// Uses the fused GPU kernel when input is GPU-resident, falls back to CPU otherwise.
// Returns (output, scales) where scales contains per-row rsqrt values for backward pass.
func (e *GPUEngine[T]) FusedRMSNormGPU(input, weight *tensor.TensorNumeric[float32], epsilon float32) (*tensor.TensorNumeric[float32], *tensor.TensorNumeric[float32], error) {
	// Native FP16 path: when input has Float16Storage, use RMSNormFP16 kernel
	// directly and produce Float16Storage output.
	if inFS, ok := any(input.GetStorage()).(*tensor.Float16Storage); ok {
		return e.fusedRMSNormFP16Native(inFS, input, weight, epsilon)
	}

	// Only use GPU path when input is GPU-resident.
	if _, ok := input.GetStorage().(*tensor.GPUStorage[float32]); !ok {
		log.Printf("DEBUG: FusedRMSNormGPU: input storage is %T (not GPUStorage[float32]), shape=%v, falling back to CPU", input.GetStorage(), input.Shape())
		return FusedRMSNorm(input, weight, epsilon)
	}

	e.setDevice()

	shape := input.Shape()
	D := shape[len(shape)-1]
	total := input.Size()
	rows := total / D

	// We need float32-specific device pointers. Cast through any.
	f32Engine, ok := any(e).(*GPUEngine[float32])
	if !ok {
		return FusedRMSNorm(input, weight, epsilon)
	}

	devIn, cleanupIn, err := getDevicePtr(f32Engine, input)
	if err != nil {
		return FusedRMSNorm(input, weight, epsilon)
	}
	defer cleanupIn()

	// Get weight device pointer. Weight may be Float16Storage (from FP16
	// weight upload), GPUStorage[float32], or CPU-resident.
	var devWeight unsafe.Pointer
	var cleanupWeight func()
	if wFS, ok := any(weight.GetStorage()).(*tensor.Float16Storage); ok {
		// Weight is FP16 on GPU. Convert to F32 for the F32 RMSNorm kernel.
		fp16W, _, _ := wFS.GPUPtr()
		wElems := weight.Size()
		f32WBytes := wElems * f32Size
		f32W, err := e.pool.Alloc(e.deviceID, f32WBytes)
		if err != nil {
			return nil, nil, fmt.Errorf("FusedRMSNormGPU: alloc f32 weight: %w", err)
		}
		cleanupWeight = func() { e.pool.Free(e.deviceID, f32W, f32WBytes) }
		if err := e.kernels.FP16ToF32(fp16W, f32W, wElems, e.stream); err != nil {
			cleanupWeight()
			return nil, nil, fmt.Errorf("FusedRMSNormGPU: fp16->f32 weight: %w", err)
		}
		devWeight = f32W
	} else {
		// If weight is CPU-resident, upload it to GPU once and swap storage
		// in-place so subsequent calls skip the H2D copy.
		if _, wGPU := weight.GetStorage().(*tensor.GPUStorage[float32]); !wGPU {
			if gpuW, err2 := tensor.ToGPU(weight); err2 == nil {
				weight.SetStorage(gpuW.GetStorage())
			}
		}
		var err error
		devWeight, cleanupWeight, err = getDevicePtr(f32Engine, weight)
		if err != nil {
			return nil, nil, fmt.Errorf("FusedRMSNormGPU: get weight ptr: %w", err)
		}
	}
	defer cleanupWeight()

	outByteSize := total * f32Size
	devOut, err := e.pool.Alloc(e.deviceID, outByteSize)
	if err != nil {
		return FusedRMSNorm(input, weight, epsilon)
	}

	scalesByteSize := rows * f32Size
	devScales, err := e.pool.Alloc(e.deviceID, scalesByteSize)
	if err != nil {
		e.pool.Free(e.deviceID, devOut, outByteSize)
		return FusedRMSNorm(input, weight, epsilon)
	}

	if err := e.kernels.RMSNorm(devIn, devWeight, devOut, devScales, epsilon, rows, D, e.stream); err != nil {
		e.pool.Free(e.deviceID, devOut, outByteSize)
		e.pool.Free(e.deviceID, devScales, scalesByteSize)
		return nil, nil, err
	}

	outTensor, err := makeGPUResult[float32](f32Engine, shape, devOut, total)
	if err != nil {
		e.pool.Free(e.deviceID, devScales, scalesByteSize)
		return nil, nil, err
	}

	// Build scales shape: same as input but last dim = 1.
	scaleShape := make([]int, len(shape))
	copy(scaleShape, shape)
	scaleShape[len(scaleShape)-1] = 1

	scalesTensor, err := makeGPUResult[float32](f32Engine, scaleShape, devScales, rows)
	if err != nil {
		return nil, nil, err
	}

	return outTensor, scalesTensor, nil
}

// fusedRMSNormFP16Native runs RMSNormFP16 kernel when input has Float16Storage.
// Weight is converted to FP16 if needed. Output is Float16Storage.
// Scales are returned as GPUStorage[float32] (used by backward pass).
func (e *GPUEngine[T]) fusedRMSNormFP16Native(
	inFS *tensor.Float16Storage,
	input, weight *tensor.TensorNumeric[float32],
	epsilon float32,
) (*tensor.TensorNumeric[float32], *tensor.TensorNumeric[float32], error) {
	e.setDevice()

	shape := input.Shape()
	D := shape[len(shape)-1]
	total := input.Size()
	rows := total / D

	// Get FP16 input pointer.
	fp16In, _, _ := inFS.GPUPtr()
	if fp16In == nil {
		return nil, nil, fmt.Errorf("fusedRMSNormFP16Native: input has no GPU pointer")
	}

	// Get FP16 weight pointer. Weight might be Float16Storage (from FP16
	// weight upload) or GPUStorage[float32] (norm weights are small and
	// may not have been converted). Convert F32 -> FP16 if needed.
	var fp16W unsafe.Pointer
	var freeW func()
	if wFS, ok := any(weight.GetStorage()).(*tensor.Float16Storage); ok {
		fp16W, _, _ = wFS.GPUPtr()
		freeW = func() {}
	} else {
		// F32 weight -- upload to GPU if needed, then convert to FP16.
		f32Engine, ok := any(e).(*GPUEngine[float32])
		if !ok {
			return nil, nil, fmt.Errorf("fusedRMSNormFP16Native: engine type mismatch")
		}
		devW, cleanupW, err := getDevicePtr(f32Engine, weight)
		if err != nil {
			return nil, nil, fmt.Errorf("fusedRMSNormFP16Native: get weight ptr: %w", err)
		}
		defer cleanupW()

		wElems := weight.Size()
		wFP16Bytes := wElems * fp16Size
		fp16W, err = e.pool.Alloc(e.deviceID, wFP16Bytes)
		if err != nil {
			return nil, nil, fmt.Errorf("fusedRMSNormFP16Native: alloc fp16 weight: %w", err)
		}
		freeW = func() { e.pool.Free(e.deviceID, fp16W, wFP16Bytes) }
		if err := e.kernels.F32ToFP16(devW, fp16W, wElems, e.stream); err != nil {
			freeW()
			return nil, nil, fmt.Errorf("fusedRMSNormFP16Native: f32->fp16 weight: %w", err)
		}
	}
	defer freeW()

	// Allocate FP16 output.
	outFP16Bytes := total * fp16Size
	fp16Out, err := e.pool.Alloc(e.deviceID, outFP16Bytes)
	if err != nil {
		return nil, nil, fmt.Errorf("fusedRMSNormFP16Native: alloc output: %w", err)
	}

	// Run RMSNormFP16 kernel.
	if err := e.kernels.RMSNormFP16(fp16In, fp16W, fp16Out, epsilon, rows, D, e.stream); err != nil {
		e.pool.Free(e.deviceID, fp16Out, outFP16Bytes)
		return nil, nil, fmt.Errorf("fusedRMSNormFP16Native: kernel: %w", err)
	}

	// Wrap output as Float16Storage.
	outFS := tensor.NewFloat16StorageGPU(fp16Out, total, e.deviceID)
	outTensor, err := tensor.New[float32](shape, nil)
	if err != nil {
		e.pool.Free(e.deviceID, fp16Out, outFP16Bytes)
		return nil, nil, err
	}
	outTensor.SetStorage(outFS)

	// Scales: the FP16 RMSNorm kernel does not output per-row scales.
	// Return nil scales -- callers that need scales should use the F32 path.
	return outTensor, nil, nil
}
