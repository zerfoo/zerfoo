//go:build cuda

package compute

import (
	"github.com/zerfoo/zerfoo/tensor"
)

// FusedRMSNormGPU implements the FusedRMSNormer interface for GPUEngine.
// Uses the fused GPU kernel when input is GPU-resident, falls back to CPU otherwise.
// Returns (output, scales) where scales contains per-row rsqrt values for backward pass.
func (e *GPUEngine[T]) FusedRMSNormGPU(input, weight *tensor.TensorNumeric[float32], epsilon float32) (*tensor.TensorNumeric[float32], *tensor.TensorNumeric[float32], error) {
	// Only use GPU path when input is GPU-resident.
	if _, ok := input.GetStorage().(*tensor.GPUStorage[float32]); !ok {
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

	devWeight, cleanupWeight, err := getDevicePtr(f32Engine, weight)
	if err != nil {
		return FusedRMSNorm(input, weight, epsilon)
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
