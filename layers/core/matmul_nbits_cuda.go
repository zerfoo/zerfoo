//go:build cuda && cutlass

package core

import (
	"unsafe"

	"github.com/zerfoo/ztensor/device"
	"github.com/zerfoo/zerfoo/internal/cuda"
	"github.com/zerfoo/zerfoo/internal/cuda/kernels"
	"github.com/zerfoo/ztensor/tensor"
)

// tryQuantizedGemm attempts to use the fused INT4 GEMM CUDA kernel.
// Returns (result, nil) on success, (nil, nil) when GPU path is not
// applicable (CPU data, non-4-bit), or (nil, err) on kernel failure.
//
// input is [..., inFeatures], quantizedWeights is [inFeatures, outFeatures/2] packed INT4.
// scale is [inFeatures] (per-row), zeroPoint is [inFeatures] uint8.
// Result is [..., outFeatures] where outFeatures = quantizedWeights cols * 2.
func tryQuantizedGemm[T tensor.Numeric](
	input *tensor.TensorNumeric[T],
	quantizedWeights *tensor.TensorNumeric[uint8],
	scale *tensor.TensorNumeric[T],
	zeroPoint *tensor.TensorNumeric[uint8],
	nbits int,
) (*tensor.TensorNumeric[T], error) {
	if nbits != 4 {
		return nil, nil
	}

	if input.GetStorage().DeviceType() != device.CUDA {
		return nil, nil
	}

	inputGPU := input.GetStorage().(*tensor.GPUStorage[T])
	inputShape := input.Shape()
	wShape := quantizedWeights.Shape()

	inFeatures := wShape[0]
	outFeatures := wShape[1] * 2

	// Flatten batch dimensions: [..., inFeatures] -> [batch, inFeatures]
	batch := 1
	for i := 0; i < len(inputShape)-1; i++ {
		batch *= inputShape[i]
	}

	// Upload quantized weights to GPU.
	wData := quantizedWeights.Data()
	wBytes := len(wData)
	devW, err := cuda.Malloc(wBytes)
	if err != nil {
		return nil, err
	}
	defer cuda.Free(devW) //nolint:errcheck

	if err := cuda.Memcpy(devW, unsafe.Pointer(&wData[0]), wBytes, cuda.MemcpyHostToDevice); err != nil {
		return nil, err
	}

	// Upload scale to GPU. Scale data is type T but kernel expects float32 pointers.
	scaleData := scale.Data()
	scaleBytes := len(scaleData) * 4 // float32 = 4 bytes
	devScale, err := cuda.Malloc(scaleBytes)
	if err != nil {
		return nil, err
	}
	defer cuda.Free(devScale) //nolint:errcheck

	// Convert scale to float32 for upload.
	scaleF32 := make([]float32, len(scaleData))
	for i, v := range scaleData {
		scaleF32[i] = float32(any(v).(float32))
	}
	if err := cuda.Memcpy(devScale, unsafe.Pointer(&scaleF32[0]), scaleBytes, cuda.MemcpyHostToDevice); err != nil {
		return nil, err
	}

	// Upload zero points to GPU.
	var devZeros unsafe.Pointer
	if zeroPoint != nil {
		zpData := zeroPoint.Data()
		zpBytes := len(zpData)
		devZeros, err = cuda.Malloc(zpBytes)
		if err != nil {
			return nil, err
		}
		defer cuda.Free(devZeros) //nolint:errcheck

		if err := cuda.Memcpy(devZeros, unsafe.Pointer(&zpData[0]), zpBytes, cuda.MemcpyHostToDevice); err != nil {
			return nil, err
		}
	}

	// Allocate output: [batch, outFeatures].
	outElems := batch * outFeatures
	oGPU, err := tensor.NewGPUStorage[T](outElems, inputGPU.DeviceID())
	if err != nil {
		return nil, err
	}

	stream, err := cuda.CreateStream()
	if err != nil {
		oGPU.Free() //nolint:errcheck
		return nil, err
	}
	defer stream.Destroy() //nolint:errcheck

	// Per-row quantization: each row has one group covering all columns.
	groupSize := outFeatures

	// C[i,j] = sum_k input[i,k] * dequant(W[k,j])
	if err := kernels.GemmInt4F32RMul(
		devW, inputGPU.Ptr(), oGPU.Ptr(),
		devScale, devZeros,
		batch, inFeatures, outFeatures, groupSize,
		stream.Ptr(),
	); err != nil {
		oGPU.Free() //nolint:errcheck
		return nil, err
	}

	if err := stream.Synchronize(); err != nil {
		oGPU.Free() //nolint:errcheck
		return nil, err
	}

	// Build output shape: replace last dim with outFeatures.
	outShape := make([]int, len(inputShape))
	copy(outShape, inputShape)
	outShape[len(outShape)-1] = outFeatures

	return tensor.NewWithStorage(outShape, oGPU)
}
