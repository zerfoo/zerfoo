package compute

import (
	"context"
	"fmt"
	"unsafe"

	"github.com/zerfoo/zerfoo/internal/gpuapi"
	"github.com/zerfoo/zerfoo/tensor"
)

const fp16Size = 2 // sizeof(__half)

// fp16BinaryOpNative runs an FP16 binary kernel directly on Float16Storage inputs
// without any F32<->FP16 conversions. Output is a new tensor with Float16Storage.
func fp16BinaryOpNative[T tensor.Numeric](
	e *GPUEngine[T],
	a, b *tensor.TensorNumeric[T],
	aFP16, bFP16 *tensor.Float16Storage,
	kernelFn func(a, b, c unsafe.Pointer, n int, stream gpuapi.Stream) error,
	dst ...*tensor.TensorNumeric[T],
) (*tensor.TensorNumeric[T], error) {
	e.setDevice()

	n := aFP16.Len()

	ptrA, _, _ := aFP16.GPUPtr()
	ptrB, _, _ := bFP16.GPUPtr()

	// Allocate FP16 output buffer.
	outBytes := n * fp16Size
	devOut, err := e.pool.Alloc(e.deviceID, outBytes)
	if err != nil {
		return nil, fmt.Errorf("fp16BinaryOpNative: alloc output: %w", err)
	}

	if err := kernelFn(ptrA, ptrB, devOut, n, e.stream); err != nil {
		e.pool.Free(e.deviceID, devOut, outBytes)
		return nil, fmt.Errorf("fp16BinaryOpNative: kernel: %w", err)
	}

	outStorage := any(tensor.NewFloat16StorageGPU(devOut, n, e.deviceID)).(tensor.Storage[T])

	if len(dst) > 0 && dst[0] != nil {
		dst[0].SetStorage(outStorage)
		dst[0].SetShape(a.Shape())
		return dst[0], nil
	}

	t, err := tensor.NewWithStorage[T](a.Shape(), outStorage)
	if err != nil {
		e.pool.Free(e.deviceID, devOut, outBytes)
		return nil, err
	}

	return t, nil
}

// getFP16DevicePtr returns the FP16 GPU device pointer for a Float16Storage.
// If the storage has no GPU pointer, it uploads the host data to the GPU.
func getFP16DevicePtr[T tensor.Numeric](
	e *GPUEngine[T],
	fs *tensor.Float16Storage,
) (unsafe.Pointer, func(), error) {
	ptr, _, _ := fs.GPUPtr()
	if ptr != nil {
		return ptr, noopCleanup, nil
	}

	// Upload host FP16 bytes to GPU.
	raw := fs.RawBytes()
	byteSize := len(raw)
	devPtr, err := e.pool.Alloc(e.deviceID, byteSize)
	if err != nil {
		return nil, nil, fmt.Errorf("getFP16DevicePtr: alloc: %w", err)
	}

	if err := e.runtime.Memcpy(devPtr, unsafe.Pointer(&raw[0]), byteSize, gpuapi.MemcpyHostToDevice); err != nil {
		e.pool.Free(e.deviceID, devPtr, byteSize)
		return nil, nil, fmt.Errorf("getFP16DevicePtr: memcpy: %w", err)
	}

	cleanup := func() {
		e.pool.Free(e.deviceID, devPtr, byteSize)
	}

	return devPtr, cleanup, nil
}

// fp16BinaryOpMixed runs an FP16 binary kernel when one input is Float16Storage
// and the other is F32 GPUStorage. The F32 input is converted to FP16 first.
func fp16BinaryOpMixed[T tensor.Numeric](
	e *GPUEngine[T],
	ctx context.Context,
	a, b *tensor.TensorNumeric[T],
	fp16Stor *tensor.Float16Storage,
	fp16IsA bool,
	kernelFn func(a, b, c unsafe.Pointer, n int, stream gpuapi.Stream) error,
	dst ...*tensor.TensorNumeric[T],
) (*tensor.TensorNumeric[T], error) {
	e.setDevice()

	n := fp16Stor.Len()

	// Get the FP16 pointer.
	fp16Ptr, fp16Cleanup, err := getFP16DevicePtr(e, fp16Stor)
	if err != nil {
		return nil, err
	}
	defer fp16Cleanup()

	// Get the F32 pointer and convert to FP16.
	var f32Tensor *tensor.TensorNumeric[T]
	if fp16IsA {
		f32Tensor = b
	} else {
		f32Tensor = a
	}

	devF32, cleanupF32, err := getDevicePtr(e, f32Tensor)
	if err != nil {
		return nil, err
	}
	defer cleanupF32()

	// Convert F32 to FP16.
	convBuf, err := e.pool.Alloc(e.deviceID, n*fp16Size)
	if err != nil {
		return nil, fmt.Errorf("fp16BinaryOpMixed: alloc conv: %w", err)
	}
	defer e.pool.Free(e.deviceID, convBuf, n*fp16Size)

	if err := e.kernels.F32ToFP16(devF32, convBuf, n, e.stream); err != nil {
		return nil, fmt.Errorf("fp16BinaryOpMixed: f32->fp16: %w", err)
	}

	// Allocate FP16 output.
	outBytes := n * fp16Size
	devOut, err := e.pool.Alloc(e.deviceID, outBytes)
	if err != nil {
		return nil, fmt.Errorf("fp16BinaryOpMixed: alloc output: %w", err)
	}

	var ptrA, ptrB unsafe.Pointer
	if fp16IsA {
		ptrA = fp16Ptr
		ptrB = convBuf
	} else {
		ptrA = convBuf
		ptrB = fp16Ptr
	}

	if err := kernelFn(ptrA, ptrB, devOut, n, e.stream); err != nil {
		e.pool.Free(e.deviceID, devOut, outBytes)
		return nil, fmt.Errorf("fp16BinaryOpMixed: kernel: %w", err)
	}

	outStorage := any(tensor.NewFloat16StorageGPU(devOut, n, e.deviceID)).(tensor.Storage[T])

	if len(dst) > 0 && dst[0] != nil {
		dst[0].SetStorage(outStorage)
		dst[0].SetShape(a.Shape())
		return dst[0], nil
	}

	t, err := tensor.NewWithStorage[T](a.Shape(), outStorage)
	if err != nil {
		e.pool.Free(e.deviceID, devOut, outBytes)
		return nil, err
	}

	return t, nil
}

// tryFP16NativeBinaryOp checks whether both inputs have Float16Storage and,
// if so, runs the FP16 kernel directly without conversion. If only one input
// is Float16Storage, it converts the F32 operand. Returns (nil, nil) when
// neither input is Float16Storage, signalling the caller to fall through.
func tryFP16NativeBinaryOp[T tensor.Numeric](
	e *GPUEngine[T],
	ctx context.Context,
	a, b *tensor.TensorNumeric[T],
	kernelFn func(a, b, c unsafe.Pointer, n int, stream gpuapi.Stream) error,
	dst ...*tensor.TensorNumeric[T],
) (*tensor.TensorNumeric[T], error) {
	aFP16, aOk := any(a.GetStorage()).(*tensor.Float16Storage)
	bFP16, bOk := any(b.GetStorage()).(*tensor.Float16Storage)

	switch {
	case aOk && bOk:
		return fp16BinaryOpNative(e, a, b, aFP16, bFP16, kernelFn, dst...)
	case aOk:
		return fp16BinaryOpMixed(e, ctx, a, b, aFP16, true, kernelFn, dst...)
	case bOk:
		return fp16BinaryOpMixed(e, ctx, a, b, bFP16, false, kernelFn, dst...)
	default:
		return nil, nil
	}
}

// fp16BinaryOp converts two F32 GPU tensors to FP16, runs an FP16 binary kernel,
// and converts the FP16 result back to F32.
func fp16BinaryOp[T tensor.Numeric](
	e *GPUEngine[T],
	ctx context.Context,
	a, b *tensor.TensorNumeric[T],
	kernelFn func(a, b, c unsafe.Pointer, n int, stream gpuapi.Stream) error,
	dst ...*tensor.TensorNumeric[T],
) (*tensor.TensorNumeric[T], error) {
	e.setDevice()

	devA, cleanupA, err := getDevicePtr(e, a)
	if err != nil {
		return e.cpu.Add(ctx, a, b, dst...)
	}
	defer cleanupA()

	devB, cleanupB, err := getDevicePtr(e, b)
	if err != nil {
		return e.cpu.Add(ctx, a, b, dst...)
	}
	defer cleanupB()

	n := a.GetStorage().Len()

	// Allocate FP16 buffers.
	fp16A, err := e.pool.Alloc(e.deviceID, n*fp16Size)
	if err != nil {
		return nil, fmt.Errorf("fp16BinaryOp: alloc fp16A: %w", err)
	}
	defer e.pool.Free(e.deviceID, fp16A, n*fp16Size)

	fp16B, err := e.pool.Alloc(e.deviceID, n*fp16Size)
	if err != nil {
		return nil, fmt.Errorf("fp16BinaryOp: alloc fp16B: %w", err)
	}
	defer e.pool.Free(e.deviceID, fp16B, n*fp16Size)

	fp16C, err := e.pool.Alloc(e.deviceID, n*fp16Size)
	if err != nil {
		return nil, fmt.Errorf("fp16BinaryOp: alloc fp16C: %w", err)
	}
	defer e.pool.Free(e.deviceID, fp16C, n*fp16Size)

	// Convert F32 -> FP16.
	if err := e.kernels.F32ToFP16(devA, fp16A, n, e.stream); err != nil {
		return nil, fmt.Errorf("fp16BinaryOp: f32->fp16 A: %w", err)
	}
	if err := e.kernels.F32ToFP16(devB, fp16B, n, e.stream); err != nil {
		return nil, fmt.Errorf("fp16BinaryOp: f32->fp16 B: %w", err)
	}

	// Run FP16 kernel.
	if err := kernelFn(fp16A, fp16B, fp16C, n, e.stream); err != nil {
		return nil, fmt.Errorf("fp16BinaryOp: kernel: %w", err)
	}

	// Allocate F32 output and convert FP16 -> F32.
	outBytes := n * f32Size
	devOut, err := e.pool.Alloc(e.deviceID, outBytes)
	if err != nil {
		return nil, fmt.Errorf("fp16BinaryOp: alloc output: %w", err)
	}
	if err := e.kernels.FP16ToF32(fp16C, devOut, n, e.stream); err != nil {
		e.pool.Free(e.deviceID, devOut, outBytes)
		return nil, fmt.Errorf("fp16BinaryOp: fp16->f32: %w", err)
	}

	return makeGPUResult[T](e, a.Shape(), devOut, n, dst...)
}

// fp16MatMul runs MatMul in FP16: converts both F32 inputs to FP16,
// uses MixedFP16Gemm (FP16 inputs, FP32 output via cublasGemmEx).
func fp16MatMul[T tensor.Numeric](
	e *GPUEngine[T],
	ctx context.Context,
	a, b *tensor.TensorNumeric[T],
	dst ...*tensor.TensorNumeric[T],
) (*tensor.TensorNumeric[T], error) {
	if e.blas == nil {
		return e.cpu.MatMul(ctx, a, b, dst...)
	}

	e.setDevice()

	aShape := a.Shape()
	bShape := b.Shape()

	if len(aShape) < 2 || len(bShape) < 2 {
		return nil, fmt.Errorf("fp16MatMul: tensors must have at least 2 dimensions")
	}

	m := aShape[len(aShape)-2]
	k := aShape[len(aShape)-1]
	bK := bShape[len(bShape)-2]
	n := bShape[len(bShape)-1]

	if k != bK {
		return nil, fmt.Errorf("fp16MatMul: incompatible inner dimensions %d != %d", k, bK)
	}

	// Compute batch dimensions.
	aBatch := aShape[:len(aShape)-2]
	bBatch := bShape[:len(bShape)-2]

	aBatchSize := 1
	for _, d := range aBatch {
		aBatchSize *= d
	}

	bBatchSize := 1
	for _, d := range bBatch {
		bBatchSize *= d
	}

	if bBatchSize != 1 && aBatchSize != bBatchSize {
		return nil, fmt.Errorf("fp16MatMul: batch dimensions %v and %v are incompatible", aBatch, bBatch)
	}

	batchSize := aBatchSize
	if bBatchSize > batchSize {
		batchSize = bBatchSize
	}

	// Get F32 device pointers.
	devA, cleanupA, err := getDevicePtr(e, a)
	if err != nil {
		return e.cpu.MatMul(ctx, a, b, dst...)
	}
	defer cleanupA()

	devB, cleanupB, err := getDevicePtr(e, b)
	if err != nil {
		return e.cpu.MatMul(ctx, a, b, dst...)
	}
	defer cleanupB()

	aMatElems := m * k
	bMatElems := k * n
	cMatElems := m * n

	totalAElems := batchSize * aMatElems
	totalBElems := batchSize * bMatElems
	totalCElems := batchSize * cMatElems

	// Allocate FP16 buffers for A and B.
	fp16ASize := totalAElems * fp16Size
	fp16A, err := e.pool.Alloc(e.deviceID, fp16ASize)
	if err != nil {
		return nil, fmt.Errorf("fp16MatMul: alloc fp16A: %w", err)
	}
	defer e.pool.Free(e.deviceID, fp16A, fp16ASize)

	fp16BSize := totalBElems * fp16Size
	fp16B, err := e.pool.Alloc(e.deviceID, fp16BSize)
	if err != nil {
		return nil, fmt.Errorf("fp16MatMul: alloc fp16B: %w", err)
	}
	defer e.pool.Free(e.deviceID, fp16B, fp16BSize)

	// Convert F32 -> FP16.
	if err := e.kernels.F32ToFP16(devA, fp16A, totalAElems, e.stream); err != nil {
		return nil, fmt.Errorf("fp16MatMul: f32->fp16 A: %w", err)
	}
	if err := e.kernels.F32ToFP16(devB, fp16B, totalBElems, e.stream); err != nil {
		return nil, fmt.Errorf("fp16MatMul: f32->fp16 B: %w", err)
	}

	// Allocate F32 output (MixedFP16Gemm outputs F32).
	outBytes := totalCElems * f32Size
	devC, err := e.pool.Alloc(e.deviceID, outBytes)
	if err != nil {
		return nil, fmt.Errorf("fp16MatMul: alloc output: %w", err)
	}

	// MixedFP16Gemm: FP16 inputs, FP32 output, FP32 accumulation.
	// Loop over batches since MixedFP16Gemm only handles single 2D GEMM.
	for batch := range batchSize {
		aOff := batch * aMatElems * fp16Size
		bOff := 0
		if bBatchSize > 1 {
			bOff = batch * bMatElems * fp16Size
		}
		cOff := batch * cMatElems * f32Size

		batchFP16A := unsafe.Add(fp16A, aOff)
		batchFP16B := unsafe.Add(fp16B, bOff)
		batchDevC := unsafe.Add(devC, cOff)

		if err := e.blas.MixedFP16Gemm(m, n, k, 1.0, batchFP16A, batchFP16B, 0.0, batchDevC); err != nil {
			e.pool.Free(e.deviceID, devC, outBytes)
			return nil, fmt.Errorf("fp16MatMul: gemm batch %d: %w", batch, err)
		}
	}

	// Build output shape.
	outShape := make([]int, 0, len(aShape))
	outShape = append(outShape, aBatch...)
	outShape = append(outShape, m, n)

	return makeGPUResult[T](e, outShape, devC, totalCElems, dst...)
}

// fp16MatMulNative runs MatMul when one or both inputs have Float16Storage.
// For Float16Storage inputs, the FP16 device pointer is used directly (no conversion).
// For F32 inputs, a single F32->FP16 conversion is performed.
// Output is Float16Storage backed by a GPU device pointer.
func fp16MatMulNative[T tensor.Numeric](
	e *GPUEngine[T],
	ctx context.Context,
	a, b *tensor.TensorNumeric[T],
	aFP16 *tensor.Float16Storage,
	bFP16 *tensor.Float16Storage,
	aIsFP16, bIsFP16 bool,
	dst ...*tensor.TensorNumeric[T],
) (*tensor.TensorNumeric[T], error) {
	if e.blas == nil {
		return e.cpu.MatMul(ctx, a, b, dst...)
	}

	e.setDevice()

	aShape := a.Shape()
	bShape := b.Shape()

	if len(aShape) < 2 || len(bShape) < 2 {
		return nil, fmt.Errorf("fp16MatMulNative: tensors must have at least 2 dimensions")
	}

	m := aShape[len(aShape)-2]
	k := aShape[len(aShape)-1]
	bK := bShape[len(bShape)-2]
	n := bShape[len(bShape)-1]

	if k != bK {
		return nil, fmt.Errorf("fp16MatMulNative: incompatible inner dimensions %d != %d", k, bK)
	}

	// Compute batch dimensions.
	aBatch := aShape[:len(aShape)-2]
	bBatch := bShape[:len(bShape)-2]

	aBatchSize := 1
	for _, d := range aBatch {
		aBatchSize *= d
	}

	bBatchSize := 1
	for _, d := range bBatch {
		bBatchSize *= d
	}

	if bBatchSize != 1 && aBatchSize != bBatchSize {
		return nil, fmt.Errorf("fp16MatMulNative: batch dimensions %v and %v are incompatible", aBatch, bBatch)
	}

	batchSize := aBatchSize
	if bBatchSize > batchSize {
		batchSize = bBatchSize
	}

	aMatElems := m * k
	bMatElems := k * n
	cMatElems := m * n

	totalAElems := batchSize * aMatElems
	totalBElems := batchSize * bMatElems
	totalCElems := batchSize * cMatElems

	// Get or convert A to FP16 device pointer.
	var devA unsafe.Pointer
	var freeA func()
	if aIsFP16 {
		ptr, _, _ := aFP16.GPUPtr()
		if ptr != nil {
			devA = ptr
			freeA = func() {}
		} else {
			// Upload FP16 host data to GPU.
			aBytes := aFP16.RawBytes()
			var err error
			devA, err = e.pool.Alloc(e.deviceID, len(aBytes))
			if err != nil {
				return nil, fmt.Errorf("fp16MatMulNative: alloc A upload: %w", err)
			}
			freeA = func() { e.pool.Free(e.deviceID, devA, len(aBytes)) }
			if err := e.runtime.Memcpy(devA, unsafe.Pointer(&aBytes[0]), len(aBytes), gpuapi.MemcpyHostToDevice); err != nil {
				freeA()
				return nil, fmt.Errorf("fp16MatMulNative: memcpy A: %w", err)
			}
		}
	} else {
		// A is F32: get device pointer and convert to FP16.
		f32A, cleanupF32A, err := getDevicePtr(e, a)
		if err != nil {
			return e.cpu.MatMul(ctx, a, b, dst...)
		}
		defer cleanupF32A()

		fp16ASize := totalAElems * fp16Size
		devA, err = e.pool.Alloc(e.deviceID, fp16ASize)
		if err != nil {
			return nil, fmt.Errorf("fp16MatMulNative: alloc fp16A: %w", err)
		}
		freeA = func() { e.pool.Free(e.deviceID, devA, fp16ASize) }
		if err := e.kernels.F32ToFP16(f32A, devA, totalAElems, e.stream); err != nil {
			freeA()
			return nil, fmt.Errorf("fp16MatMulNative: f32->fp16 A: %w", err)
		}
	}
	defer freeA()

	// Get or convert B to FP16 device pointer.
	var devB unsafe.Pointer
	var freeB func()
	if bIsFP16 {
		ptr, _, _ := bFP16.GPUPtr()
		if ptr != nil {
			devB = ptr
			freeB = func() {}
		} else {
			bBytes := bFP16.RawBytes()
			var err error
			devB, err = e.pool.Alloc(e.deviceID, len(bBytes))
			if err != nil {
				return nil, fmt.Errorf("fp16MatMulNative: alloc B upload: %w", err)
			}
			freeB = func() { e.pool.Free(e.deviceID, devB, len(bBytes)) }
			if err := e.runtime.Memcpy(devB, unsafe.Pointer(&bBytes[0]), len(bBytes), gpuapi.MemcpyHostToDevice); err != nil {
				freeB()
				return nil, fmt.Errorf("fp16MatMulNative: memcpy B: %w", err)
			}
		}
	} else {
		f32B, cleanupF32B, err := getDevicePtr(e, b)
		if err != nil {
			return e.cpu.MatMul(ctx, a, b, dst...)
		}
		defer cleanupF32B()

		fp16BSize := totalBElems * fp16Size
		devB, err = e.pool.Alloc(e.deviceID, fp16BSize)
		if err != nil {
			return nil, fmt.Errorf("fp16MatMulNative: alloc fp16B: %w", err)
		}
		freeB = func() { e.pool.Free(e.deviceID, devB, fp16BSize) }
		if err := e.kernels.F32ToFP16(f32B, devB, totalBElems, e.stream); err != nil {
			freeB()
			return nil, fmt.Errorf("fp16MatMulNative: f32->fp16 B: %w", err)
		}
	}
	defer freeB()

	// Allocate FP16 output.
	outFP16Bytes := totalCElems * fp16Size
	devC, err := e.pool.Alloc(e.deviceID, outFP16Bytes)
	if err != nil {
		return nil, fmt.Errorf("fp16MatMulNative: alloc output: %w", err)
	}

	// Float16Gemm: FP16 inputs, FP16 output, FP32 accumulation.
	// Loop over batches since Float16Gemm only handles single 2D GEMM.
	for batch := range batchSize {
		aOff := batch * aMatElems * fp16Size
		bOff := 0
		if bBatchSize > 1 {
			bOff = batch * bMatElems * fp16Size
		}
		cOff := batch * cMatElems * fp16Size

		batchA := unsafe.Add(devA, aOff)
		batchB := unsafe.Add(devB, bOff)
		batchC := unsafe.Add(devC, cOff)

		if err := e.blas.Float16Gemm(m, n, k, 1.0, batchA, batchB, 0.0, batchC); err != nil {
			e.pool.Free(e.deviceID, devC, outFP16Bytes)
			return nil, fmt.Errorf("fp16MatMulNative: gemm batch %d: %w", batch, err)
		}
	}

	// Build output shape.
	outShape := make([]int, 0, len(aShape))
	outShape = append(outShape, aBatch...)
	outShape = append(outShape, m, n)

	// Wrap output as Float16Storage.
	// Float16Storage implements Storage[float32]; T is float32 when FP16 storage is in play.
	fp16Out := tensor.NewFloat16StorageGPU(devC, totalCElems, e.deviceID)
	storageT := any(fp16Out).(tensor.Storage[T])

	t, err := tensor.NewWithStorage[T](outShape, storageT)
	if err != nil {
		e.pool.Free(e.deviceID, devC, outFP16Bytes)
		return nil, fmt.Errorf("fp16MatMulNative: create result: %w", err)
	}

	return t, nil
}

// fp16ScaledSoftmax converts F32 input to FP16, runs ScaledSoftmaxFP16, converts back.
func fp16ScaledSoftmax[T tensor.Numeric](
	e *GPUEngine[T],
	input *tensor.TensorNumeric[T],
	scale float32,
	outer, inner, axisSize int,
) (*tensor.TensorNumeric[T], error) {
	e.setDevice()

	n := input.GetStorage().Len()

	devIn, cleanupIn, err := getDevicePtr(e, input)
	if err != nil {
		return nil, fmt.Errorf("fp16ScaledSoftmax input: %w", err)
	}
	defer cleanupIn()

	// Allocate FP16 in/out.
	fp16In, err := e.pool.Alloc(e.deviceID, n*fp16Size)
	if err != nil {
		return nil, fmt.Errorf("fp16ScaledSoftmax: alloc fp16In: %w", err)
	}
	defer e.pool.Free(e.deviceID, fp16In, n*fp16Size)

	fp16Out, err := e.pool.Alloc(e.deviceID, n*fp16Size)
	if err != nil {
		return nil, fmt.Errorf("fp16ScaledSoftmax: alloc fp16Out: %w", err)
	}
	defer e.pool.Free(e.deviceID, fp16Out, n*fp16Size)

	// Convert and run.
	if err := e.kernels.F32ToFP16(devIn, fp16In, n, e.stream); err != nil {
		return nil, fmt.Errorf("fp16ScaledSoftmax: f32->fp16: %w", err)
	}
	if err := e.kernels.ScaledSoftmaxFP16(fp16In, fp16Out, outer, inner, axisSize, scale, e.stream); err != nil {
		return nil, fmt.Errorf("fp16ScaledSoftmax: kernel: %w", err)
	}

	// Convert back to F32.
	outBytes := n * f32Size
	devOut, err := e.pool.Alloc(e.deviceID, outBytes)
	if err != nil {
		return nil, fmt.Errorf("fp16ScaledSoftmax: alloc output: %w", err)
	}
	if err := e.kernels.FP16ToF32(fp16Out, devOut, n, e.stream); err != nil {
		e.pool.Free(e.deviceID, devOut, outBytes)
		return nil, fmt.Errorf("fp16ScaledSoftmax: fp16->f32: %w", err)
	}

	return makeGPUResult[T](e, input.Shape(), devOut, n)
}

// fp16FusedAddRMSNorm converts F32 inputs to FP16, runs the FP16 RMSNorm kernel,
// and converts outputs back to F32.
func fp16FusedAddRMSNorm[T tensor.Numeric](
	e *GPUEngine[T],
	input, residual, weight *tensor.TensorNumeric[T],
	eps float32,
) (normed *tensor.TensorNumeric[T], residualOut *tensor.TensorNumeric[T], scales *tensor.TensorNumeric[T], err error) {
	// For FP16 fused add+rmsnorm, we decompose into:
	// 1. F32 Add (input + residual) -- stays in F32 for the residual stream
	// 2. FP16 RMSNorm on the sum
	e.setDevice()

	inShape := input.Shape()
	D := inShape[len(inShape)-1]
	rows := 1
	for _, d := range inShape[:len(inShape)-1] {
		rows *= d
	}
	n := rows * D

	// Get F32 device pointers.
	inPtr, inCleanup, err := getDevicePtr(e, input)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("fp16FusedAddRMSNorm input: %w", err)
	}
	defer inCleanup()

	resPtr, resCleanup, err := getDevicePtr(e, residual)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("fp16FusedAddRMSNorm residual: %w", err)
	}
	defer resCleanup()

	wPtr, wCleanup, err := getDevicePtr(e, weight)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("fp16FusedAddRMSNorm weight: %w", err)
	}
	defer wCleanup()

	outBytes := n * f32Size

	// Step 1: Compute sum = input + residual in F32 (preserves residual stream precision).
	devSum, err := e.pool.Alloc(e.deviceID, outBytes)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("fp16FusedAddRMSNorm alloc sum: %w", err)
	}
	if err := e.kernels.Add(inPtr, resPtr, devSum, n, e.stream); err != nil {
		e.pool.Free(e.deviceID, devSum, outBytes)
		return nil, nil, nil, fmt.Errorf("fp16FusedAddRMSNorm add: %w", err)
	}

	// Step 2: Convert sum and weight to FP16, run RMSNormFP16.
	fp16Sum, err := e.pool.Alloc(e.deviceID, n*fp16Size)
	if err != nil {
		e.pool.Free(e.deviceID, devSum, outBytes)
		return nil, nil, nil, fmt.Errorf("fp16FusedAddRMSNorm alloc fp16Sum: %w", err)
	}
	defer e.pool.Free(e.deviceID, fp16Sum, n*fp16Size)

	wD := D
	fp16W, err := e.pool.Alloc(e.deviceID, wD*fp16Size)
	if err != nil {
		e.pool.Free(e.deviceID, devSum, outBytes)
		return nil, nil, nil, fmt.Errorf("fp16FusedAddRMSNorm alloc fp16W: %w", err)
	}
	defer e.pool.Free(e.deviceID, fp16W, wD*fp16Size)

	fp16Out, err := e.pool.Alloc(e.deviceID, n*fp16Size)
	if err != nil {
		e.pool.Free(e.deviceID, devSum, outBytes)
		return nil, nil, nil, fmt.Errorf("fp16FusedAddRMSNorm alloc fp16Out: %w", err)
	}
	defer e.pool.Free(e.deviceID, fp16Out, n*fp16Size)

	if err := e.kernels.F32ToFP16(devSum, fp16Sum, n, e.stream); err != nil {
		e.pool.Free(e.deviceID, devSum, outBytes)
		return nil, nil, nil, fmt.Errorf("fp16FusedAddRMSNorm f32->fp16 sum: %w", err)
	}
	if err := e.kernels.F32ToFP16(wPtr, fp16W, wD, e.stream); err != nil {
		e.pool.Free(e.deviceID, devSum, outBytes)
		return nil, nil, nil, fmt.Errorf("fp16FusedAddRMSNorm f32->fp16 weight: %w", err)
	}

	if err := e.kernels.RMSNormFP16(fp16Sum, fp16W, fp16Out, eps, rows, D, e.stream); err != nil {
		e.pool.Free(e.deviceID, devSum, outBytes)
		return nil, nil, nil, fmt.Errorf("fp16FusedAddRMSNorm rmsnorm: %w", err)
	}

	// Convert normed output FP16 -> F32.
	devNormed, err := e.pool.Alloc(e.deviceID, outBytes)
	if err != nil {
		e.pool.Free(e.deviceID, devSum, outBytes)
		return nil, nil, nil, fmt.Errorf("fp16FusedAddRMSNorm alloc normed: %w", err)
	}
	if err := e.kernels.FP16ToF32(fp16Out, devNormed, n, e.stream); err != nil {
		e.pool.Free(e.deviceID, devSum, outBytes)
		e.pool.Free(e.deviceID, devNormed, outBytes)
		return nil, nil, nil, fmt.Errorf("fp16FusedAddRMSNorm fp16->f32: %w", err)
	}

	normedT, err := makeGPUResult[T](e, inShape, devNormed, n)
	if err != nil {
		e.pool.Free(e.deviceID, devSum, outBytes)
		return nil, nil, nil, err
	}
	sumT, err := makeGPUResult[T](e, inShape, devSum, n)
	if err != nil {
		return nil, nil, nil, err
	}

	return normedT, sumT, nil, nil
}

// fp16ScaledSoftmaxNative runs ScaledSoftmaxFP16 directly on Float16Storage GPU data.
// No F32<->FP16 conversion is performed. Output is Float16Storage.
func fp16ScaledSoftmaxNative[T tensor.Numeric](
	e *GPUEngine[T],
	fs *tensor.Float16Storage,
	shape []int,
	scale float32,
	outer, inner, axisSize int,
) (*tensor.TensorNumeric[T], error) {
	e.setDevice()

	fp16In, _, _ := fs.GPUPtr()
	if fp16In == nil {
		return nil, fmt.Errorf("fp16ScaledSoftmaxNative: Float16Storage has no GPU pointer")
	}

	n := fs.Len()
	fp16OutBytes := n * fp16Size
	fp16Out, err := e.pool.Alloc(e.deviceID, fp16OutBytes)
	if err != nil {
		return nil, fmt.Errorf("fp16ScaledSoftmaxNative: alloc fp16Out: %w", err)
	}

	if err := e.kernels.ScaledSoftmaxFP16(fp16In, fp16Out, outer, inner, axisSize, scale, e.stream); err != nil {
		e.pool.Free(e.deviceID, fp16Out, fp16OutBytes)
		return nil, fmt.Errorf("fp16ScaledSoftmaxNative: kernel: %w", err)
	}

	outFS := &tensor.Float16Storage{}
	outFS.Set(make([]float32, n))
	outFS.SetGPUPtr(fp16Out, fp16OutBytes, e.deviceID)

	t, err := tensor.NewWithStorage[T](shape, any(outFS).(tensor.Storage[T]))
	if err != nil {
		e.pool.Free(e.deviceID, fp16Out, fp16OutBytes)
		return nil, fmt.Errorf("fp16ScaledSoftmaxNative: wrap tensor: %w", err)
	}

	return t, nil
}

// fp16FusedAddRMSNormNative runs FP16 Add + RMSNorm directly on Float16Storage GPU data.
// No F32<->FP16 conversion is performed. Both outputs are Float16Storage.
func fp16FusedAddRMSNormNative[T tensor.Numeric](
	e *GPUEngine[T],
	inFS, resFS *tensor.Float16Storage,
	input, weight *tensor.TensorNumeric[T],
	eps float32,
) (normed *tensor.TensorNumeric[T], residualOut *tensor.TensorNumeric[T], scales *tensor.TensorNumeric[T], err error) {
	e.setDevice()

	inShape := input.Shape()
	D := inShape[len(inShape)-1]
	rows := 1
	for _, d := range inShape[:len(inShape)-1] {
		rows *= d
	}
	n := rows * D

	fp16In, _, _ := inFS.GPUPtr()
	if fp16In == nil {
		return nil, nil, nil, fmt.Errorf("fp16FusedAddRMSNormNative: input Float16Storage has no GPU pointer")
	}

	fp16Res, _, _ := resFS.GPUPtr()
	if fp16Res == nil {
		return nil, nil, nil, fmt.Errorf("fp16FusedAddRMSNormNative: residual Float16Storage has no GPU pointer")
	}

	// Weight may be Float16Storage or F32. Check and handle both.
	var fp16W unsafe.Pointer
	if wFS, ok := any(weight.GetStorage()).(*tensor.Float16Storage); ok {
		fp16W, _, _ = wFS.GPUPtr()
		if fp16W == nil {
			return nil, nil, nil, fmt.Errorf("fp16FusedAddRMSNormNative: weight Float16Storage has no GPU pointer")
		}
	} else {
		// Convert F32 weight to FP16.
		wPtr, wCleanup, wErr := getDevicePtr(e, weight)
		if wErr != nil {
			return nil, nil, nil, fmt.Errorf("fp16FusedAddRMSNormNative weight: %w", wErr)
		}
		defer wCleanup()

		fp16WSize := D * fp16Size
		fp16W, err = e.pool.Alloc(e.deviceID, fp16WSize)
		if err != nil {
			return nil, nil, nil, fmt.Errorf("fp16FusedAddRMSNormNative: alloc fp16W: %w", err)
		}
		defer e.pool.Free(e.deviceID, fp16W, fp16WSize)

		if err := e.kernels.F32ToFP16(wPtr, fp16W, D, e.stream); err != nil {
			return nil, nil, nil, fmt.Errorf("fp16FusedAddRMSNormNative: f32->fp16 weight: %w", err)
		}
	}

	fp16Bytes := n * fp16Size

	// Step 1: sum = input + residual in FP16.
	fp16Sum, err := e.pool.Alloc(e.deviceID, fp16Bytes)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("fp16FusedAddRMSNormNative: alloc fp16Sum: %w", err)
	}

	if err := e.kernels.AddFP16(fp16In, fp16Res, fp16Sum, n, e.stream); err != nil {
		e.pool.Free(e.deviceID, fp16Sum, fp16Bytes)
		return nil, nil, nil, fmt.Errorf("fp16FusedAddRMSNormNative: add: %w", err)
	}

	// Step 2: normed = rmsnorm(sum, weight) in FP16.
	fp16Normed, err := e.pool.Alloc(e.deviceID, fp16Bytes)
	if err != nil {
		e.pool.Free(e.deviceID, fp16Sum, fp16Bytes)
		return nil, nil, nil, fmt.Errorf("fp16FusedAddRMSNormNative: alloc fp16Normed: %w", err)
	}

	if err := e.kernels.RMSNormFP16(fp16Sum, fp16W, fp16Normed, eps, rows, D, e.stream); err != nil {
		e.pool.Free(e.deviceID, fp16Sum, fp16Bytes)
		e.pool.Free(e.deviceID, fp16Normed, fp16Bytes)
		return nil, nil, nil, fmt.Errorf("fp16FusedAddRMSNormNative: rmsnorm: %w", err)
	}

	// Wrap outputs as Float16Storage tensors.
	normedFS := &tensor.Float16Storage{}
	normedFS.Set(make([]float32, n))
	normedFS.SetGPUPtr(fp16Normed, fp16Bytes, e.deviceID)

	sumFS := &tensor.Float16Storage{}
	sumFS.Set(make([]float32, n))
	sumFS.SetGPUPtr(fp16Sum, fp16Bytes, e.deviceID)

	normedT, nErr := tensor.NewWithStorage[T](inShape, any(normedFS).(tensor.Storage[T]))
	if nErr != nil {
		e.pool.Free(e.deviceID, fp16Sum, fp16Bytes)
		e.pool.Free(e.deviceID, fp16Normed, fp16Bytes)
		return nil, nil, nil, nErr
	}

	sumT2, sErr := tensor.NewWithStorage[T](inShape, any(sumFS).(tensor.Storage[T]))
	if sErr != nil {
		e.pool.Free(e.deviceID, fp16Sum, fp16Bytes)
		return nil, nil, nil, sErr
	}

	return normedT, sumT2, nil, nil
}
