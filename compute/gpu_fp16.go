package compute

import (
	"context"
	"fmt"
	"unsafe"

	"github.com/zerfoo/zerfoo/internal/gpuapi"
	"github.com/zerfoo/zerfoo/tensor"
)

const fp16Size = 2 // sizeof(__half)

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
