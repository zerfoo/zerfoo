package compute

import (
	"context"
	"fmt"
	"log"
	"unsafe"

	"github.com/zerfoo/zerfoo/internal/cublas"
	"github.com/zerfoo/zerfoo/internal/gpuapi"
	"github.com/zerfoo/zerfoo/tensor"
)

// fp8Scratchpad holds pre-allocated, reusable device buffers for FP8 MatMul.
// Buffers are grow-only: if a call needs a larger buffer, the old one is freed
// and a bigger one allocated. This avoids per-call arena allocations that
// previously exhausted the 2GB arena and caused slow MemPool fallbacks.
type fp8Scratchpad struct {
	// fp16Buf is a reusable FP16 conversion buffer for activations (A or B).
	fp16Buf     unsafe.Pointer
	fp16BufSize int

	// scaleOne is a persistent device float32 with value 1.0, used as the
	// scale pointer for FP16 activations (which need no additional scaling).
	scaleOne unsafe.Pointer
}

// ensure returns fp16Buf, growing it if needed. The returned pointer is owned
// by the scratchpad and must NOT be freed by the caller.
func (s *fp8Scratchpad) ensure(pool gpuapi.MemPool, deviceID, byteSize int) (unsafe.Pointer, error) {
	if s.fp16Buf != nil && s.fp16BufSize >= byteSize {
		return s.fp16Buf, nil
	}
	// Grow: free old, allocate new.
	if s.fp16Buf != nil {
		pool.Free(deviceID, s.fp16Buf, s.fp16BufSize)
		s.fp16Buf = nil
		s.fp16BufSize = 0
	}
	ptr, err := pool.Alloc(deviceID, byteSize)
	if err != nil {
		return nil, err
	}
	s.fp16Buf = ptr
	s.fp16BufSize = byteSize
	return ptr, nil
}

// free releases all scratchpad device memory back to the pool.
func (s *fp8Scratchpad) free(pool gpuapi.MemPool, deviceID int) {
	if s.fp16Buf != nil {
		pool.Free(deviceID, s.fp16Buf, s.fp16BufSize)
		s.fp16Buf = nil
		s.fp16BufSize = 0
	}
	if s.scaleOne != nil {
		pool.Free(deviceID, s.scaleOne, f32Size)
		s.scaleOne = nil
	}
}

// getFP8Scratch returns the engine's FP8 scratchpad, initializing it lazily.
// The scaleOne device pointer is uploaded once on first call.
func (e *GPUEngine[T]) getFP8Scratch() (*fp8Scratchpad, error) {
	if e.fp8Scratch != nil {
		return e.fp8Scratch, nil
	}
	s := &fp8Scratchpad{}
	// Allocate and upload scale = 1.0 (used for FP16 activations).
	ptr, err := e.pool.Alloc(e.deviceID, f32Size)
	if err != nil {
		return nil, fmt.Errorf("fp8Scratch: alloc scaleOne: %w", err)
	}
	one := float32(1.0)
	if err := e.runtime.Memcpy(ptr, unsafe.Pointer(&one), f32Size, gpuapi.MemcpyHostToDevice); err != nil {
		e.pool.Free(e.deviceID, ptr, f32Size)
		return nil, fmt.Errorf("fp8Scratch: upload scaleOne: %w", err)
	}
	s.scaleOne = ptr
	e.fp8Scratch = s
	return s, nil
}

// getLtHandle returns the engine's cuBLASLt handle, creating it lazily.
func (e *GPUEngine[T]) getLtHandle() (*cublas.LtHandle, error) {
	if e.ltHandle != nil {
		return e.ltHandle, nil
	}
	h, err := cublas.LtCreateHandle()
	if err != nil {
		return nil, err
	}
	e.ltHandle = h
	return h, nil
}

// matMulFP8 handles MatMul where A has FP8E4M3Storage (FP8 weights as A).
// A is [M, K] in FP8 E4M3, B is [K, N] in FP32 -> C is [M, N] in FP32.
// Uses cublasLtMatmul with per-tensor scaling.
func (e *GPUEngine[T]) matMulFP8(
	ctx context.Context,
	fs *tensor.FP8E4M3Storage,
	a, b *tensor.TensorNumeric[T],
	dst ...*tensor.TensorNumeric[T],
) (*tensor.TensorNumeric[T], error) {
	aShape := a.Shape()
	bShape := b.Shape()

	if len(aShape) < 2 || len(bShape) < 2 {
		return e.cpu.MatMul(ctx, a, b, dst...)
	}
	if len(aShape) > 2 || len(bShape) > 2 {
		return e.cpu.MatMul(ctx, a, b, dst...)
	}

	e.setDevice()

	ltH, err := e.getLtHandle()
	if err != nil {
		return e.cpu.MatMul(ctx, a, b, dst...)
	}

	scratch, err := e.getFP8Scratch()
	if err != nil {
		return e.cpu.MatMul(ctx, a, b, dst...)
	}

	m := aShape[0]
	k := aShape[1]
	n := bShape[1]

	// Get FP8 device pointer for A (pre-uploaded or upload now).
	var devA unsafe.Pointer
	var freeA func()
	if ptr, _, _ := fs.GPUPtr(); ptr != nil {
		devA = ptr
		freeA = func() {}
	} else {
		aBytes := fs.RawBytes()
		devA, err = e.pool.Alloc(e.deviceID, len(aBytes))
		if err != nil {
			return e.cpu.MatMul(ctx, a, b, dst...)
		}
		freeA = func() { e.pool.Free(e.deviceID, devA, len(aBytes)) }
		if err := e.runtime.Memcpy(devA, unsafe.Pointer(&aBytes[0]), len(aBytes), gpuapi.MemcpyHostToDevice); err != nil {
			freeA()
			return e.cpu.MatMul(ctx, a, b, dst...)
		}
	}
	defer freeA()

	// Get A scale pointer on GPU.
	var scaleAPtr unsafe.Pointer
	var freeScaleA func()
	if ptr := fs.ScaleGPUPtr(); ptr != nil {
		scaleAPtr = ptr
		freeScaleA = func() {}
	} else {
		scaleAPtr, err = e.pool.Alloc(e.deviceID, f32Size)
		if err != nil {
			return e.cpu.MatMul(ctx, a, b, dst...)
		}
		freeScaleA = func() { e.pool.Free(e.deviceID, scaleAPtr, f32Size) }
		scale := fs.Scale()
		if err := e.runtime.Memcpy(scaleAPtr, unsafe.Pointer(&scale), f32Size, gpuapi.MemcpyHostToDevice); err != nil {
			freeScaleA()
			return e.cpu.MatMul(ctx, a, b, dst...)
		}
	}
	defer freeScaleA()

	// Get F32 device pointer for B, convert to FP16 using scratchpad buffer.
	devBF32, cleanupB, err := getDevicePtr(e, b)
	if err != nil {
		return e.cpu.MatMul(ctx, a, b, dst...)
	}
	defer cleanupB()

	bElems := k * n
	fp16BSize := bElems * fp16Size
	fp16B, err := scratch.ensure(e.pool, e.deviceID, fp16BSize)
	if err != nil {
		return e.cpu.MatMul(ctx, a, b, dst...)
	}

	if err := e.kernels.F32ToFP16(devBF32, fp16B, bElems, e.stream); err != nil {
		return nil, fmt.Errorf("matMulFP8: f32->fp16 B: %w", err)
	}

	// B scale = 1.0 — use persistent scaleOne from scratchpad.
	scaleBPtr := scratch.scaleOne

	// Allocate FP32 output.
	cElems := m * n
	cSize := cElems * f32Size
	devC, err := e.pool.Alloc(e.deviceID, cSize)
	if err != nil {
		return nil, fmt.Errorf("matMulFP8: alloc output: %w", err)
	}

	scaleAVal := fs.Scale()
	scaleBVal := float32(1.0)
	log.Printf("[FP8 diag] matMulFP8: m=%d n=%d k=%d scaleA=%.6g (fp8 weight) scaleB=%.6g (fp16 act) scaleAPtr=%v scaleBPtr=%v",
		m, n, k, scaleAVal, scaleBVal, scaleAPtr, scaleBPtr)

	if err := ltMatmulFP8(ltH, m, n, k, devA, scaleAPtr, cublas.CudaR8F_E4M3, fp16B, scaleBPtr, cublas.CudaR16F, devC, e.stream); err != nil {
		e.pool.Free(e.deviceID, devC, cSize)
		return nil, fmt.Errorf("matMulFP8: cublasLtMatmul: %w", err)
	}

	return makeGPUResult[T](e, []int{m, n}, devC, cElems, dst...)
}

// matMulFP8BWeight handles MatMul where B has FP8E4M3Storage (FP8 weights as B).
// A is [batch..., M, K] in FP32, B is [K, N] in FP8 E4M3 -> C is [batch..., M, N] in FP32.
// Uses cublasLtMatmul with per-tensor scaling.
func (e *GPUEngine[T]) matMulFP8BWeight(
	ctx context.Context,
	a *tensor.TensorNumeric[T],
	fs *tensor.FP8E4M3Storage,
	b *tensor.TensorNumeric[T],
	dst ...*tensor.TensorNumeric[T],
) (*tensor.TensorNumeric[T], error) {
	aShape := a.Shape()
	bShape := b.Shape()

	if len(aShape) < 2 || len(bShape) < 2 {
		return e.cpu.MatMul(ctx, a, b, dst...)
	}
	if len(bShape) > 2 {
		return e.cpu.MatMul(ctx, a, b, dst...)
	}

	e.setDevice()

	ltH, err := e.getLtHandle()
	if err != nil {
		return e.cpu.MatMul(ctx, a, b, dst...)
	}

	scratch, err := e.getFP8Scratch()
	if err != nil {
		return e.cpu.MatMul(ctx, a, b, dst...)
	}

	// Flatten A's batch dims: [batch..., m, k] -> [m_total, k]
	k := aShape[len(aShape)-1]
	m := 1
	for i := 0; i < len(aShape)-1; i++ {
		m *= aShape[i]
	}
	n := bShape[1]

	// Build output shape: [batch..., m_last, n].
	outShape := make([]int, len(aShape))
	copy(outShape, aShape[:len(aShape)-1])
	outShape[len(outShape)-1] = n

	// Get FP8 device pointer for B (pre-uploaded or upload now).
	var devB unsafe.Pointer
	var freeB func()
	if ptr, _, _ := fs.GPUPtr(); ptr != nil {
		devB = ptr
		freeB = func() {}
	} else {
		bBytes := fs.RawBytes()
		devB, err = e.pool.Alloc(e.deviceID, len(bBytes))
		if err != nil {
			return e.cpu.MatMul(ctx, a, b, dst...)
		}
		freeB = func() { e.pool.Free(e.deviceID, devB, len(bBytes)) }
		if err := e.runtime.Memcpy(devB, unsafe.Pointer(&bBytes[0]), len(bBytes), gpuapi.MemcpyHostToDevice); err != nil {
			freeB()
			return e.cpu.MatMul(ctx, a, b, dst...)
		}
	}
	defer freeB()

	// Get B scale pointer on GPU.
	var scaleBPtr unsafe.Pointer
	var freeScaleB func()
	if ptr := fs.ScaleGPUPtr(); ptr != nil {
		scaleBPtr = ptr
		freeScaleB = func() {}
	} else {
		scaleBPtr, err = e.pool.Alloc(e.deviceID, f32Size)
		if err != nil {
			return e.cpu.MatMul(ctx, a, b, dst...)
		}
		freeScaleB = func() { e.pool.Free(e.deviceID, scaleBPtr, f32Size) }
		scale := fs.Scale()
		if err := e.runtime.Memcpy(scaleBPtr, unsafe.Pointer(&scale), f32Size, gpuapi.MemcpyHostToDevice); err != nil {
			freeScaleB()
			return e.cpu.MatMul(ctx, a, b, dst...)
		}
	}
	defer freeScaleB()

	// Convert A from FP32 to FP16 using scratchpad buffer.
	devAF32, cleanupA, err := getDevicePtr(e, a)
	if err != nil {
		return e.cpu.MatMul(ctx, a, b, dst...)
	}
	defer cleanupA()

	aElems := m * k
	fp16ASize := aElems * fp16Size
	fp16A, err := scratch.ensure(e.pool, e.deviceID, fp16ASize)
	if err != nil {
		return e.cpu.MatMul(ctx, a, b, dst...)
	}

	if err := e.kernels.F32ToFP16(devAF32, fp16A, aElems, e.stream); err != nil {
		return nil, fmt.Errorf("matMulFP8BWeight: f32->fp16 A: %w", err)
	}

	// A scale = 1.0 — use persistent scaleOne from scratchpad.
	scaleAPtr := scratch.scaleOne
	scaleAVal := float32(1.0)

	// Allocate FP32 output.
	cElems := m * n
	cSize := cElems * f32Size
	devC, err := e.pool.Alloc(e.deviceID, cSize)
	if err != nil {
		return nil, fmt.Errorf("matMulFP8BWeight: alloc output: %w", err)
	}

	// cublasLtMatmul: A_fp16 [m,k] x B_fp8 [k,n] -> C_f32 [m,n]
	scaleBVal := fs.Scale()
	log.Printf("[FP8 diag] matMulFP8BWeight: m=%d n=%d k=%d scaleA=%.6g (fp16 act) scaleB=%.6g (fp8 weight) scaleAPtr=%v scaleBPtr=%v",
		m, n, k, scaleAVal, scaleBVal, scaleAPtr, scaleBPtr)

	if err := ltMatmulFP8(ltH, m, n, k, fp16A, scaleAPtr, cublas.CudaR16F, devB, scaleBPtr, cublas.CudaR8F_E4M3, devC, e.stream); err != nil {
		e.pool.Free(e.deviceID, devC, cSize)
		return nil, fmt.Errorf("matMulFP8BWeight: cublasLtMatmul: %w", err)
	}

	return makeGPUResult[T](e, outShape, devC, cElems, dst...)
}

// ltMatmulFP8 performs C = A * B using cublasLtMatmul with FP8/FP16 mixed inputs.
// A is [m,k], B is [k,n], C is [m,n] in FP32 output.
// aPtr and bPtr are device pointers to the input matrices.
// aType and bType specify the CUDA data types (CudaR8F_E4M3 or CudaR16F).
// scaleA and scaleB are device pointers to per-tensor float32 scale factors.
// For FP8 inputs, the scale is the absmax scale; for FP16 inputs, scale = 1.0.
//
// cublasLt uses column-major layout. For row-major A[m,k] * B[k,n] = C[m,n]:
// We treat the row-major data as column-major transposed:
//   - A_rm[m,k] is B_cm[k,m] (col-major)
//   - B_rm[k,n] is A_cm[n,k] (col-major)
//   - C_rm[m,n] is C_cm[n,m] (col-major)
//
// So we compute: C_cm = A_cm * B_cm, i.e. [n,k] * [k,m] = [n,m]
func ltMatmulFP8(
	ltH *cublas.LtHandle,
	m, n, k int,
	aPtr unsafe.Pointer, scaleA unsafe.Pointer, aType cublas.CudaDataType,
	bPtr unsafe.Pointer, scaleB unsafe.Pointer, bType cublas.CudaDataType,
	cPtr unsafe.Pointer,
	stream gpuapi.Stream,
) error {
	// Create matmul descriptor: FP32 compute, FP32 scale type.
	desc, err := cublas.CreateMatmulDesc(cublas.LtComputeF32, cublas.CudaR32F)
	if err != nil {
		return fmt.Errorf("CreateMatmulDesc: %w", err)
	}
	defer desc.Destroy()

	// Set scale pointers for A and B (device pointers to float32).
	// Row-major to col-major swap: cuBLAS A = our B, cuBLAS B = our A.
	if err := desc.SetAttribute(cublas.LtMatmulDescAScalePointer, unsafe.Pointer(&scaleB), int(unsafe.Sizeof(scaleB))); err != nil {
		return fmt.Errorf("set A scale: %w", err)
	}
	if err := desc.SetAttribute(cublas.LtMatmulDescBScalePointer, unsafe.Pointer(&scaleA), int(unsafe.Sizeof(scaleA))); err != nil {
		return fmt.Errorf("set B scale: %w", err)
	}

	// Determine data types for cuBLAS A (our B) and cuBLAS B (our A).
	// We always pass the FP8 matrix and the FP16 matrix in the right positions.
	// After the row-major to col-major swap:
	//   cuBLAS-A = B_rm[k,n] -> col-major [n,k], leading dim = n
	//   cuBLAS-B = A_rm[m,k] -> col-major [k,m], leading dim = k
	//   cuBLAS-C = C_rm[m,n] -> col-major [n,m], leading dim = n

	// Create matrix layouts (column-major after row/col swap).
	// cuBLAS-A (our B): [n, k], ld = n — use bType since cuBLAS-A = our B
	layoutA, err := cublas.CreateMatrixLayout(bType, n, k, n)
	if err != nil {
		return fmt.Errorf("layout A: %w", err)
	}
	defer layoutA.Destroy()

	// cuBLAS-B (our A): [k, m], ld = k — use aType since cuBLAS-B = our A
	layoutB, err := cublas.CreateMatrixLayout(aType, k, m, k)
	if err != nil {
		return fmt.Errorf("layout B: %w", err)
	}
	defer layoutB.Destroy()

	// cuBLAS-C/D (output): [n, m], ld = n, FP32
	layoutC, err := cublas.CreateMatrixLayout(cublas.CudaR32F, n, m, n)
	if err != nil {
		return fmt.Errorf("layout C: %w", err)
	}
	defer layoutC.Destroy()

	layoutD, err := cublas.CreateMatrixLayout(cublas.CudaR32F, n, m, n)
	if err != nil {
		return fmt.Errorf("layout D: %w", err)
	}
	defer layoutD.Destroy()

	// Create preference and get heuristic algorithm.
	pref, err := cublas.CreateMatmulPreference()
	if err != nil {
		return fmt.Errorf("CreateMatmulPreference: %w", err)
	}
	defer pref.Destroy()

	results, err := cublas.MatmulAlgoGetHeuristic(ltH, desc, layoutA, layoutB, layoutC, layoutD, pref, 1)
	if err != nil {
		return fmt.Errorf("MatmulAlgoGetHeuristic: %w", err)
	}
	if len(results) == 0 {
		return fmt.Errorf("no suitable cublasLt algorithm found for FP8 matmul")
	}

	log.Printf("[FP8 diag] ltMatmulFP8: m=%d n=%d k=%d aType=%d bType=%d scaleA=%v scaleB=%v aPtr=%v bPtr=%v",
		m, n, k, aType, bType, scaleA, scaleB, aPtr, bPtr)

	// alpha = 1.0, beta = 0.0 (host scalars, FP32).
	alpha := float32(1.0)
	beta := float32(0.0)

	var streamPtr uintptr
	if stream != nil {
		streamPtr = uintptr(stream.Ptr())
	}

	// cublasLtMatmul: C = alpha * cuBLAS-A * cuBLAS-B + beta * C
	// cuBLAS-A = our B (row-major), cuBLAS-B = our A (row-major)
	return cublas.LtMatmul(
		ltH, desc,
		unsafe.Pointer(&alpha),
		bPtr, layoutA, // cuBLAS-A = our B
		aPtr, layoutB, // cuBLAS-B = our A
		unsafe.Pointer(&beta),
		cPtr, layoutC,
		cPtr, layoutD,
		&results[0],
		nil, 0, // no workspace
		streamPtr,
	)
}
