//go:build cuda

package compute

import (
	"context"
	"fmt"
	"sync/atomic"
	"unsafe"

	"github.com/zerfoo/zerfoo/internal/cublas"
	"github.com/zerfoo/zerfoo/internal/cuda"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
)

// GPUEngine is a CUDA-accelerated implementation of the Engine interface.
// MatMul uses cuBLAS for maximum performance. Elementwise, scalar, activation,
// and math operations use native CUDA kernels for float32 types.
// Operations without GPU kernels delegate to CPUEngine.
//
// GPUEngine uses a device-resident pipeline: output tensors have GPUStorage
// so data stays on GPU between chained operations. A memory pool avoids
// per-operation cudaMalloc/cudaFree, and a dedicated CUDA stream enables
// async kernel execution.
type GPUEngine[T tensor.Numeric] struct {
	cpu    *CPUEngine[T]
	handle *cublas.Handle
	pool   *cuda.MemPool
	stream *cuda.Stream

	// oomFallbackCount tracks how many times an OOM triggered CPU fallback.
	oomFallbackCount atomic.Int64
}

// NewGPUEngine creates a new GPUEngine backed by a cuBLAS handle, memory pool,
// and CUDA stream. Call Close() when done to release all resources.
func NewGPUEngine[T tensor.Numeric](ops numeric.Arithmetic[T]) (*GPUEngine[T], error) {
	h, err := cublas.CreateHandle()
	if err != nil {
		return nil, fmt.Errorf("failed to create cuBLAS handle: %w", err)
	}

	stream, err := cuda.CreateStream()
	if err != nil {
		_ = h.Destroy()

		return nil, fmt.Errorf("failed to create CUDA stream: %w", err)
	}

	if err := h.SetStream(stream.Ptr()); err != nil {
		_ = stream.Destroy()
		_ = h.Destroy()

		return nil, fmt.Errorf("failed to set cuBLAS stream: %w", err)
	}

	return &GPUEngine[T]{
		cpu:    NewCPUEngine(ops),
		handle: h,
		pool:   cuda.NewMemPool(),
		stream: stream,
	}, nil
}

// Close releases the cuBLAS handle, CUDA stream, and drains the memory pool.
// The engine must not be used after Close.
func (e *GPUEngine[T]) Close() error {
	var firstErr error

	if e.pool != nil {
		if err := e.pool.Drain(); err != nil && firstErr == nil {
			firstErr = err
		}
	}

	if e.stream != nil {
		if err := e.stream.Destroy(); err != nil && firstErr == nil {
			firstErr = err
		}
	}

	if e.handle != nil {
		if err := e.handle.Destroy(); err != nil && firstErr == nil {
			firstErr = err
		}
	}

	return firstErr
}

// OOMFallbackCount returns the number of times GPU OOM triggered CPU fallback.
func (e *GPUEngine[T]) OOMFallbackCount() int64 {
	return e.oomFallbackCount.Load()
}

// Ops returns the arithmetic ops for this engine.
func (e *GPUEngine[T]) Ops() numeric.Arithmetic[T] { return e.cpu.Ops() }

// streamPtr returns the raw stream pointer for kernel calls.
func (e *GPUEngine[T]) streamPtr() unsafe.Pointer {
	if e.stream != nil {
		return e.stream.Ptr()
	}

	return nil
}

// MatMul performs matrix multiplication using cuBLAS for float32 tensors.
// For non-float32 types, it falls back to the CPU implementation.
// Supports 2D matrices and batched matmul (3D+ tensors).
func (e *GPUEngine[T]) MatMul(ctx context.Context, a, b *tensor.TensorNumeric[T], dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	// Only float32 has a cuBLAS path; fall back for other types.
	var zero T
	if _, ok := any(zero).(float32); !ok {
		return e.cpu.MatMul(ctx, a, b, dst...)
	}

	if a == nil || b == nil {
		return nil, fmt.Errorf("MatMul: input tensors must not be nil")
	}

	aShape := a.Shape()
	bShape := b.Shape()

	if len(aShape) < 2 || len(bShape) < 2 {
		return nil, fmt.Errorf("MatMul: tensors must have at least 2 dimensions, got %d and %d", len(aShape), len(bShape))
	}

	// Extract matrix dimensions from the last two axes.
	aRows := aShape[len(aShape)-2]
	aCols := aShape[len(aShape)-1]
	bRows := bShape[len(bShape)-2]
	bCols := bShape[len(bShape)-1]

	if aCols != bRows {
		return nil, fmt.Errorf("MatMul: incompatible shapes %v and %v (inner dimensions %d != %d)", aShape, bShape, aCols, bRows)
	}

	m, k, n := aRows, aCols, bCols

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

	// For batched matmul: a has batch dims, b may be unbatched (broadcast).
	if bBatchSize != 1 && aBatchSize != bBatchSize {
		return nil, fmt.Errorf("MatMul: batch dimensions %v and %v are incompatible", aBatch, bBatch)
	}

	batchSize := aBatchSize
	if bBatchSize > batchSize {
		batchSize = bBatchSize
	}

	// Build output shape.
	outShape := make([]int, 0, len(aShape))
	outShape = append(outShape, aBatch...)
	outShape = append(outShape, m, n)

	// Get device pointers for inputs.
	devA, cleanupA, err := getDevicePtr(e, a)
	if err != nil {
		e.oomFallbackCount.Add(1)

		return e.cpu.MatMul(ctx, a, b, dst...)
	}

	defer cleanupA()

	devB, cleanupB, err := getDevicePtr(e, b)
	if err != nil {
		e.oomFallbackCount.Add(1)

		return e.cpu.MatMul(ctx, a, b, dst...)
	}

	defer cleanupB()

	elemSize := int(unsafe.Sizeof(float32(0)))
	aMatSize := m * k
	bMatSize := k * n
	cMatSize := m * n

	// Allocate device output.
	devCTotal, err := e.pool.Alloc(batchSize * cMatSize * elemSize)
	if err != nil {
		e.oomFallbackCount.Add(1)

		return e.cpu.MatMul(ctx, a, b, dst...)
	}

	s := e.streamPtr()

	for batch := range batchSize {
		aOff := batch * aMatSize * elemSize
		bOff := 0
		if bBatchSize > 1 {
			bOff = batch * bMatSize * elemSize
		}

		cOff := batch * cMatSize * elemSize

		batchDevA := unsafe.Add(devA, aOff)
		batchDevB := unsafe.Add(devB, bOff)
		batchDevC := unsafe.Add(devCTotal, cOff)

		// cuBLAS Sgemm
		if err := cublas.Sgemm(e.handle, m, n, k, 1.0, batchDevA, batchDevB, 0.0, batchDevC); err != nil {
			e.pool.Free(devCTotal, batchSize*cMatSize*elemSize)

			return nil, fmt.Errorf("MatMul: cublasSgemm batch %d: %w", batch, err)
		}
	}

	// Synchronize stream before creating the storage.
	if e.stream != nil {
		if err := e.stream.Synchronize(); err != nil {
			e.pool.Free(devCTotal, batchSize*cMatSize*elemSize)

			return nil, fmt.Errorf("MatMul: stream sync: %w", err)
		}
	}

	_ = s // stream used via cuBLAS handle

	return makeGPUResult[T](outShape, devCTotal, batchSize*cMatSize, dst...)
}

// --- GPU-accelerated and fallback methods ---

func (e *GPUEngine[T]) UnaryOp(ctx context.Context, a *tensor.TensorNumeric[T], op func(T) T, dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	return e.cpu.UnaryOp(ctx, a, op, dst...)
}

func (e *GPUEngine[T]) Add(ctx context.Context, a, b *tensor.TensorNumeric[T], dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	return e.gpuAdd(ctx, a, b, dst...)
}

func (e *GPUEngine[T]) Sub(ctx context.Context, a, b *tensor.TensorNumeric[T], dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	return e.gpuSub(ctx, a, b, dst...)
}

func (e *GPUEngine[T]) Mul(ctx context.Context, a, b *tensor.TensorNumeric[T], dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	return e.gpuMul(ctx, a, b, dst...)
}

func (e *GPUEngine[T]) Div(ctx context.Context, a, b *tensor.TensorNumeric[T], dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	return e.gpuDiv(ctx, a, b, dst...)
}

func (e *GPUEngine[T]) Transpose(ctx context.Context, a *tensor.TensorNumeric[T], axes []int, dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	return e.cpu.Transpose(ctx, a, axes, dst...)
}

func (e *GPUEngine[T]) Sum(ctx context.Context, a *tensor.TensorNumeric[T], axis int, keepDims bool, dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	return e.gpuSum(ctx, a, axis, keepDims, dst...)
}

func (e *GPUEngine[T]) Exp(ctx context.Context, a *tensor.TensorNumeric[T], dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	return e.gpuExp(ctx, a, dst...)
}

func (e *GPUEngine[T]) Log(ctx context.Context, a *tensor.TensorNumeric[T], dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	return e.gpuLog(ctx, a, dst...)
}

func (e *GPUEngine[T]) Tanh(ctx context.Context, a *tensor.TensorNumeric[T], dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	return e.gpuTanh(ctx, a, dst...)
}

func (e *GPUEngine[T]) TanhPrime(ctx context.Context, a, upstream *tensor.TensorNumeric[T], dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	return e.gpuTanhPrime(ctx, a, upstream, dst...)
}

func (e *GPUEngine[T]) Pow(ctx context.Context, base, exponent *tensor.TensorNumeric[T], dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	return e.gpuPow(ctx, base, exponent, dst...)
}

func (e *GPUEngine[T]) Zero(ctx context.Context, a *tensor.TensorNumeric[T]) error {
	return e.cpu.Zero(ctx, a)
}

func (e *GPUEngine[T]) Zeros(ctx context.Context, a *tensor.TensorNumeric[T], shape []int) error {
	return e.cpu.Zeros(ctx, a, shape)
}

func (e *GPUEngine[T]) Copy(ctx context.Context, dst, src *tensor.TensorNumeric[T]) error {
	return e.cpu.Copy(ctx, dst, src)
}

func (e *GPUEngine[T]) Gather(ctx context.Context, params *tensor.TensorNumeric[T], indices *tensor.TensorNumeric[int], output *tensor.TensorNumeric[T]) error {
	return e.cpu.Gather(ctx, params, indices, output)
}

func (e *GPUEngine[T]) ScatterAdd(ctx context.Context, dEmbeddingTable *tensor.TensorNumeric[T], indices *tensor.TensorNumeric[int], dOut *tensor.TensorNumeric[T]) error {
	return e.cpu.ScatterAdd(ctx, dEmbeddingTable, indices, dOut)
}

func (e *GPUEngine[T]) RandomUniform(ctx context.Context, t *tensor.TensorNumeric[T], minVal, maxVal T) error {
	return e.cpu.RandomUniform(ctx, t, minVal, maxVal)
}

func (e *GPUEngine[T]) Fill(ctx context.Context, t *tensor.TensorNumeric[T], value T) error {
	return e.gpuFill(ctx, t, value)
}

func (e *GPUEngine[T]) MulScalar(ctx context.Context, a *tensor.TensorNumeric[T], scalar T, dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	return e.gpuMulScalar(ctx, a, scalar, dst...)
}

func (e *GPUEngine[T]) DivScalar(ctx context.Context, a *tensor.TensorNumeric[T], scalar T, dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	return e.gpuDivScalar(ctx, a, scalar, dst...)
}

func (e *GPUEngine[T]) Softmax(ctx context.Context, a *tensor.TensorNumeric[T], axis int, dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	return e.gpuSoftmax(ctx, a, axis, dst...)
}

func (e *GPUEngine[T]) ReduceSum(ctx context.Context, a *tensor.TensorNumeric[T], axis int, keepDims bool, dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	return e.gpuReduceSum(ctx, a, axis, keepDims, dst...)
}

func (e *GPUEngine[T]) AddScalar(ctx context.Context, a *tensor.TensorNumeric[T], scalar T, dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	return e.gpuAddScalar(ctx, a, scalar, dst...)
}

func (e *GPUEngine[T]) Sqrt(ctx context.Context, a *tensor.TensorNumeric[T], dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	return e.gpuSqrt(ctx, a, dst...)
}

func (e *GPUEngine[T]) Split(ctx context.Context, a *tensor.TensorNumeric[T], numSplits int, axis int) ([]*tensor.TensorNumeric[T], error) {
	return e.cpu.Split(ctx, a, numSplits, axis)
}

func (e *GPUEngine[T]) Concat(ctx context.Context, tensors []*tensor.TensorNumeric[T], axis int, dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	return e.cpu.Concat(ctx, tensors, axis, dst...)
}

func (e *GPUEngine[T]) Repeat(ctx context.Context, a *tensor.TensorNumeric[T], axis int, repetitions int, dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	return e.cpu.Repeat(ctx, a, axis, repetitions, dst...)
}

func (e *GPUEngine[T]) OneHot(ctx context.Context, input *tensor.TensorNumeric[int], depth int, dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	return e.cpu.OneHot(ctx, input, depth, dst...)
}

func (e *GPUEngine[T]) Reshape(ctx context.Context, a *tensor.TensorNumeric[T], shape []int, dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	return e.cpu.Reshape(ctx, a, shape, dst...)
}

func (e *GPUEngine[T]) ReduceMean(ctx context.Context, a *tensor.TensorNumeric[T], axis int, keepDims bool, dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	return e.gpuReduceMean(ctx, a, axis, keepDims, dst...)
}

func (e *GPUEngine[T]) Rsqrt(ctx context.Context, a *tensor.TensorNumeric[T], dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	return e.gpuRsqrt(ctx, a, dst...)
}

// Static type assertion: GPUEngine satisfies Engine.
var _ Engine[float32] = (*GPUEngine[float32])(nil)
