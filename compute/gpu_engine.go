//go:build cuda

package compute

import (
	"context"
	"fmt"
	"sync/atomic"
	"unsafe"

	"github.com/zerfoo/float16"
	"github.com/zerfoo/zerfoo/internal/gpuapi"
	"github.com/zerfoo/zerfoo/log"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
)

// GPUEngine is a GPU-accelerated implementation of the Engine interface.
// MatMul uses BLAS for maximum performance. Elementwise, scalar, activation,
// and math operations use native GPU kernels for float32 types.
// Operations without GPU kernels delegate to CPUEngine.
//
// GPUEngine uses a device-resident pipeline: output tensors have GPUStorage
// so data stays on GPU between chained operations. A memory pool avoids
// per-operation malloc/free, and a dedicated stream enables async kernel execution.
//
// GPUEngine is backend-agnostic via the GRAL interfaces (internal/gpuapi/).
// The CUDA, ROCm, and OpenCL adapters implement these interfaces.
type GPUEngine[T tensor.Numeric] struct {
	cpu      *CPUEngine[T]
	runtime  gpuapi.Runtime
	blas     gpuapi.BLAS
	dnn      gpuapi.DNN
	kernels  gpuapi.KernelRunner
	pool     gpuapi.MemPool
	stream   gpuapi.Stream
	logger   log.Logger
	deviceID int

	// oomFallbackCount tracks how many times an OOM triggered CPU fallback.
	oomFallbackCount atomic.Int64
}

// NewGPUEngine creates a new GPUEngine backed by CUDA via the GRAL abstraction.
// An optional deviceID selects the GPU (default 0).
// Call Close() when done to release all resources.
func NewGPUEngine[T tensor.Numeric](ops numeric.Arithmetic[T], deviceID ...int) (*GPUEngine[T], error) {
	dev := 0
	if len(deviceID) > 0 {
		dev = deviceID[0]
	}

	rt := gpuapi.NewCUDARuntime()
	if err := rt.SetDevice(dev); err != nil {
		return nil, fmt.Errorf("failed to set GPU device %d: %w", dev, err)
	}

	blas, err := gpuapi.NewCUDABlas()
	if err != nil {
		return nil, fmt.Errorf("failed to create BLAS handle: %w", err)
	}

	stream, err := rt.CreateStream()
	if err != nil {
		_ = blas.Destroy()
		return nil, fmt.Errorf("failed to create GPU stream: %w", err)
	}

	if err := blas.SetStream(stream); err != nil {
		_ = stream.Destroy()
		_ = blas.Destroy()
		return nil, fmt.Errorf("failed to set BLAS stream: %w", err)
	}

	dnn, err := gpuapi.NewCUDADNN()
	if err != nil {
		_ = stream.Destroy()
		_ = blas.Destroy()
		return nil, fmt.Errorf("failed to create DNN handle: %w", err)
	}

	if err := dnn.SetStream(stream); err != nil {
		_ = dnn.Destroy()
		_ = stream.Destroy()
		_ = blas.Destroy()
		return nil, fmt.Errorf("failed to set DNN stream: %w", err)
	}

	l := log.Nop()
	l.Info("gpu engine initialized", "device", fmt.Sprintf("%d", dev), "pool", "enabled", "stream", "enabled")

	return &GPUEngine[T]{
		cpu:      NewCPUEngine(ops),
		runtime:  rt,
		blas:     blas,
		dnn:      dnn,
		kernels:  gpuapi.NewCUDAKernels(),
		pool:     gpuapi.NewCUDAMemPool(),
		stream:   stream,
		logger:   l,
		deviceID: dev,
	}, nil
}

// DeviceID returns the GPU device ID this engine is bound to.
func (e *GPUEngine[T]) DeviceID() int { return e.deviceID }

// setDevice ensures the correct GPU device context for the calling goroutine.
func (e *GPUEngine[T]) setDevice() {
	_ = e.runtime.SetDevice(e.deviceID)
}

// SetLogger replaces the engine's logger.
func (e *GPUEngine[T]) SetLogger(l log.Logger) {
	if l == nil {
		l = log.Nop()
	}
	e.logger = l
	e.cpu.SetLogger(l)
}

// UploadWeights copies CPU-resident tensors to GPU device memory in place.
// Tensors that already have GPUStorage are skipped. Q4 quantized weights
// get their raw bytes uploaded and cached in Q4Storage to avoid per-op H2D.
// This is called once at model load time.
func (e *GPUEngine[T]) UploadWeights(tensors []*tensor.TensorNumeric[float32]) error {
	e.setDevice()
	uploaded := 0
	q4Uploaded := 0
	for _, t := range tensors {
		if t == nil {
			continue
		}
		// Upload Q4 raw bytes to GPU and cache the pointer.
		// Q4 weights are the biggest win because they're used in every MatMul
		// and are expensive to re-upload (~18 bytes per 32 floats).
		if qs, ok := any(t.GetStorage()).(*tensor.Q4Storage); ok {
			if ptr, _, _ := qs.GPUPtr(); ptr != nil {
				continue // already on GPU
			}
			rawBytes := qs.RawBytes()
			devPtr, err := e.pool.Alloc(e.deviceID, len(rawBytes))
			if err != nil {
				return fmt.Errorf("alloc Q4 GPU (shape %v): %w", t.Shape(), err)
			}
			if err := e.runtime.Memcpy(devPtr, unsafe.Pointer(&rawBytes[0]), len(rawBytes), gpuapi.MemcpyHostToDevice); err != nil {
				e.pool.Free(e.deviceID, devPtr, len(rawBytes))
				return fmt.Errorf("upload Q4 (shape %v): %w", t.Shape(), err)
			}
			qs.SetGPUPtr(devPtr, len(rawBytes), e.deviceID)
			q4Uploaded++
			continue
		}
		// Upload float32 weights to GPU. With Pow, binary ops, and
		// Split/Concat now running on GPU, float32 weights benefit from
		// staying on device (eliminates per-op H2D copies for norm weights).
		if _, ok := t.GetStorage().(*tensor.GPUStorage[float32]); ok {
			continue // already on GPU
		}
		data := t.Data()
		n := len(data)
		if n == 0 {
			continue
		}
		byteSize := n * f32Size
		devPtr, err := e.pool.Alloc(e.deviceID, byteSize)
		if err != nil {
			return fmt.Errorf("alloc f32 GPU (shape %v): %w", t.Shape(), err)
		}
		if err := e.runtime.Memcpy(devPtr, unsafe.Pointer(&data[0]), byteSize, gpuapi.MemcpyHostToDevice); err != nil {
			e.pool.Free(e.deviceID, devPtr, byteSize)
			return fmt.Errorf("upload f32 (shape %v): %w", t.Shape(), err)
		}
		gs, err := tensor.NewGPUStorageFromPtr[float32](devPtr, n, e.deviceID)
		if err != nil {
			e.pool.Free(e.deviceID, devPtr, byteSize)
			return fmt.Errorf("create GPU storage (shape %v): %w", t.Shape(), err)
		}
		t.SetStorage(gs)
		uploaded++
	}
	if uploaded > 0 || q4Uploaded > 0 {
		e.logger.Info("weights uploaded to GPU",
			"f32", fmt.Sprintf("%d", uploaded),
			"q4", fmt.Sprintf("%d", q4Uploaded),
			"device", fmt.Sprintf("%d", e.deviceID))
	}
	return nil
}

// Close releases the BLAS handle, DNN handle, GPU stream, and drains the memory pool.
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

	if e.dnn != nil {
		if err := e.dnn.Destroy(); err != nil && firstErr == nil {
			firstErr = err
		}
	}

	if e.blas != nil {
		if err := e.blas.Destroy(); err != nil && firstErr == nil {
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

// MatMul performs matrix multiplication using GPU BLAS for float32 and BFloat16
// tensors. For Q4_0 quantized tensors, uses the Q4 dequant-GEMM kernel.
// For other types, it falls back to the CPU implementation.
// Supports 2D matrices and batched matmul (3D+ tensors).
func (e *GPUEngine[T]) MatMul(ctx context.Context, a, b *tensor.TensorNumeric[T], dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	// Check for Q4 quantized storage on A.
	// Use any() to avoid impossible type assertion when T != float32.
	if qs, ok := any(a.GetStorage()).(*tensor.Q4Storage); ok {
		return e.matMulQ4(ctx, qs, a, b, dst...)
	}

	// float32 and BFloat16 have GPU BLAS paths; fall back for other types.
	var zero T
	_, isFloat32 := any(zero).(float32)
	_, isBFloat16 := any(zero).(float16.BFloat16)
	if !isFloat32 && !isBFloat16 {
		return e.cpu.MatMul(ctx, a, b, dst...)
	}

	e.setDevice()

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
		e.logger.Warn("MatMul: GPU alloc failed, falling back to CPU", "error", err.Error())

		return e.cpu.MatMul(ctx, a, b, dst...)
	}

	defer cleanupA()

	devB, cleanupB, err := getDevicePtr(e, b)
	if err != nil {
		e.oomFallbackCount.Add(1)
		e.logger.Warn("MatMul: GPU alloc failed, falling back to CPU", "error", err.Error())

		return e.cpu.MatMul(ctx, a, b, dst...)
	}

	defer cleanupB()

	elemSize := int(unsafe.Sizeof(zero))
	aMatSize := m * k
	bMatSize := k * n
	cMatSize := m * n

	// Allocate device output.
	devCTotal, err := e.pool.Alloc(e.deviceID, batchSize*cMatSize*elemSize)
	if err != nil {
		e.oomFallbackCount.Add(1)
		e.logger.Warn("MatMul: GPU output alloc failed, falling back to CPU", "error", err.Error())

		return e.cpu.MatMul(ctx, a, b, dst...)
	}

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

		var blasErr error
		if isBFloat16 {
			blasErr = e.blas.BFloat16Gemm(m, n, k, 1.0, batchDevA, batchDevB, 0.0, batchDevC)
		} else {
			blasErr = e.blas.Sgemm(m, n, k, 1.0, batchDevA, batchDevB, 0.0, batchDevC)
		}

		if blasErr != nil {
			e.pool.Free(e.deviceID, devCTotal, batchSize*cMatSize*elemSize)

			return nil, fmt.Errorf("MatMul: BLAS batch %d: %w", batch, blasErr)
		}
	}

	return makeGPUResult[T](e, outShape, devCTotal, batchSize*cMatSize, dst...)
}

// matMulQ4 handles GPU Q4_0 dequant-GEMM: C = dequant(A_q4) * B.
// Only supports unbatched 2D for now; batched Q4 falls back to CPU.
func (e *GPUEngine[T]) matMulQ4(ctx context.Context, qs *tensor.Q4Storage, a, b *tensor.TensorNumeric[T], dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	aShape := a.Shape()
	bShape := b.Shape()

	if len(aShape) < 2 || len(bShape) < 2 {
		return e.cpu.MatMul(ctx, a, b, dst...)
	}

	m := aShape[len(aShape)-2]
	k := aShape[len(aShape)-1]
	n := bShape[len(bShape)-1]

	// Only handle unbatched 2D for now.
	if len(aShape) > 2 || len(bShape) > 2 {
		return e.cpu.MatMul(ctx, a, b, dst...)
	}

	if k%32 != 0 {
		return e.cpu.MatMul(ctx, a, b, dst...)
	}

	e.setDevice()

	// Use pre-uploaded Q4 GPU pointer if available; otherwise upload now.
	var devA unsafe.Pointer
	var freeA func()
	if ptr, _, _ := qs.GPUPtr(); ptr != nil {
		devA = ptr
		freeA = func() {} // pre-uploaded; do not free
	} else {
		aBytes := qs.RawBytes()
		var err error
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

	// Upload B (float32) to GPU.
	devB, cleanupB, err := getDevicePtr(e, b)
	if err != nil {
		return e.cpu.MatMul(ctx, a, b, dst...)
	}
	defer cleanupB()

	// Allocate output C.
	cSize := m * n * int(unsafe.Sizeof(float32(0)))
	devC, err := e.pool.Alloc(e.deviceID, cSize)
	if err != nil {
		return e.cpu.MatMul(ctx, a, b, dst...)
	}

	if err := e.kernels.GemmQ4F32(devA, devB, devC, m, k, n, e.stream); err != nil {
		e.pool.Free(e.deviceID, devC, cSize)
		return e.cpu.MatMul(ctx, a, b, dst...)
	}

	return makeGPUResult[T](e, []int{m, n}, devC, m*n, dst...)
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
	if !isFloat32[T]() {
		return e.cpu.Transpose(ctx, a, axes, dst...)
	}

	// Only use GPU path for GPU-resident tensors.
	if _, ok := a.GetStorage().(*tensor.GPUStorage[T]); !ok {
		return e.cpu.Transpose(ctx, a, axes, dst...)
	}

	e.setDevice()

	shape := a.Shape()
	rank := len(shape)

	// Default: reverse axes (same as CPU Transpose with nil axes).
	if len(axes) == 0 {
		axes = make([]int, rank)
		for i := range rank {
			axes[i] = rank - 1 - i
		}
	}

	if len(axes) != rank {
		return e.cpu.Transpose(ctx, a, axes, dst...)
	}

	// Compute total elements.
	total := 1
	for _, d := range shape {
		total *= d
	}

	// Compute input strides.
	inStrides := make([]int, rank)
	stride := 1
	for i := rank - 1; i >= 0; i-- {
		inStrides[i] = stride
		stride *= shape[i]
	}

	// Compute output shape.
	outShape := make([]int, rank)
	for i, ax := range axes {
		outShape[i] = shape[ax]
	}

	devIn, cleanupIn, err := getDevicePtr(e, a)
	if err != nil {
		return e.cpu.Transpose(ctx, a, axes, dst...)
	}
	defer cleanupIn()

	byteSize := total * f32Size
	devOut, err := e.pool.Alloc(e.deviceID, byteSize)
	if err != nil {
		return e.cpu.Transpose(ctx, a, axes, dst...)
	}

	// Fast path: 2D transpose.
	if rank == 2 && axes[0] == 1 && axes[1] == 0 {
		if err := e.kernels.Transpose2D(devIn, devOut, shape[0], shape[1], e.stream); err != nil {
			e.pool.Free(e.deviceID, devOut, byteSize)
			return nil, err
		}
		return makeGPUResult[T](e, outShape, devOut, total, dst...)
	}

	// General N-D transpose via stride-based kernel.
	inStrides32 := make([]int32, rank)
	outShape32 := make([]int32, rank)
	perm32 := make([]int32, rank)
	for i := range rank {
		inStrides32[i] = int32(inStrides[i])
		outShape32[i] = int32(outShape[i])
		perm32[i] = int32(axes[i])
	}

	if err := e.kernels.TransposeND(devIn, devOut, inStrides32, outShape32, perm32, rank, total, e.stream); err != nil {
		e.pool.Free(e.deviceID, devOut, byteSize)
		return nil, err
	}

	return makeGPUResult[T](e, outShape, devOut, total, dst...)
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
	if !isFloat32[T]() {
		return e.cpu.Gather(ctx, params, indices, output)
	}

	// Only use GPU path when params are GPU-resident.
	if _, ok := params.GetStorage().(*tensor.GPUStorage[T]); !ok {
		return e.cpu.Gather(ctx, params, indices, output)
	}

	e.setDevice()

	pShape := params.Shape()
	if len(pShape) != 2 {
		return e.cpu.Gather(ctx, params, indices, output)
	}
	V := pShape[0]
	D := pShape[1]

	// Flatten indices to get N.
	idxData := indices.Data()
	N := len(idxData)
	if N == 0 {
		return nil
	}

	// Get device pointer for params (should be zero-copy since GPU-resident).
	devParams, cleanupParams, err := getDevicePtr(e, params)
	if err != nil {
		return e.cpu.Gather(ctx, params, indices, output)
	}
	defer cleanupParams()

	// Upload indices to GPU as int32 (Go int is 64-bit on amd64/arm64).
	indices32 := make([]int32, N)
	for i, v := range idxData {
		indices32[i] = int32(v)
	}
	idxByteSize := N * 4
	devIdx, err := e.pool.Alloc(e.deviceID, idxByteSize)
	if err != nil {
		return e.cpu.Gather(ctx, params, indices, output)
	}
	defer e.pool.Free(e.deviceID, devIdx, idxByteSize)

	if err := e.runtime.Memcpy(devIdx, unsafe.Pointer(&indices32[0]), idxByteSize, gpuapi.MemcpyHostToDevice); err != nil {
		return e.cpu.Gather(ctx, params, indices, output)
	}

	// Allocate output on GPU.
	outByteSize := N * D * f32Size
	devOut, err := e.pool.Alloc(e.deviceID, outByteSize)
	if err != nil {
		return e.cpu.Gather(ctx, params, indices, output)
	}

	if err := e.kernels.Gather(devParams, devIdx, devOut, N, D, V, e.stream); err != nil {
		e.pool.Free(e.deviceID, devOut, outByteSize)
		return fmt.Errorf("GPU Gather: %w", err)
	}

	// Set output storage to GPU.
	gs, err := tensor.NewGPUStorageFromPtr[T](devOut, N*D, e.deviceID)
	if err != nil {
		e.pool.Free(e.deviceID, devOut, outByteSize)
		return err
	}
	output.SetStorage(gs)

	return nil
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
	if !isFloat32[T]() {
		return e.cpu.Split(ctx, a, numSplits, axis)
	}
	gs, ok := a.GetStorage().(*tensor.GPUStorage[T])
	if !ok {
		return e.cpu.Split(ctx, a, numSplits, axis)
	}
	return e.gpuSplit(gs.Ptr(), a.Shape(), numSplits, axis)
}

func (e *GPUEngine[T]) Concat(ctx context.Context, tensors []*tensor.TensorNumeric[T], axis int, dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if !isFloat32[T]() || len(tensors) == 0 {
		return e.cpu.Concat(ctx, tensors, axis, dst...)
	}
	// Check all inputs are GPU-resident.
	ptrs := make([]unsafe.Pointer, len(tensors))
	for i, t := range tensors {
		gs, ok := t.GetStorage().(*tensor.GPUStorage[T])
		if !ok {
			return e.cpu.Concat(ctx, tensors, axis, dst...)
		}
		ptrs[i] = gs.Ptr()
	}
	return e.gpuConcat(ptrs, tensors, axis, dst...)
}

func (e *GPUEngine[T]) Repeat(ctx context.Context, a *tensor.TensorNumeric[T], axis int, repetitions int, dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	return e.cpu.Repeat(ctx, a, axis, repetitions, dst...)
}

func (e *GPUEngine[T]) OneHot(ctx context.Context, input *tensor.TensorNumeric[int], depth int, dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	return e.cpu.OneHot(ctx, input, depth, dst...)
}

func (e *GPUEngine[T]) Reshape(ctx context.Context, a *tensor.TensorNumeric[T], shape []int, dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	// For GPU-resident tensors, reshape is a zero-copy view change.
	if gs, ok := a.GetStorage().(*tensor.GPUStorage[T]); ok && isFloat32[T]() {
		// Resolve -1 dimension.
		currentSize := a.Size()
		inferredShape := make([]int, len(shape))
		copy(inferredShape, shape)
		inferIdx := -1
		knownSize := 1
		for i, d := range inferredShape {
			if d == -1 {
				inferIdx = i
			} else {
				knownSize *= d
			}
		}
		if inferIdx >= 0 {
			inferredShape[inferIdx] = currentSize / knownSize
		}
		// Verify size matches.
		newSize := 1
		for _, d := range inferredShape {
			newSize *= d
		}
		if newSize != currentSize {
			return e.cpu.Reshape(ctx, a, shape, dst...)
		}
		return tensor.NewWithStorage[T](inferredShape, gs)
	}
	return e.cpu.Reshape(ctx, a, shape, dst...)
}

func (e *GPUEngine[T]) ReduceMean(ctx context.Context, a *tensor.TensorNumeric[T], axis int, keepDims bool, dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	return e.gpuReduceMean(ctx, a, axis, keepDims, dst...)
}

func (e *GPUEngine[T]) Rsqrt(ctx context.Context, a *tensor.TensorNumeric[T], dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	return e.gpuRsqrt(ctx, a, dst...)
}

// Sync synchronizes the GPU stream, blocking until all enqueued operations complete.
// Use for benchmarking or when explicit synchronization is needed.
func (e *GPUEngine[T]) Sync() error {
	if e.stream != nil {
		return e.stream.Synchronize()
	}

	return nil
}

// Static type assertion: GPUEngine satisfies Engine.
var _ Engine[float32] = (*GPUEngine[float32])(nil)
