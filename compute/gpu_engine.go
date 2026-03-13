package compute

import (
	"context"
	"fmt"
	"os"
	"sync/atomic"
	"unsafe"

	"github.com/zerfoo/float16"
	"github.com/zerfoo/zerfoo/internal/cublas"
	"github.com/zerfoo/zerfoo/internal/cuda"
	"github.com/zerfoo/zerfoo/internal/gpuapi"
	"github.com/zerfoo/zerfoo/log"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
)

// DType selects the compute precision for GPU operations.
type DType int

const (
	// DTypeF32 uses float32 for all compute (default).
	DTypeF32 DType = iota
	// DTypeFP16 uses FP16 for elementwise ops and MatMul.
	// Activations are converted F32->FP16 before compute and FP16->F32 after.
	// Reductions (RMSNorm, Softmax) accumulate in FP32 for precision.
	DTypeFP16

	// DTypeFP8 uses FP8 E4M3 weights with FP16 compute for element-wise ops.
	// Weights are quantized to FP8 at load time, dequantized to FP16 on GPU.
	// MatMul uses cublasLtMatmul (auto-detected via FP8E4M3Storage).
	DTypeFP8
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

	// dtype selects FP16 or FP32 compute precision. When DTypeFP16,
	// elementwise ops convert F32 inputs to FP16, run FP16 kernels,
	// and convert outputs back to F32.
	dtype DType

	// ltHandle is the cuBLASLt handle, initialized lazily on first FP8 matmul.
	ltHandle *cublas.LtHandle

	// fp8Scratch holds pre-allocated, reusable FP16 conversion buffers for FP8
	// MatMul operations. Lazily initialized on first FP8 matmul call.
	// Buffers grow-only to avoid repeated reallocation.
	fp8Scratch *fp8Scratchpad

	// oomFallbackCount tracks how many times an OOM triggered CPU fallback.
	oomFallbackCount atomic.Int64

	// managedMem is true when the device supports concurrent managed memory
	// access (e.g., GB10 with NVLink-C2C). When true, weight uploads use
	// cudaMallocManaged instead of cudaMalloc + explicit H2D copy.
	managedMem bool
}

// NewGPUEngine creates a new GPUEngine backed by CUDA via the GRAL abstraction.
// An optional deviceID selects the GPU (default 0).
// Call Close() when done to release all resources.
func NewGPUEngine[T tensor.Numeric](ops numeric.Arithmetic[T], deviceID ...int) (*GPUEngine[T], error) {
	if !cuda.Available() {
		return nil, fmt.Errorf("CUDA runtime not available")
	}

	dev := 0
	if len(deviceID) > 0 {
		dev = deviceID[0]
	}

	rt := gpuapi.NewCUDARuntime()
	if err := rt.SetDevice(dev); err != nil {
		return nil, fmt.Errorf("failed to set GPU device %d: %w", dev, err)
	}

	l := log.Nop()

	var blas gpuapi.BLAS
	if gpuapi.BLASFactory != nil {
		var err error
		blas, err = gpuapi.BLASFactory()
		if err != nil {
			l.Warn("BLAS not available, MatMul will fall back to CPU", "error", err.Error())
		}
	}

	stream, err := rt.CreateStream()
	if err != nil {
		if blas != nil {
			_ = blas.Destroy()
		}
		return nil, fmt.Errorf("failed to create GPU stream: %w", err)
	}

	if blas != nil {
		if err := blas.SetStream(stream); err != nil {
			_ = stream.Destroy()
			_ = blas.Destroy()
			return nil, fmt.Errorf("failed to set BLAS stream: %w", err)
		}
	}

	var dnn gpuapi.DNN
	if gpuapi.DNNFactory != nil {
		var err error
		dnn, err = gpuapi.DNNFactory()
		if err != nil {
			l.Warn("DNN not available, cuDNN ops will return errors", "error", err.Error())
		}
	}

	if dnn != nil {
		if err := dnn.SetStream(stream); err != nil {
			_ = dnn.Destroy()
			_ = stream.Destroy()
			if blas != nil {
				_ = blas.Destroy()
			}
			return nil, fmt.Errorf("failed to set DNN stream: %w", err)
		}
	}

	// Managed memory (cudaMallocManaged) is opt-in: on GB10 it causes ~13%
	// throughput regression due to page fault overhead. Enable with
	// ZERFOO_ENABLE_MANAGED_MEM=1 after validating with cudaMemPrefetchAsync.
	managedMem := cuda.ManagedMemorySupported(dev) && os.Getenv("ZERFOO_ENABLE_MANAGED_MEM") != ""
	l.Info("gpu engine initialized", "device", fmt.Sprintf("%d", dev), "pool", "enabled", "stream", "enabled", "managedMemory", fmt.Sprintf("%v", managedMem))

	fallbackPool := cuda.NewMemPool()
	cuda.SetDefaultMemPool(fallbackPool)

	// Arena pool: 2GB pre-allocated region for per-inference intermediates.
	// On DGX Spark with 128GB unified memory, this is a small fraction.
	// Falls back to MemPool if arena is exhausted.
	const arenaSize = 2 * 1024 * 1024 * 1024 // 2 GB
	arenaPool, err := gpuapi.NewCUDAArenaPool(dev, arenaSize, fallbackPool)
	if err == nil {
		cuda.SetDefaultArenaPool(arenaPool.Inner())
	}
	if err != nil {
		l.Warn("arena pool not available, falling back to MemPool", "error", err.Error())
		bucketPool := gpuapi.NewCUDAMemPoolFrom(fallbackPool)
		return &GPUEngine[T]{
			cpu:        NewCPUEngine(ops),
			runtime:    rt,
			blas:       blas,
			dnn:        dnn,
			kernels:    gpuapi.NewCUDAKernels(),
			pool:       bucketPool,
			stream:     stream,
			logger:     l,
			deviceID:   dev,
			managedMem: managedMem,
		}, nil
	}

	return &GPUEngine[T]{
		cpu:        NewCPUEngine(ops),
		runtime:    rt,
		blas:       blas,
		dnn:        dnn,
		kernels:    gpuapi.NewCUDAKernels(),
		pool:       arenaPool,
		stream:     stream,
		logger:     l,
		deviceID:   dev,
		managedMem: managedMem,
	}, nil
}

// DeviceID returns the GPU device ID this engine is bound to.
func (e *GPUEngine[T]) DeviceID() int { return e.deviceID }

// IsManagedMemory returns true if the engine uses managed memory for
// weight uploads and the arena allocator.
func (e *GPUEngine[T]) IsManagedMemory() bool { return e.managedMem }

// ResetPool resets the arena pool, reclaiming all per-pass allocations.
// This is a no-op if the pool is not arena-backed.
func (e *GPUEngine[T]) ResetPool() {
	if arena, ok := e.pool.(*gpuapi.CUDAArenaPool); ok {
		arena.Reset()
	}
}

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

// SetDType sets the compute precision for elementwise ops and MatMul.
// DTypeFP16 enables the FP16 inference path: F32 inputs are converted to
// FP16 on GPU, FP16 kernels run, and results are converted back to F32.
func (e *GPUEngine[T]) SetDType(d DType) {
	e.dtype = d
}

// DTypeValue returns the current compute precision.
func (e *GPUEngine[T]) DTypeValue() DType {
	return e.dtype
}

// UploadWeights copies CPU-resident tensors to GPU device memory in place.
// Tensors that already have GPUStorage are skipped. Q4 quantized weights
// get their raw bytes uploaded and cached in Q4Storage to avoid per-op H2D.
// This is called once at model load time.
//
// On devices with managed memory support (e.g., GB10), weights are allocated
// with cudaMallocManaged and populated via direct CPU memcpy. The GPU can
// then access them without any explicit H2D transfer.
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
			// Use GPU-optimized separated layout: scales grouped, then packed
			// data grouped per row. This enables 128-bit vectorized loads and
			// coalesced memory access in the Q4 GEMV kernel.
			shape := t.Shape()
			blocksPerRow := 0
			if len(shape) == 2 {
				blocksPerRow = shape[1] / 32
			}
			rawBytes := qs.RawBytesGPU(blocksPerRow)
			devPtr, err := e.allocWeight(len(rawBytes))
			if err != nil {
				return fmt.Errorf("alloc Q4 GPU (shape %v): %w", t.Shape(), err)
			}
			if err := e.uploadBytes(devPtr, rawBytes); err != nil {
				_ = e.runtime.Free(devPtr)
				return fmt.Errorf("upload Q4 (shape %v): %w", t.Shape(), err)
			}
			qs.SetGPUPtr(devPtr, len(rawBytes), e.deviceID)
			q4Uploaded++
			continue
		}
		// Upload Q4_K raw bytes to GPU for fused GEMV kernel.
		// Q4_K super-blocks (144 bytes per 256 values) are uploaded contiguously.
		if qs, ok := any(t.GetStorage()).(*tensor.Q4KStorage); ok {
			if ptr, _, _ := qs.GPUPtr(); ptr != nil {
				continue // already on GPU
			}
			rawBytes := qs.RawBytes()
			devPtr, err := e.runtime.Malloc(len(rawBytes))
			if err != nil {
				return fmt.Errorf("alloc Q4_K GPU (shape %v): %w", t.Shape(), err)
			}
			if err := e.runtime.Memcpy(devPtr, unsafe.Pointer(&rawBytes[0]), len(rawBytes), gpuapi.MemcpyHostToDevice); err != nil {
				_ = e.runtime.Free(devPtr)
				return fmt.Errorf("upload Q4_K (shape %v): %w", t.Shape(), err)
			}
			qs.SetGPUPtr(devPtr, len(rawBytes), e.deviceID)
			q4Uploaded++
			continue
		}
		// Upload FP8 E4M3 raw bytes to GPU and cache the pointer.
		// FP8 weights (1 byte/element) are used with cublasLtMatmul for
		// mixed-precision MatMul (FP8 weights × FP16 activations → FP32 output).
		if fs, ok := any(t.GetStorage()).(*tensor.FP8E4M3Storage); ok {
			if ptr, _, _ := fs.GPUPtr(); ptr != nil {
				continue // already on GPU
			}
			rawBytes := fs.RawBytes()
			devPtr, err := e.allocWeight(len(rawBytes))
			if err != nil {
				return fmt.Errorf("alloc FP8 GPU (shape %v): %w", t.Shape(), err)
			}
			if err := e.uploadBytes(devPtr, rawBytes); err != nil {
				_ = e.runtime.Free(devPtr)
				return fmt.Errorf("upload FP8 (shape %v): %w", t.Shape(), err)
			}
			fs.SetGPUPtr(devPtr, len(rawBytes), e.deviceID)
			// Upload per-tensor scale factor (single float32) to GPU.
			scale := fs.Scale()
			scaleBytes := unsafe.Slice((*byte)(unsafe.Pointer(&scale)), f32Size)
			scalePtr, err := e.allocWeight(f32Size)
			if err != nil {
				return fmt.Errorf("alloc FP8 scale GPU (shape %v): %w", t.Shape(), err)
			}
			if err := e.uploadBytes(scalePtr, scaleBytes); err != nil {
				_ = e.runtime.Free(scalePtr)
				return fmt.Errorf("upload FP8 scale (shape %v): %w", t.Shape(), err)
			}
			fs.SetScaleGPUPtr(scalePtr)
			uploaded++
			continue
		}
		// Upload BFloat16 raw bytes to GPU and cache the pointer.
		// BF16 weights (2 bytes/element) are used with cublasGemmEx for
		// mixed-precision MatMul (BF16 weights × FP32 activations → FP32 output).
		if bs, ok := any(t.GetStorage()).(*tensor.BFloat16Storage); ok {
			if ptr, _, _ := bs.GPUPtr(); ptr != nil {
				continue // already on GPU
			}
			rawBytes := bs.RawBytes()
			devPtr, err := e.allocWeight(len(rawBytes))
			if err != nil {
				return fmt.Errorf("alloc BF16 GPU (shape %v): %w", t.Shape(), err)
			}
			if err := e.uploadBytes(devPtr, rawBytes); err != nil {
				_ = e.runtime.Free(devPtr)
				return fmt.Errorf("upload BF16 (shape %v): %w", t.Shape(), err)
			}
			bs.SetGPUPtr(devPtr, len(rawBytes), e.deviceID)
			uploaded++
			continue
		}
		// Upload float32 weights to GPU. With Pow, binary ops, and
		// Split/Concat now running on GPU, float32 weights benefit from
		// staying on device (eliminates per-op H2D copies for norm weights).
		if _, ok := t.GetStorage().(*tensor.GPUStorage[float32]); ok {
			continue // already on GPU
		}
		if _, ok := any(t.GetStorage()).(*tensor.Float16Storage); ok {
			continue // already converted to FP16 on GPU
		}
		// Skip Q8 tensors -- they are uploaded as raw bytes in the Q8 loop below.
		if _, ok := any(t.GetStorage()).(*tensor.Q8Storage); ok {
			continue
		}
		data := t.Data()
		n := len(data)
		if n == 0 {
			continue
		}
		byteSize := n * f32Size
		devPtr, err := e.allocWeight(byteSize)
		if err != nil {
			return fmt.Errorf("alloc f32 GPU (shape %v): %w", t.Shape(), err)
		}
		src := unsafe.Slice((*byte)(unsafe.Pointer(&data[0])), byteSize)
		if err := e.uploadBytes(devPtr, src); err != nil {
			_ = e.runtime.Free(devPtr)
			return fmt.Errorf("upload f32 (shape %v): %w", t.Shape(), err)
		}
		// F32 weights (norm gains, embedding table) stay as GPUStorage[float32].
		// Norm weights are tiny (model_dim elements) so FP16 savings are negligible.
		// The per-op FP16 compute paths (fp16FusedAddRMSNorm, etc.) handle F32->FP16
		// conversion on the fly. The Gather output FP16 conversion is the entry point
		// for FP16 activations in the forward pass.
		gs, err := tensor.NewGPUStorageFromPtr[float32](devPtr, n, e.deviceID)
		if err != nil {
			_ = e.runtime.Free(devPtr)
			return fmt.Errorf("create GPU storage (shape %v): %w", t.Shape(), err)
		}
		t.SetStorage(gs)
		uploaded++
	}
	// Upload Q8 quantized weights as raw Q8 bytes (36 bytes per 32 values).
	// This keeps Q8 data compressed on GPU (325 MB vs 1.2 GB as F32) and uses
	// the Q8 dequant-GEMM kernel to dequantize during matmul.
	q8Uploaded := 0
	for _, t := range tensors {
		if t == nil {
			continue
		}
		qs, ok := any(t.GetStorage()).(*tensor.Q8Storage)
		if !ok {
			continue
		}
		if ptr, _, _ := qs.GPUPtr(); ptr != nil {
			continue // already on GPU
		}
		rawBytes := qs.RawBytes()
		devPtr, err := e.allocWeight(len(rawBytes))
		if err != nil {
			return fmt.Errorf("alloc Q8 GPU (shape %v): %w", t.Shape(), err)
		}
		if err := e.uploadBytes(devPtr, rawBytes); err != nil {
			_ = e.runtime.Free(devPtr)
			return fmt.Errorf("upload Q8 (shape %v): %w", t.Shape(), err)
		}
		qs.SetGPUPtr(devPtr, len(rawBytes), e.deviceID)
		q8Uploaded++
	}
	method := "H2D"
	if e.managedMem {
		method = "managed"
	}
	if uploaded > 0 || q4Uploaded > 0 || q8Uploaded > 0 {
		e.logger.Info("weights uploaded to GPU",
			"f32", fmt.Sprintf("%d", uploaded),
			"q4", fmt.Sprintf("%d", q4Uploaded),
			"q8", fmt.Sprintf("%d", q8Uploaded),
			"device", fmt.Sprintf("%d", e.deviceID),
			"method", method)
	}
	return nil
}

// allocWeight allocates permanent memory for a weight tensor.
// Uses cudaMallocManaged on devices with managed memory support,
// otherwise uses cudaMalloc.
func (e *GPUEngine[T]) allocWeight(byteSize int) (unsafe.Pointer, error) {
	if e.managedMem {
		return cuda.MallocManaged(byteSize)
	}
	return e.runtime.Malloc(byteSize)
}

// uploadBytes copies src bytes into a device (or managed) pointer.
// With managed memory, this is a direct CPU memcpy (no H2D needed).
// Without managed memory, this uses cudaMemcpy H2D.
func (e *GPUEngine[T]) uploadBytes(devPtr unsafe.Pointer, src []byte) error {
	if e.managedMem {
		dst := unsafe.Slice((*byte)(devPtr), len(src))
		copy(dst, src)
		return nil
	}
	return e.runtime.Memcpy(devPtr, unsafe.Pointer(&src[0]), len(src), gpuapi.MemcpyHostToDevice)
}

// Stream returns the engine's GPU stream as an unsafe.Pointer (cudaStream_t).
func (e *GPUEngine[T]) Stream() unsafe.Pointer {
	if e.stream == nil {
		return nil
	}
	return e.stream.Ptr()
}

// Close releases the BLAS handle, DNN handle, GPU stream, and drains the memory pool.
// The engine must not be used after Close.
func (e *GPUEngine[T]) Close() error {
	var firstErr error

	// Free FP8 scratch buffers before draining the pool.
	if e.fp8Scratch != nil {
		e.fp8Scratch.free(e.pool, e.deviceID)
		e.fp8Scratch = nil
	}

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

	if e.ltHandle != nil {
		if err := e.ltHandle.Destroy(); err != nil && firstErr == nil {
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
	if a == nil || b == nil {
		return nil, fmt.Errorf("MatMul: input tensors must not be nil")
	}

	// Check for Q4_K quantized storage on A.
	if qs, ok := any(a.GetStorage()).(*tensor.Q4KStorage); ok {
		return e.matMulQ4K(ctx, qs, a, b, dst...)
	}

	// Check for Q4_K quantized storage on B (virtual-transposed weights).
	if qs, ok := any(b.GetStorage()).(*tensor.Q4KStorage); ok {
		return e.matMulQ4KBWeight(ctx, a, qs, b, dst...)
	}

	// Check for Q4 quantized storage on A.
	// Use any() to avoid impossible type assertion when T != float32.
	if qs, ok := any(a.GetStorage()).(*tensor.Q4Storage); ok {
		return e.matMulQ4(ctx, qs, a, b, dst...)
	}

	// Check for Q4 quantized storage on B (virtual-transposed weights).
	if qs, ok := any(b.GetStorage()).(*tensor.Q4Storage); ok {
		return e.matMulQ4BWeight(ctx, a, qs, b, dst...)
	}

	// Check for Q8 quantized storage on A.
	if qs, ok := any(a.GetStorage()).(*tensor.Q8Storage); ok {
		return e.matMulQ8(ctx, qs, a, b, dst...)
	}

	// Check for Q8 quantized storage on B (virtual-transposed weights).
	if qs, ok := any(b.GetStorage()).(*tensor.Q8Storage); ok {
		return e.matMulQ8BWeight(ctx, a, qs, b, dst...)
	}

	// FP16/FP8/BF16 storage checks — skip entirely for F32 compute.
	if e.dtype != DTypeF32 {
		// Check for Float16Storage on A or B — pass FP16 device pointers directly.
		aFP16, aIsFP16 := any(a.GetStorage()).(*tensor.Float16Storage)
		bFP16, bIsFP16 := any(b.GetStorage()).(*tensor.Float16Storage)
		if aIsFP16 || bIsFP16 {
			return fp16MatMulNative(e, ctx, a, b, aFP16, bFP16, aIsFP16, bIsFP16, dst...)
		}

		// Check for FP8 E4M3 storage on A (FP8 weights × FP32 activations).
		if fs, ok := any(a.GetStorage()).(*tensor.FP8E4M3Storage); ok {
			return e.matMulFP8(ctx, fs, a, b, dst...)
		}

		// Check for FP8 E4M3 storage on B (FP8 weights as B operand).
		if fs, ok := any(b.GetStorage()).(*tensor.FP8E4M3Storage); ok {
			return e.matMulFP8BWeight(ctx, a, fs, b, dst...)
		}

		// Check for BFloat16Storage on A (BF16 weights × FP32 activations).
		if bs, ok := any(a.GetStorage()).(*tensor.BFloat16Storage); ok {
			return e.matMulBF16(ctx, bs, a, b, dst...)
		}

		// Check for BFloat16Storage on B (BF16 weights as B operand).
		if bs, ok := any(b.GetStorage()).(*tensor.BFloat16Storage); ok {
			return e.matMulBF16BWeight(ctx, a, bs, b, dst...)
		}
	}

	// FP16 compute path: convert F32 inputs to FP16, run mixed-precision GEMM.
	if (e.dtype == DTypeFP16 || e.dtype == DTypeFP8) && isFloat32[T]() {
		aShape := a.Shape()
		bShape := b.Shape()
		if len(aShape) >= 2 && len(bShape) >= 2 {
			return fp16MatMul(e, ctx, a, b, dst...)
		}
	}

	// float32 and BFloat16 have GPU BLAS paths; fall back for other types.
	var zero T
	_, isFloat32 := any(zero).(float32)
	_, isBFloat16 := any(zero).(float16.BFloat16)
	if !isFloat32 && !isBFloat16 {
		return e.cpu.MatMul(ctx, a, b, dst...)
	}

	if e.blas == nil {
		return e.cpu.MatMul(ctx, a, b, dst...)
	}

	e.setDevice()

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

	// Use strided batched GEMM when available for float32 with batch > 1.
	// This replaces N sequential Sgemm calls with a single cuBLAS call.
	// When bBatchSize == 1, strideB = 0 broadcasts B across all batches
	// (supported by cuBLAS). This is critical for GQA attention where K/V
	// have fewer heads than Q, replacing N individual Sgemm calls with 1.
	if batchSize > 1 && isFloat32 {
		if batched, ok := e.blas.(gpuapi.BLASBatched); ok {
			strideA := int64(aMatSize)
			strideBVal := int64(bMatSize)
			if bBatchSize == 1 {
				strideBVal = 0
			}
			strideC := int64(cMatSize)
			if err := batched.SgemmStridedBatched(m, n, k, 1.0,
				devA, strideA, devB, strideBVal, 0.0,
				devCTotal, strideC, batchSize); err != nil {
				e.pool.Free(e.deviceID, devCTotal, batchSize*cMatSize*elemSize)
				return nil, fmt.Errorf("MatMul: batched GEMM: %w", err)
			}
			return makeGPUResult[T](e, outShape, devCTotal, batchSize*cMatSize, dst...)
		}
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

// MatMulTransposeB computes C = A * B^T using cuBLAS SgemmNT, avoiding
// an explicit Transpose allocation and kernel launch.
// A is [...batch, m, k], B is [...batch, n, k], result is [...batch, m, n].
// Supports batch broadcasting (bBatch=1 broadcasts B across A's batch).
func (e *GPUEngine[T]) MatMulTransposeB(ctx context.Context, a, b *tensor.TensorNumeric[T], dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if a == nil || b == nil {
		return nil, fmt.Errorf("MatMulTransposeB: input tensors must not be nil")
	}

	// Check for SgemmNT support.
	ntBLAS, ok := e.blas.(gpuapi.BLASTransposeB)
	if !ok {
		// Fall back: explicit transpose + standard MatMul.
		kT, err := e.Transpose(ctx, b, []int{0, 2, 1})
		if err != nil {
			return nil, err
		}
		return e.MatMul(ctx, a, kT, dst...)
	}

	var zero T
	_, isFloat32 := any(zero).(float32)
	if !isFloat32 {
		kT, err := e.Transpose(ctx, b, []int{0, 2, 1})
		if err != nil {
			return nil, err
		}
		return e.MatMul(ctx, a, kT, dst...)
	}

	e.setDevice()

	aShape := a.Shape()
	bShape := b.Shape()

	if len(aShape) < 2 || len(bShape) < 2 {
		return nil, fmt.Errorf("MatMulTransposeB: tensors must have at least 2 dimensions")
	}

	// A: [..., m, k], B: [..., n, k] → C: [..., m, n]
	m := aShape[len(aShape)-2]
	k := aShape[len(aShape)-1]
	n := bShape[len(bShape)-2] // B's second-to-last dim is n (B^T would be [k, n])
	bk := bShape[len(bShape)-1]

	if k != bk {
		return nil, fmt.Errorf("MatMulTransposeB: k dimensions must match: A[..., %d] vs B[..., %d]", k, bk)
	}

	aBatchSize := 1
	for _, d := range aShape[:len(aShape)-2] {
		aBatchSize *= d
	}
	bBatchSize := 1
	for _, d := range bShape[:len(bShape)-2] {
		bBatchSize *= d
	}
	if bBatchSize != 1 && aBatchSize != bBatchSize {
		return nil, fmt.Errorf("MatMulTransposeB: batch dimensions %v and %v are incompatible", aShape[:len(aShape)-2], bShape[:len(bShape)-2])
	}

	batchSize := aBatchSize
	if bBatchSize > batchSize {
		batchSize = bBatchSize
	}

	outShape := make([]int, 0, len(aShape))
	outShape = append(outShape, aShape[:len(aShape)-2]...)
	outShape = append(outShape, m, n)

	devA, cleanupA, err := getDevicePtr(e, a)
	if err != nil {
		kT, tErr := e.Transpose(ctx, b, []int{0, 2, 1})
		if tErr != nil {
			return nil, tErr
		}
		return e.MatMul(ctx, a, kT, dst...)
	}
	defer cleanupA()

	devB, cleanupB, err := getDevicePtr(e, b)
	if err != nil {
		kT, tErr := e.Transpose(ctx, b, []int{0, 2, 1})
		if tErr != nil {
			return nil, tErr
		}
		return e.MatMul(ctx, a, kT, dst...)
	}
	defer cleanupB()

	elemSize := int(unsafe.Sizeof(zero))
	aMatSize := m * k
	bMatSize := n * k
	cMatSize := m * n

	devC, err := e.pool.Alloc(e.deviceID, batchSize*cMatSize*elemSize)
	if err != nil {
		kT, tErr := e.Transpose(ctx, b, []int{0, 2, 1})
		if tErr != nil {
			return nil, tErr
		}
		return e.MatMul(ctx, a, kT, dst...)
	}

	// Use strided batched NT GEMM when available for batch > 1.
	// This replaces N sequential SgemmNT calls with a single cuBLAS call.
	// When bBatchSize == 1, strideB = 0 broadcasts B across all batches
	// (supported by cuBLAS). Critical for GQA attention decode performance.
	if batchSize > 1 {
		if batchedNT, ok := e.blas.(gpuapi.BLASBatchedTransposeB); ok {
			strideBVal := int64(bMatSize)
			if bBatchSize == 1 {
				strideBVal = 0
			}
			if err := batchedNT.SgemmNTStridedBatched(m, n, k, 1.0,
				devA, int64(aMatSize), devB, strideBVal, 0.0,
				devC, int64(cMatSize), batchSize); err != nil {
				e.pool.Free(e.deviceID, devC, batchSize*cMatSize*elemSize)
				return nil, fmt.Errorf("MatMulTransposeB: batched NT GEMM: %w", err)
			}
			return makeGPUResult[T](e, outShape, devC, batchSize*cMatSize, dst...)
		}
	}

	for batch := range batchSize {
		aOff := batch * aMatSize * elemSize
		bOff := 0
		if bBatchSize > 1 {
			bOff = batch * bMatSize * elemSize
		}
		cOff := batch * cMatSize * elemSize

		if err := ntBLAS.SgemmNT(m, n, k, 1.0,
			unsafe.Add(devA, aOff),
			unsafe.Add(devB, bOff),
			0.0,
			unsafe.Add(devC, cOff),
		); err != nil {
			e.pool.Free(e.deviceID, devC, batchSize*cMatSize*elemSize)
			return nil, fmt.Errorf("MatMulTransposeB: BLAS batch %d: %w", batch, err)
		}
	}

	return makeGPUResult[T](e, outShape, devC, batchSize*cMatSize, dst...)
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
		bpr := k / 32
		aBytes := qs.RawBytesGPU(bpr)
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

	totalBlocks := m * (k / 32)
	dataOff := tensor.Q4GPUDataOffset(totalBlocks)
	if err := e.kernels.GemmQ4F32(devA, devB, devC, m, k, n, dataOff, e.stream); err != nil {
		e.pool.Free(e.deviceID, devC, cSize)
		return e.cpu.MatMul(ctx, a, b, dst...)
	}

	return makeGPUResult[T](e, []int{m, n}, devC, m*n, dst...)
}

// matMulQ4BWeight handles MatMul where B has Q4 storage (virtual-transposed weight).
// B's shape after virtual transpose is [K, N], but the Q4 data is laid out as [N, K].
// We compute C[M, N] = A[M, K] * dequant(B)^T by reformulating as:
//   C_temp[N, M] = gemm_q4(B_q4[N, K], A^T[K, M])
//
// For GEMV (M=1), A^T[K,1] is just A's data as a column, and C_temp[N,1]
// can be reshaped to [1, N] without a physical transpose.
func (e *GPUEngine[T]) matMulQ4BWeight(ctx context.Context, a *tensor.TensorNumeric[T], qs *tensor.Q4Storage, b *tensor.TensorNumeric[T], dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	aShape := a.Shape()
	bShape := b.Shape()

	if len(aShape) < 2 || len(bShape) < 2 {
		return e.cpu.MatMul(ctx, a, b, dst...)
	}

	// B must be 2D (virtual-transposed weight).
	if len(bShape) > 2 {
		return e.cpu.MatMul(ctx, a, b, dst...)
	}

	// Flatten A's batch dims: [batch..., m, k] -> [m_total, k]
	k := aShape[len(aShape)-1]
	m := 1
	for i := 0; i < len(aShape)-1; i++ {
		m *= aShape[i]
	}
	n := bShape[1] // columns of B (after virtual transpose)

	// Q4 original layout is [N, K]. Verify K is a multiple of 32.
	if k%32 != 0 {
		return e.cpu.MatMul(ctx, a, b, dst...)
	}

	// Build output shape: [batch..., m_last, n] matching standard MatMul broadcast.
	outShape := make([]int, len(aShape))
	copy(outShape, aShape[:len(aShape)-1])
	outShape[len(outShape)-1] = n

	e.setDevice()

	// Get Q4 device pointer (pre-uploaded or upload now).
	var devQ4 unsafe.Pointer
	var freeQ4 func()
	if ptr, _, _ := qs.GPUPtr(); ptr != nil {
		devQ4 = ptr
		freeQ4 = func() {}
	} else {
		bpr := k / 32
		q4Bytes := qs.RawBytesGPU(bpr)
		var err error
		devQ4, err = e.pool.Alloc(e.deviceID, len(q4Bytes))
		if err != nil {
			return e.cpu.MatMul(ctx, a, b, dst...)
		}
		freeQ4 = func() { e.pool.Free(e.deviceID, devQ4, len(q4Bytes)) }
		if err := e.runtime.Memcpy(devQ4, unsafe.Pointer(&q4Bytes[0]), len(q4Bytes), gpuapi.MemcpyHostToDevice); err != nil {
			freeQ4()
			return e.cpu.MatMul(ctx, a, b, dst...)
		}
	}
	defer freeQ4()

	// Upload A to GPU as F32. A's data is contiguous [m, k] regardless of
	// original batch shape, so the kernel sees it as a flat 2D matrix.
	devA, cleanupA, err := getDevicePtr(e, a)
	if err != nil {
		return e.cpu.MatMul(ctx, a, b, dst...)
	}
	defer cleanupA()

	f32Size := int(unsafe.Sizeof(float32(0)))

	if m == 1 {
		// GEMV fast path: C_temp[N, 1] = gemm_q4(B_q4[N,K], A^T[K,1])
		// A is [1, K], A^T is [K, 1] -- same data, just different shape.
		cSize := n * f32Size
		devC, err := e.pool.Alloc(e.deviceID, cSize)
		if err != nil {
			return e.cpu.MatMul(ctx, a, b, dst...)
		}

		q4DataOff := tensor.Q4GPUDataOffset(qs.NumBlocks())
		if err := e.kernels.GemmQ4F32(devQ4, devA, devC, n, k, 1, q4DataOff, e.stream); err != nil {
			e.pool.Free(e.deviceID, devC, cSize)
			return e.cpu.MatMul(ctx, a, b, dst...)
		}

		return makeGPUResult[T](e, outShape, devC, n, dst...)
	}

	// General GEMM: C_temp[N, M] = gemm_q4(B_q4[N,K], A^T[K,M])
	// Transpose flattened A[M, K] -> A^T[K, M] on GPU.
	aFlat, err := e.Reshape(ctx, a, []int{m, k})
	if err != nil {
		return e.cpu.MatMul(ctx, a, b, dst...)
	}
	aT, err := e.Transpose(ctx, aFlat, []int{1, 0})
	if err != nil {
		return e.cpu.MatMul(ctx, a, b, dst...)
	}
	devAT, cleanupAT, err := getDevicePtr(e, aT)
	if err != nil {
		return e.cpu.MatMul(ctx, a, b, dst...)
	}
	defer cleanupAT()

	cTempSize := n * m * f32Size
	devCTemp, err := e.pool.Alloc(e.deviceID, cTempSize)
	if err != nil {
		return e.cpu.MatMul(ctx, a, b, dst...)
	}

	q4DataOff2 := tensor.Q4GPUDataOffset(qs.NumBlocks())
	if err := e.kernels.GemmQ4F32(devQ4, devAT, devCTemp, n, k, m, q4DataOff2, e.stream); err != nil {
		e.pool.Free(e.deviceID, devCTemp, cTempSize)
		return e.cpu.MatMul(ctx, a, b, dst...)
	}

	// C_temp is [N, M], transpose to [M, N], then reshape to outShape.
	cTempTensor, err := makeGPUResult[T](e, []int{n, m}, devCTemp, n*m)
	if err != nil {
		e.pool.Free(e.deviceID, devCTemp, cTempSize)
		return nil, err
	}

	cFlat, err := e.Transpose(ctx, cTempTensor, []int{1, 0})
	if err != nil {
		return nil, err
	}
	return e.Reshape(ctx, cFlat, outShape, dst...)
}

// matMulQ4K handles GPU Q4_K dequant-GEMM when Q4_K storage is on A.
// For GEMV (n==1, single-column B), uses fused dequant+GEMV kernel.
// For general GEMM (n>1), dequantizes Q4_K to F32 on GPU then calls cuBLAS Sgemm.
func (e *GPUEngine[T]) matMulQ4K(ctx context.Context, qs *tensor.Q4KStorage, a, b *tensor.TensorNumeric[T], dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	aShape := a.Shape()
	bShape := b.Shape()

	if len(aShape) < 2 || len(bShape) < 2 {
		return e.cpu.MatMul(ctx, a, b, dst...)
	}

	// Only handle unbatched 2D for now.
	if len(aShape) > 2 || len(bShape) > 2 {
		return e.cpu.MatMul(ctx, a, b, dst...)
	}

	m := aShape[0]
	k := aShape[1]
	n := bShape[1]

	// K must be a multiple of 256 for Q4_K super-blocks.
	if k%256 != 0 {
		return e.cpu.MatMul(ctx, a, b, dst...)
	}

	e.setDevice()

	// Get Q4_K device pointer (pre-uploaded or upload now).
	var devW unsafe.Pointer
	var freeW func()
	if ptr, _, _ := qs.GPUPtr(); ptr != nil {
		devW = ptr
		freeW = func() {}
	} else {
		rawBytes := qs.RawBytes()
		var err error
		devW, err = e.pool.Alloc(e.deviceID, len(rawBytes))
		if err != nil {
			return e.cpu.MatMul(ctx, a, b, dst...)
		}
		freeW = func() { e.pool.Free(e.deviceID, devW, len(rawBytes)) }
		if err := e.runtime.Memcpy(devW, unsafe.Pointer(&rawBytes[0]), len(rawBytes), gpuapi.MemcpyHostToDevice); err != nil {
			freeW()
			return e.cpu.MatMul(ctx, a, b, dst...)
		}
	}
	defer freeW()

	// Fused GEMV path: y = dequant(W_q4k) * x, when n==1.
	if n == 1 {
		devX, cleanupX, err := getDevicePtr(e, b)
		if err != nil {
			return e.cpu.MatMul(ctx, a, b, dst...)
		}
		defer cleanupX()

		f32Size := int(unsafe.Sizeof(float32(0)))
		cSize := m * f32Size
		devY, err := e.pool.Alloc(e.deviceID, cSize)
		if err != nil {
			return e.cpu.MatMul(ctx, a, b, dst...)
		}

		if err := e.kernels.GemvQ4KF32(devW, devX, devY, m, k, e.stream); err != nil {
			e.pool.Free(e.deviceID, devY, cSize)
			return e.cpu.MatMul(ctx, a, b, dst...)
		}

		return makeGPUResult[T](e, []int{m, n}, devY, m*n, dst...)
	}

	// General GEMM: dequantize Q4_K to F32 on GPU, then cuBLAS Sgemm.
	// C[M,N] = dequant(A_q4k)[M,K] * B[K,N]
	f32Size := int(unsafe.Sizeof(float32(0)))

	// Dequantize A to F32.
	dequantSize := m * k * f32Size
	devAF32, err := e.pool.Alloc(e.deviceID, dequantSize)
	if err != nil {
		return e.cpu.MatMul(ctx, a, b, dst...)
	}
	defer e.pool.Free(e.deviceID, devAF32, dequantSize)

	if err := e.kernels.DequantQ4KF32(devW, devAF32, m, k, e.stream); err != nil {
		return e.cpu.MatMul(ctx, a, b, dst...)
	}

	// Upload B to GPU.
	devB, cleanupB, err := getDevicePtr(e, b)
	if err != nil {
		return e.cpu.MatMul(ctx, a, b, dst...)
	}
	defer cleanupB()

	// Allocate output C.
	cSize := m * n * f32Size
	devC, err := e.pool.Alloc(e.deviceID, cSize)
	if err != nil {
		return e.cpu.MatMul(ctx, a, b, dst...)
	}

	if err := e.blas.Sgemm(m, n, k, 1.0, devAF32, devB, 0.0, devC); err != nil {
		e.pool.Free(e.deviceID, devC, cSize)
		return nil, fmt.Errorf("matMulQ4K: Sgemm: %w", err)
	}

	return makeGPUResult[T](e, []int{m, n}, devC, m*n, dst...)
}

// matMulQ4KBWeight handles MatMul where B has Q4_K storage (virtual-transposed weight).
// B's shape after virtual transpose is [K, N], but the Q4_K data is laid out as [N, K].
// For GEMV (m==1, single-token decode), uses fused dequant+GEMV kernel directly
// on the Q4_K weight data, halving memory bandwidth vs separate dequant + GEMM.
// For general GEMM (m>1), dequantizes Q4_K to F32 on GPU then calls cuBLAS SgemmNT.
func (e *GPUEngine[T]) matMulQ4KBWeight(ctx context.Context, a *tensor.TensorNumeric[T], qs *tensor.Q4KStorage, b *tensor.TensorNumeric[T], dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	aShape := a.Shape()
	bShape := b.Shape()

	if len(aShape) < 2 || len(bShape) < 2 {
		return e.cpu.MatMul(ctx, a, b, dst...)
	}

	// B must be 2D (virtual-transposed weight).
	if len(bShape) > 2 {
		return e.cpu.MatMul(ctx, a, b, dst...)
	}

	// Flatten A's batch dims: [batch..., m, k] -> [m_total, k]
	k := aShape[len(aShape)-1]
	m := 1
	for i := 0; i < len(aShape)-1; i++ {
		m *= aShape[i]
	}
	n := bShape[1] // columns of B (after virtual transpose)

	// K must be a multiple of 256 for Q4_K super-blocks.
	if k%256 != 0 {
		return e.cpu.MatMul(ctx, a, b, dst...)
	}

	// Build output shape: [batch..., m_last, n].
	outShape := make([]int, len(aShape))
	copy(outShape, aShape[:len(aShape)-1])
	outShape[len(outShape)-1] = n

	e.setDevice()

	// Get Q4_K device pointer (pre-uploaded or upload now).
	// Q4_K data is stored as [N, K] super-blocks.
	var devQ4K unsafe.Pointer
	var freeQ4K func()
	if ptr, _, _ := qs.GPUPtr(); ptr != nil {
		devQ4K = ptr
		freeQ4K = func() {}
	} else {
		rawBytes := qs.RawBytes()
		var err error
		devQ4K, err = e.pool.Alloc(e.deviceID, len(rawBytes))
		if err != nil {
			return e.cpu.MatMul(ctx, a, b, dst...)
		}
		freeQ4K = func() { e.pool.Free(e.deviceID, devQ4K, len(rawBytes)) }
		if err := e.runtime.Memcpy(devQ4K, unsafe.Pointer(&rawBytes[0]), len(rawBytes), gpuapi.MemcpyHostToDevice); err != nil {
			freeQ4K()
			return e.cpu.MatMul(ctx, a, b, dst...)
		}
	}
	defer freeQ4K()

	// Fused GEMV path: y[n] = sum_k dequant(B_q4k[n, k]) * x[k], when m==1.
	if m == 1 {
		devX, cleanupX, err := getDevicePtr(e, a)
		if err != nil {
			return e.cpu.MatMul(ctx, a, b, dst...)
		}
		defer cleanupX()

		f32Size := int(unsafe.Sizeof(float32(0)))
		cSize := n * f32Size
		devY, err := e.pool.Alloc(e.deviceID, cSize)
		if err != nil {
			return e.cpu.MatMul(ctx, a, b, dst...)
		}

		if err := e.kernels.GemvQ4KF32(devQ4K, devX, devY, n, k, e.stream); err != nil {
			e.pool.Free(e.deviceID, devY, cSize)
			return e.cpu.MatMul(ctx, a, b, dst...)
		}

		return makeGPUResult[T](e, outShape, devY, n, dst...)
	}

	// General GEMM: dequantize Q4_K to F32 on GPU, then cuBLAS.
	// Q4_K data is [N, K]. Dequantize gives F32 [N, K].
	// We need C[M,N] = A[M,K] * B^T where B = dequant(B_q4k)[N,K].
	// Use SgemmNT: C = A * B^T (A is [M,K], B is [N,K], C is [M,N]).
	f32Size := int(unsafe.Sizeof(float32(0)))

	// Dequantize B to F32 [N, K].
	dequantSize := n * k * f32Size
	devBF32, err := e.pool.Alloc(e.deviceID, dequantSize)
	if err != nil {
		return e.cpu.MatMul(ctx, a, b, dst...)
	}
	defer e.pool.Free(e.deviceID, devBF32, dequantSize)

	if err := e.kernels.DequantQ4KF32(devQ4K, devBF32, n, k, e.stream); err != nil {
		return e.cpu.MatMul(ctx, a, b, dst...)
	}

	// Upload A to GPU.
	devA, cleanupA, err := getDevicePtr(e, a)
	if err != nil {
		return e.cpu.MatMul(ctx, a, b, dst...)
	}
	defer cleanupA()

	// Allocate output C [M, N].
	cSize := m * n * f32Size
	devC, err := e.pool.Alloc(e.deviceID, cSize)
	if err != nil {
		return e.cpu.MatMul(ctx, a, b, dst...)
	}

	// Use SgemmNT if available (avoids explicit transpose).
	if ntBLAS, ok := e.blas.(gpuapi.BLASTransposeB); ok {
		if err := ntBLAS.SgemmNT(m, n, k, 1.0, devA, devBF32, 0.0, devC); err != nil {
			e.pool.Free(e.deviceID, devC, cSize)
			return nil, fmt.Errorf("matMulQ4KBWeight: SgemmNT: %w", err)
		}
		return makeGPUResult[T](e, outShape, devC, m*n, dst...)
	}

	// Fallback: transpose dequantized B then use Sgemm.
	devBT, err := e.pool.Alloc(e.deviceID, dequantSize)
	if err != nil {
		e.pool.Free(e.deviceID, devC, cSize)
		return e.cpu.MatMul(ctx, a, b, dst...)
	}
	defer e.pool.Free(e.deviceID, devBT, dequantSize)

	if err := e.kernels.Transpose2D(devBF32, devBT, n, k, e.stream); err != nil {
		e.pool.Free(e.deviceID, devC, cSize)
		return e.cpu.MatMul(ctx, a, b, dst...)
	}

	if err := e.blas.Sgemm(m, n, k, 1.0, devA, devBT, 0.0, devC); err != nil {
		e.pool.Free(e.deviceID, devC, cSize)
		return nil, fmt.Errorf("matMulQ4KBWeight: Sgemm: %w", err)
	}

	return makeGPUResult[T](e, outShape, devC, m*n, dst...)
}

// matMulQ8 handles GPU Q8_0 dequant-GEMM: C = dequant(A_q8) * B.
// Only supports unbatched 2D for now; batched Q8 falls back to CPU.
func (e *GPUEngine[T]) matMulQ8(ctx context.Context, qs *tensor.Q8Storage, a, b *tensor.TensorNumeric[T], dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
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

	// Use pre-uploaded Q8 GPU pointer if available; otherwise upload now.
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

	if err := e.kernels.GemmQ8F32(devA, devB, devC, m, k, n, e.stream); err != nil {
		e.pool.Free(e.deviceID, devC, cSize)
		return e.cpu.MatMul(ctx, a, b, dst...)
	}

	return makeGPUResult[T](e, []int{m, n}, devC, m*n, dst...)
}

// matMulQ8BWeight handles MatMul where B has Q8 storage (virtual-transposed weight).
// B's shape after virtual transpose is [K, N], but the Q8 data is laid out as [N, K].
// We compute C[M, N] = A[M, K] * dequant(B)^T by reformulating as:
//
//	C_temp[N, M] = gemm_q8(B_q8[N, K], A^T[K, M])
//
// For GEMV (M=1), A^T[K,1] is just A's data as a column, and C_temp[N,1]
// can be reshaped to [1, N] without a physical transpose.
func (e *GPUEngine[T]) matMulQ8BWeight(ctx context.Context, a *tensor.TensorNumeric[T], qs *tensor.Q8Storage, b *tensor.TensorNumeric[T], dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	aShape := a.Shape()
	bShape := b.Shape()

	if len(aShape) < 2 || len(bShape) < 2 {
		return e.cpu.MatMul(ctx, a, b, dst...)
	}

	// B must be 2D (virtual-transposed weight).
	if len(bShape) > 2 {
		return e.cpu.MatMul(ctx, a, b, dst...)
	}

	// Flatten A's batch dims: [batch..., m, k] -> [m_total, k]
	k := aShape[len(aShape)-1]
	m := 1
	for i := 0; i < len(aShape)-1; i++ {
		m *= aShape[i]
	}
	n := bShape[1] // columns of B (after virtual transpose)

	// Q8 original layout is [N, K]. Verify K is a multiple of 32.
	if k%32 != 0 {
		return e.cpu.MatMul(ctx, a, b, dst...)
	}

	// Build output shape: [batch..., m_last, n] matching standard MatMul broadcast.
	outShape := make([]int, len(aShape))
	copy(outShape, aShape[:len(aShape)-1])
	outShape[len(outShape)-1] = n

	e.setDevice()

	// Get Q8 device pointer (pre-uploaded or upload now).
	var devQ8 unsafe.Pointer
	var freeQ8 func()
	if ptr, _, _ := qs.GPUPtr(); ptr != nil {
		devQ8 = ptr
		freeQ8 = func() {}
	} else {
		q8Bytes := qs.RawBytes()
		var err error
		devQ8, err = e.pool.Alloc(e.deviceID, len(q8Bytes))
		if err != nil {
			return e.cpu.MatMul(ctx, a, b, dst...)
		}
		freeQ8 = func() { e.pool.Free(e.deviceID, devQ8, len(q8Bytes)) }
		if err := e.runtime.Memcpy(devQ8, unsafe.Pointer(&q8Bytes[0]), len(q8Bytes), gpuapi.MemcpyHostToDevice); err != nil {
			freeQ8()
			return e.cpu.MatMul(ctx, a, b, dst...)
		}
	}
	defer freeQ8()

	// Upload A to GPU as F32.
	devA, cleanupA, err := getDevicePtr(e, a)
	if err != nil {
		return e.cpu.MatMul(ctx, a, b, dst...)
	}
	defer cleanupA()

	f32Size := int(unsafe.Sizeof(float32(0)))

	if m == 1 {
		// GEMV fast path: C_temp[N, 1] = gemm_q8(B_q8[N,K], A^T[K,1])
		cSize := n * f32Size
		devC, err := e.pool.Alloc(e.deviceID, cSize)
		if err != nil {
			return e.cpu.MatMul(ctx, a, b, dst...)
		}

		if err := e.kernels.GemmQ8F32(devQ8, devA, devC, n, k, 1, e.stream); err != nil {
			e.pool.Free(e.deviceID, devC, cSize)
			return e.cpu.MatMul(ctx, a, b, dst...)
		}

		return makeGPUResult[T](e, outShape, devC, n, dst...)
	}

	// General GEMM: C_temp[N, M] = gemm_q8(B_q8[N,K], A^T[K,M])
	aFlat, err := e.Reshape(ctx, a, []int{m, k})
	if err != nil {
		return e.cpu.MatMul(ctx, a, b, dst...)
	}
	aT, err := e.Transpose(ctx, aFlat, []int{1, 0})
	if err != nil {
		return e.cpu.MatMul(ctx, a, b, dst...)
	}
	devAT, cleanupAT, err := getDevicePtr(e, aT)
	if err != nil {
		return e.cpu.MatMul(ctx, a, b, dst...)
	}
	defer cleanupAT()

	cTempSize := n * m * f32Size
	devCTemp, err := e.pool.Alloc(e.deviceID, cTempSize)
	if err != nil {
		return e.cpu.MatMul(ctx, a, b, dst...)
	}

	if err := e.kernels.GemmQ8F32(devQ8, devAT, devCTemp, n, k, m, e.stream); err != nil {
		e.pool.Free(e.deviceID, devCTemp, cTempSize)
		return e.cpu.MatMul(ctx, a, b, dst...)
	}

	// C_temp is [N, M], transpose to [M, N], then reshape to outShape.
	cTempTensor, err := makeGPUResult[T](e, []int{n, m}, devCTemp, n*m)
	if err != nil {
		e.pool.Free(e.deviceID, devCTemp, cTempSize)
		return nil, err
	}

	cFlat, err := e.Transpose(ctx, cTempTensor, []int{1, 0})
	if err != nil {
		return nil, err
	}
	return e.Reshape(ctx, cFlat, outShape, dst...)
}

// matMulBF16 handles MatMul where A has BFloat16Storage.
// A is [M, K] in BF16, B is [K, N] in FP32 → C is [M, N] in FP32.
// B's FP32 data is converted to BF16 on the fly, then MixedBF16Gemm
// computes with BF16 inputs and FP32 output via cublasGemmEx.
func (e *GPUEngine[T]) matMulBF16(ctx context.Context, bs *tensor.BFloat16Storage, a, b *tensor.TensorNumeric[T], dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	aShape := a.Shape()
	bShape := b.Shape()

	if len(aShape) < 2 || len(bShape) < 2 {
		return e.cpu.MatMul(ctx, a, b, dst...)
	}
	if len(aShape) > 2 || len(bShape) > 2 {
		return e.cpu.MatMul(ctx, a, b, dst...)
	}

	if e.blas == nil {
		return e.cpu.MatMul(ctx, a, b, dst...)
	}

	m := aShape[0]
	k := aShape[1]
	n := bShape[1]

	e.setDevice()

	// Get BF16 device pointer for A (pre-uploaded or upload now).
	var devA unsafe.Pointer
	var freeA func()
	if ptr, _, _ := bs.GPUPtr(); ptr != nil {
		devA = ptr
		freeA = func() {}
	} else {
		aBytes := bs.RawBytes()
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

	// Convert B from FP32 to BF16 and upload.
	// BFloat16Storage is Storage[float32], so T is float32 here.
	bData := b.Data()
	bF32 := *(*[]float32)(unsafe.Pointer(&bData))
	bBF16 := tensor.NewBFloat16Storage(bF32)
	bBytes := bBF16.RawBytes()
	devB, err := e.pool.Alloc(e.deviceID, len(bBytes))
	if err != nil {
		return e.cpu.MatMul(ctx, a, b, dst...)
	}
	defer e.pool.Free(e.deviceID, devB, len(bBytes))

	if err := e.runtime.Memcpy(devB, unsafe.Pointer(&bBytes[0]), len(bBytes), gpuapi.MemcpyHostToDevice); err != nil {
		return e.cpu.MatMul(ctx, a, b, dst...)
	}

	// Allocate FP32 output.
	cSize := m * n * f32Size
	devC, err := e.pool.Alloc(e.deviceID, cSize)
	if err != nil {
		return e.cpu.MatMul(ctx, a, b, dst...)
	}

	if err := e.blas.MixedBF16Gemm(m, n, k, 1.0, devA, devB, 0.0, devC); err != nil {
		e.pool.Free(e.deviceID, devC, cSize)
		return e.cpu.MatMul(ctx, a, b, dst...)
	}

	return makeGPUResult[T](e, []int{m, n}, devC, m*n, dst...)
}

// matMulBF16BWeight handles MatMul where B has BFloat16Storage.
// A is [M, K] in FP32, B is [K, N] in BF16 → C is [M, N] in FP32.
// A's FP32 data is converted to BF16 on the fly, then MixedBF16Gemm
// computes with BF16 inputs and FP32 output via cublasGemmEx.
func (e *GPUEngine[T]) matMulBF16BWeight(ctx context.Context, a *tensor.TensorNumeric[T], bs *tensor.BFloat16Storage, b *tensor.TensorNumeric[T], dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	aShape := a.Shape()
	bShape := b.Shape()

	if len(aShape) < 2 || len(bShape) < 2 {
		return e.cpu.MatMul(ctx, a, b, dst...)
	}
	if len(bShape) > 2 {
		return e.cpu.MatMul(ctx, a, b, dst...)
	}

	if e.blas == nil {
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

	e.setDevice()

	// Convert A from FP32 to BF16 and upload.
	// BFloat16Storage is Storage[float32], so T is float32 here.
	aData := a.Data()
	aF32 := *(*[]float32)(unsafe.Pointer(&aData))
	aBF16 := tensor.NewBFloat16Storage(aF32)
	aBytes := aBF16.RawBytes()
	devA, err := e.pool.Alloc(e.deviceID, len(aBytes))
	if err != nil {
		return e.cpu.MatMul(ctx, a, b, dst...)
	}
	defer e.pool.Free(e.deviceID, devA, len(aBytes))

	if err := e.runtime.Memcpy(devA, unsafe.Pointer(&aBytes[0]), len(aBytes), gpuapi.MemcpyHostToDevice); err != nil {
		return e.cpu.MatMul(ctx, a, b, dst...)
	}

	// Get BF16 device pointer for B (pre-uploaded or upload now).
	var devB unsafe.Pointer
	var freeB func()
	if ptr, _, _ := bs.GPUPtr(); ptr != nil {
		devB = ptr
		freeB = func() {}
	} else {
		bBytes := bs.RawBytes()
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

	// Allocate FP32 output.
	cSize := m * n * f32Size
	devC, err := e.pool.Alloc(e.deviceID, cSize)
	if err != nil {
		return e.cpu.MatMul(ctx, a, b, dst...)
	}

	if err := e.blas.MixedBF16Gemm(m, n, k, 1.0, devA, devB, 0.0, devC); err != nil {
		e.pool.Free(e.deviceID, devC, cSize)
		return e.cpu.MatMul(ctx, a, b, dst...)
	}

	return makeGPUResult[T](e, outShape, devC, m*n, dst...)
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
	_, isGPU := a.GetStorage().(*tensor.GPUStorage[T])
	isFP16 := false
	if e.dtype != DTypeF32 {
		_, isFP16 = any(a.GetStorage()).(*tensor.Float16Storage)
	}
	if !isGPU && !isFP16 {
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

	// GPU transpose kernel supports up to 4D; fall back to CPU for higher ranks.
	if rank > 4 {
		return e.cpu.Transpose(ctx, a, axes, dst...)
	}

	// Compute output shape.
	outShape := make([]int, rank)
	for i, ax := range axes {
		outShape[i] = shape[ax]
	}

	// Fast path: if the transpose only swaps unit-sized dimensions, it is
	// equivalent to a reshape (no data movement). This is common during
	// single-token generation where seqLen=1. Check by comparing the
	// non-unit dimensions in input vs output order.
	if isTransposeReshape(shape, outShape) {
		if e.dtype != DTypeF32 {
			if fs, ok := any(a.GetStorage()).(*tensor.Float16Storage); ok {
				storageT := any(fs).(tensor.Storage[T])
				t, tErr := tensor.NewWithStorage[T](outShape, storageT)
				if tErr != nil {
					return nil, tErr
				}
				return t, nil
			}
		}
		gs := a.GetStorage().(*tensor.GPUStorage[T])
		viewGS := gs.View(gs.Len())
		t, tErr := tensor.NewWithStorage[T](outShape, viewGS)
		if tErr != nil {
			return nil, tErr
		}
		if len(dst) > 0 && dst[0] != nil {
			dst[0].SetStorage(viewGS)
			dst[0].SetShape(outShape)
			return dst[0], nil
		}
		return t, nil
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
	// Precompute output strides on the host so the kernel avoids O(ndim^2) per thread.
	outStrides := make([]int, rank)
	outStride := 1
	for i := rank - 1; i >= 0; i-- {
		outStrides[i] = outStride
		outStride *= outShape[i]
	}

	inStrides32 := make([]int32, rank)
	outStrides32 := make([]int32, rank)
	perm32 := make([]int32, rank)
	for i := range rank {
		inStrides32[i] = int32(inStrides[i])
		outStrides32[i] = int32(outStrides[i])
		perm32[i] = int32(axes[i])
	}

	if err := e.kernels.TransposeND(devIn, devOut, inStrides32, outStrides32, perm32, rank, total, e.stream); err != nil {
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

	// Check whether params are GPU-resident (F32 or FP16 storage).
	_, isGPU := params.GetStorage().(*tensor.GPUStorage[T])
	var fp16Stor *tensor.Float16Storage
	isFP16 := false
	if e.dtype != DTypeF32 {
		fp16Stor, isFP16 = any(params.GetStorage()).(*tensor.Float16Storage)
	}
	if !isGPU && !isFP16 {
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

	// Get device pointer for params. For Float16Storage, convert FP16->F32
	// into a temporary buffer so the F32 Gather kernel can operate on it.
	var devParams unsafe.Pointer
	var cleanupParams func()
	if isFP16 {
		fp16Ptr, _, _ := fp16Stor.GPUPtr()
		if fp16Ptr == nil {
			return e.cpu.Gather(ctx, params, indices, output)
		}
		nElems := V * D
		f32Bytes := nElems * f32Size
		f32Ptr, err := e.pool.Alloc(e.deviceID, f32Bytes)
		if err != nil {
			return e.cpu.Gather(ctx, params, indices, output)
		}
		if err := e.kernels.FP16ToF32(fp16Ptr, f32Ptr, nElems, e.stream); err != nil {
			e.pool.Free(e.deviceID, f32Ptr, f32Bytes)
			return e.cpu.Gather(ctx, params, indices, output)
		}
		devParams = f32Ptr
		cleanupParams = func() { e.pool.Free(e.deviceID, f32Ptr, f32Bytes) }
	} else {
		var err error
		devParams, cleanupParams, err = getDevicePtr(e, params)
		if err != nil {
			return e.cpu.Gather(ctx, params, indices, output)
		}
	}
	defer cleanupParams()

	// Upload indices to GPU as int64 (Go int on 64-bit platforms).
	// The gather kernel accepts int64 indices directly, avoiding the
	// CPU-side int64→int32 conversion that would trigger a D2H copy
	// for GPU-resident indices and block CUDA graph capture.
	intSize := int(unsafe.Sizeof(int(0)))
	idxByteSize := N * intSize
	devIdx, err := e.pool.Alloc(e.deviceID, idxByteSize)
	if err != nil {
		return e.cpu.Gather(ctx, params, indices, output)
	}
	defer e.pool.Free(e.deviceID, devIdx, idxByteSize)

	if err := e.runtime.Memcpy(devIdx, unsafe.Pointer(&idxData[0]), idxByteSize, gpuapi.MemcpyHostToDevice); err != nil {
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

	// When dtype is FP16, convert the F32 gather output to FP16 on GPU.
	// This is the single F32->FP16 conversion point for the entire forward pass;
	// all downstream ops receive Float16Storage and operate in FP16 natively.
	if e.dtype == DTypeFP16 {
		outElems := N * D
		fp16Bytes := outElems * fp16Size
		fp16Ptr, err := e.pool.Alloc(e.deviceID, fp16Bytes)
		if err != nil {
			e.pool.Free(e.deviceID, devOut, outByteSize)
			return fmt.Errorf("Gather FP16 alloc: %w", err)
		}
		if err := e.kernels.F32ToFP16(devOut, fp16Ptr, outElems, e.stream); err != nil {
			e.pool.Free(e.deviceID, fp16Ptr, fp16Bytes)
			e.pool.Free(e.deviceID, devOut, outByteSize)
			return fmt.Errorf("Gather F32->FP16: %w", err)
		}
		e.pool.Free(e.deviceID, devOut, outByteSize)
		fs := any(tensor.NewFloat16StorageGPU(fp16Ptr, outElems, e.deviceID)).(tensor.Storage[T])
		output.SetStorage(fs)
		return nil
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
	if gs, ok := a.GetStorage().(*tensor.GPUStorage[T]); ok {
		return e.gpuSplit(gs.Ptr(), a.Shape(), numSplits, axis)
	}
	if e.dtype != DTypeF32 {
		if fs, ok := any(a.GetStorage()).(*tensor.Float16Storage); ok {
			ptr, _, _ := fs.GPUPtr()
			if ptr != nil {
				return e.gpuSplitFP16(ptr, a.Shape(), numSplits, axis)
			}
		}
	}
	return e.cpu.Split(ctx, a, numSplits, axis)
}

func (e *GPUEngine[T]) Concat(ctx context.Context, tensors []*tensor.TensorNumeric[T], axis int, dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if !isFloat32[T]() || len(tensors) == 0 {
		return e.cpu.Concat(ctx, tensors, axis, dst...)
	}
	// Check all inputs are GPU-resident (GPUStorage or Float16Storage).
	ptrs := make([]unsafe.Pointer, len(tensors))
	allFP16 := true
	for i, t := range tensors {
		if gs, ok := t.GetStorage().(*tensor.GPUStorage[T]); ok {
			ptrs[i] = gs.Ptr()
			allFP16 = false
		} else if e.dtype != DTypeF32 {
			if fs, ok := any(t.GetStorage()).(*tensor.Float16Storage); ok {
				p, _, _ := fs.GPUPtr()
				if p == nil {
					return e.cpu.Concat(ctx, tensors, axis, dst...)
				}
				ptrs[i] = p
			} else {
				return e.cpu.Concat(ctx, tensors, axis, dst...)
			}
		} else {
			return e.cpu.Concat(ctx, tensors, axis, dst...)
		}
	}
	if allFP16 && e.dtype != DTypeF32 {
		return e.gpuConcatFP16(ptrs, tensors, axis, dst...)
	}
	return e.gpuConcat(ptrs, tensors, axis, dst...)
}

func (e *GPUEngine[T]) Repeat(ctx context.Context, a *tensor.TensorNumeric[T], axis int, repetitions int, dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if !isFloat32[T]() || a == nil {
		return e.cpu.Repeat(ctx, a, axis, repetitions, dst...)
	}

	shape := a.Shape()
	if axis < 0 || axis >= len(shape) {
		return e.cpu.Repeat(ctx, a, axis, repetitions, dst...)
	}
	if repetitions <= 0 {
		return e.cpu.Repeat(ctx, a, axis, repetitions, dst...)
	}

	// Get device pointer (handles GPUStorage[T] and Float16Storage).
	isFP16 := false
	if e.dtype != DTypeF32 {
		_, isFP16 = any(a.GetStorage()).(*tensor.Float16Storage)
	}
	gs, isGPU := a.GetStorage().(*tensor.GPUStorage[T])
	if !isGPU && !isFP16 {
		return e.cpu.Repeat(ctx, a, axis, repetitions, dst...)
	}

	e.setDevice()

	newShape := make([]int, len(shape))
	copy(newShape, shape)
	newShape[axis] *= repetitions

	outElems := 1
	for _, d := range newShape {
		outElems *= d
	}
	outBytes := outElems * f32Size

	devOut, err := e.pool.Alloc(e.deviceID, outBytes)
	if err != nil {
		return e.cpu.Repeat(ctx, a, axis, repetitions, dst...)
	}

	var devA unsafe.Pointer
	var cleanupA func()
	if isGPU {
		devA = gs.Ptr()
		cleanupA = func() {}
	} else {
		// Float16Storage: convert FP16→F32 for the F32 repeat kernel.
		f32Engine, ok := any(e).(*GPUEngine[float32])
		if !ok {
			e.pool.Free(e.deviceID, devOut, outBytes)
			return e.cpu.Repeat(ctx, a, axis, repetitions, dst...)
		}
		devA, cleanupA, err = getDevicePtr(f32Engine, any(a).(*tensor.TensorNumeric[float32]))
		if err != nil {
			e.pool.Free(e.deviceID, devOut, outBytes)
			return e.cpu.Repeat(ctx, a, axis, repetitions, dst...)
		}
	}
	defer cleanupA()

	// Compute dimensions for the repeat kernel.
	outerSize := 1
	for i := 0; i < axis; i++ {
		outerSize *= shape[i]
	}
	axisDim := shape[axis]
	innerSize := 1
	for i := axis + 1; i < len(shape); i++ {
		innerSize *= shape[i]
	}

	if err := e.kernels.Repeat(devA, devOut, outerSize, axisDim, innerSize, repetitions, e.stream); err != nil {
		e.pool.Free(e.deviceID, devOut, outBytes)
		return e.cpu.Repeat(ctx, a, axis, repetitions, dst...)
	}

	// For FP16 inputs, convert the F32 output back to Float16Storage.
	if isFP16 {
		fp16Bytes := outElems * fp16Size
		fp16Out, allocErr := e.pool.Alloc(e.deviceID, fp16Bytes)
		if allocErr != nil {
			e.pool.Free(e.deviceID, devOut, outBytes)
			return e.cpu.Repeat(ctx, a, axis, repetitions, dst...)
		}
		if convErr := e.kernels.F32ToFP16(devOut, fp16Out, outElems, e.stream); convErr != nil {
			e.pool.Free(e.deviceID, devOut, outBytes)
			e.pool.Free(e.deviceID, fp16Out, fp16Bytes)
			return e.cpu.Repeat(ctx, a, axis, repetitions, dst...)
		}
		e.pool.Free(e.deviceID, devOut, outBytes)
		fs := tensor.NewFloat16StorageGPU(fp16Out, outElems, e.deviceID)
		storageT := any(fs).(tensor.Storage[T])
		return tensor.NewWithStorage[T](newShape, storageT)
	}

	return makeGPUResult[T](e, newShape, devOut, outElems, dst...)
}

func (e *GPUEngine[T]) OneHot(ctx context.Context, input *tensor.TensorNumeric[int], depth int, dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	return e.cpu.OneHot(ctx, input, depth, dst...)
}

func (e *GPUEngine[T]) Reshape(ctx context.Context, a *tensor.TensorNumeric[T], shape []int, dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	// Resolve -1 dimension and verify size.
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
	newSize := 1
	for _, d := range inferredShape {
		newSize *= d
	}

	// Float16Storage: zero-copy reshape (same GPU pointer, new shape).
	if e.dtype != DTypeF32 {
		if fs, ok := any(a.GetStorage()).(*tensor.Float16Storage); ok && newSize == currentSize {
			return tensor.NewWithStorage[T](inferredShape, any(fs).(tensor.Storage[T]))
		}
	}

	// GPUStorage[T]: zero-copy reshape.
	if gs, ok := a.GetStorage().(*tensor.GPUStorage[T]); ok && isFloat32[T]() && newSize == currentSize {
		return tensor.NewWithStorage[T](inferredShape, gs.View(gs.Len()))
	}

	return e.cpu.Reshape(ctx, a, shape, dst...)
}

func (e *GPUEngine[T]) ReduceMean(ctx context.Context, a *tensor.TensorNumeric[T], axis int, keepDims bool, dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	return e.gpuReduceMean(ctx, a, axis, keepDims, dst...)
}

func (e *GPUEngine[T]) Rsqrt(ctx context.Context, a *tensor.TensorNumeric[T], dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	return e.gpuRsqrt(ctx, a, dst...)
}

// GPUArgmax finds the index of the maximum element in a GPU-resident float32 tensor.
// Returns the index as an int without copying the full tensor to the host.
// Only copies back a single int32 (4 bytes) instead of the entire tensor.
func (e *GPUEngine[T]) GPUArgmax(t *tensor.TensorNumeric[float32]) (int, error) {
	gs, ok := t.GetStorage().(*tensor.GPUStorage[float32])
	if !ok {
		return 0, fmt.Errorf("GPUArgmax: tensor not GPU-resident")
	}

	e.setDevice()

	n := gs.Len()
	devInput := gs.Ptr()

	// Allocate scratch: 2 * ceil(n/256) * 4 bytes (blockVals + blockIdxs).
	numBlocks := (n + 255) / 256
	scratchSize := 2 * numBlocks * 4
	devScratch, err := e.pool.Alloc(e.deviceID, scratchSize)
	if err != nil {
		return 0, fmt.Errorf("GPUArgmax: scratch alloc: %w", err)
	}
	defer e.pool.Free(e.deviceID, devScratch, scratchSize)

	// Allocate device result (single int32).
	devResult, err := e.pool.Alloc(e.deviceID, 4)
	if err != nil {
		return 0, fmt.Errorf("GPUArgmax: result alloc: %w", err)
	}
	defer e.pool.Free(e.deviceID, devResult, 4)

	if err := e.kernels.Argmax(devInput, devResult, devScratch, n, e.stream); err != nil {
		return 0, fmt.Errorf("GPUArgmax: %w", err)
	}

	// Copy single int32 result back to host.
	var result int32
	if err := e.runtime.Memcpy(unsafe.Pointer(&result), devResult, 4, gpuapi.MemcpyDeviceToHost); err != nil {
		return 0, fmt.Errorf("GPUArgmax: D2H copy: %w", err)
	}

	return int(result), nil
}

// ConvertFP16ToF32 converts a tensor with Float16Storage to a regular float32
// GPU tensor using the FP16->F32 kernel. Returns the input unchanged if it
// does not have Float16Storage.
func (e *GPUEngine[T]) ConvertFP16ToF32(t *tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
	fs, ok := any(t.GetStorage()).(*tensor.Float16Storage)
	if !ok {
		return t, nil
	}

	fp16Ptr, _, _ := fs.GPUPtr()
	if fp16Ptr == nil {
		// CPU-side Float16Storage: decode via Slice (no GPU conversion possible).
		data := fs.Slice()
		out, err := tensor.New(t.Shape(), data)
		if err != nil {
			return nil, fmt.Errorf("ConvertFP16ToF32: create f32 tensor: %w", err)
		}
		return out, nil
	}

	e.setDevice()

	nElems := fs.Len()
	f32Bytes := nElems * f32Size
	f32Ptr, err := e.pool.Alloc(e.deviceID, f32Bytes)
	if err != nil {
		return nil, fmt.Errorf("ConvertFP16ToF32: alloc: %w", err)
	}

	if err := e.kernels.FP16ToF32(fp16Ptr, f32Ptr, nElems, e.stream); err != nil {
		e.pool.Free(e.deviceID, f32Ptr, f32Bytes)
		return nil, fmt.Errorf("ConvertFP16ToF32: kernel: %w", err)
	}

	gs, err := tensor.NewGPUStorageFromPtr[float32](f32Ptr, nElems, e.deviceID)
	if err != nil {
		e.pool.Free(e.deviceID, f32Ptr, f32Bytes)
		return nil, fmt.Errorf("ConvertFP16ToF32: gpu storage: %w", err)
	}
	out, err := tensor.NewWithStorage[float32](t.Shape(), gs)
	if err != nil {
		return nil, fmt.Errorf("ConvertFP16ToF32: wrap tensor: %w", err)
	}
	return out, nil
}

// GPUFusedRoPE applies rotary position embeddings in a single GPU kernel launch.
// This replaces Split + 4 Mul + Sub + Add + Concat (8 operations, ~10 D2D memcpy) with 1 kernel.
func (e *GPUEngine[T]) GPUFusedRoPE(input, cosAngles, sinAngles *tensor.TensorNumeric[T], rotaryDim int) (*tensor.TensorNumeric[T], error) {
	shape := input.Shape()
	if len(shape) != 3 {
		return nil, fmt.Errorf("GPUFusedRoPE: expected 3D input [batch, seq, dim], got %dD", len(shape))
	}

	batch := shape[0]
	seqLen := shape[1]
	headDim := shape[2]
	halfRotary := rotaryDim / 2

	cosShape := cosAngles.Shape()
	if len(cosShape) != 2 || cosShape[0] < seqLen || cosShape[1] < halfRotary {
		return nil, fmt.Errorf("GPUFusedRoPE: cos shape %v incompatible with seq_len=%d half_rotary=%d", cosShape, seqLen, halfRotary)
	}
	cosStride := cosShape[1]

	// Get device pointers for input, cos, sin.
	inPtr, inCleanup, err := getDevicePtr(e, input)
	if err != nil {
		return nil, fmt.Errorf("GPUFusedRoPE input: %w", err)
	}
	defer inCleanup()

	cosPtr, cosCleanup, err := getDevicePtr(e, cosAngles)
	if err != nil {
		return nil, fmt.Errorf("GPUFusedRoPE cos: %w", err)
	}
	defer cosCleanup()

	sinPtr, sinCleanup, err := getDevicePtr(e, sinAngles)
	if err != nil {
		return nil, fmt.Errorf("GPUFusedRoPE sin: %w", err)
	}
	defer sinCleanup()

	// Allocate output.
	outElems := batch * seqLen * headDim
	outBytes := outElems * f32Size
	e.setDevice()
	devOut, err := e.pool.Alloc(e.deviceID, outBytes)
	if err != nil {
		return nil, fmt.Errorf("GPUFusedRoPE alloc: %w", err)
	}

	if err := e.kernels.FusedRoPEF32(inPtr, cosPtr, sinPtr, devOut, batch, seqLen, headDim, halfRotary, cosStride, e.stream); err != nil {
		e.pool.Free(e.deviceID, devOut, outBytes)
		return nil, err
	}

	return makeGPUResult[T](e, shape, devOut, outElems)
}

// GPUFusedSwiGLU computes SwiGLU(w1, w3) = w1 * sigmoid(w1) * w3 in a single GPU kernel.
// This replaces Concat + Split + sigmoid + Mul + Mul (5 operations, ~4 D2D memcpy per layer) with 1 kernel.
func (e *GPUEngine[T]) GPUFusedSwiGLU(w1, w3 *tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	w1Shape := w1.Shape()
	w3Shape := w3.Shape()
	if len(w1Shape) == 0 || len(w3Shape) == 0 {
		return nil, fmt.Errorf("GPUFusedSwiGLU: empty shape")
	}

	// Validate shapes match.
	n1 := 1
	for _, d := range w1Shape {
		n1 *= d
	}
	n3 := 1
	for _, d := range w3Shape {
		n3 *= d
	}
	if n1 != n3 {
		return nil, fmt.Errorf("GPUFusedSwiGLU: w1 (%d elems) and w3 (%d elems) size mismatch", n1, n3)
	}

	w1Ptr, w1Cleanup, err := getDevicePtr(e, w1)
	if err != nil {
		return nil, fmt.Errorf("GPUFusedSwiGLU w1: %w", err)
	}
	defer w1Cleanup()

	w3Ptr, w3Cleanup, err := getDevicePtr(e, w3)
	if err != nil {
		return nil, fmt.Errorf("GPUFusedSwiGLU w3: %w", err)
	}
	defer w3Cleanup()

	outBytes := n1 * f32Size
	e.setDevice()
	devOut, err := e.pool.Alloc(e.deviceID, outBytes)
	if err != nil {
		return nil, fmt.Errorf("GPUFusedSwiGLU alloc: %w", err)
	}

	if err := e.kernels.FusedSwiGLUF32(w1Ptr, w3Ptr, devOut, n1, e.stream); err != nil {
		e.pool.Free(e.deviceID, devOut, outBytes)
		return nil, err
	}

	return makeGPUResult[T](e, w1Shape, devOut, n1)
}

// GPUScaledSoftmax computes softmax(input * scale) in a single GPU kernel launch.
// This replaces MulScalar + Softmax (2 kernel launches) with 1, saving 26 launches
// per token for 26 transformer layers.
func (e *GPUEngine[T]) GPUScaledSoftmax(input *tensor.TensorNumeric[T], scale float32, axis int) (*tensor.TensorNumeric[T], error) {
	if !isFloat32[T]() {
		return nil, fmt.Errorf("GPUScaledSoftmax: only float32 supported")
	}

	e.setDevice()

	if input == nil {
		return nil, fmt.Errorf("GPUScaledSoftmax: input tensor must not be nil")
	}

	shape := input.Shape()
	rank := len(shape)

	if rank == 0 {
		return nil, fmt.Errorf("GPUScaledSoftmax: scalar tensors not supported")
	}

	if axis < 0 {
		axis = rank + axis
	}

	if axis < 0 || axis >= rank {
		return nil, fmt.Errorf("GPUScaledSoftmax: axis %d out of bounds for %d dimensions", axis, rank)
	}

	n := input.GetStorage().Len()

	inner := 1
	for i := axis + 1; i < rank; i++ {
		inner *= shape[i]
	}

	outer := 1
	for i := 0; i < axis; i++ {
		outer *= shape[i]
	}

	axisSize := shape[axis]

	// FP16 paths — skip entirely for F32 compute.
	if e.dtype != DTypeF32 {
		// Native FP16 path: input already has Float16Storage on GPU — no conversion needed.
		if fs, ok := any(input.GetStorage()).(*tensor.Float16Storage); ok {
			return fp16ScaledSoftmaxNative(e, fs, input.Shape(), scale, outer, inner, axisSize)
		}
		// FP16 path: convert to FP16, run FP16 scaled softmax, convert back.
		return fp16ScaledSoftmax(e, input, scale, outer, inner, axisSize)
	}

	devIn, cleanupIn, err := getDevicePtr(e, input)
	if err != nil {
		return nil, fmt.Errorf("GPUScaledSoftmax input: %w", err)
	}
	defer cleanupIn()

	byteSize := n * f32Size
	devOut, err := e.pool.Alloc(e.deviceID, byteSize)
	if err != nil {
		return nil, fmt.Errorf("GPUScaledSoftmax alloc: %w", err)
	}

	if err := e.kernels.ScaledSoftmaxF32(devIn, devOut, outer, inner, axisSize, scale, e.stream); err != nil {
		e.pool.Free(e.deviceID, devOut, byteSize)
		return nil, err
	}

	return makeGPUResult[T](e, shape, devOut, n)
}

// GPUFusedAddRMSNorm computes sum = input + residual and
// normed = rmsnorm(sum, weight, eps) in a single GPU kernel launch.
// Both inputs are read-only; outputs go to separate buffers.
// This replaces Add + RMSNorm (2 kernel launches) with 1.
func (e *GPUEngine[T]) GPUFusedAddRMSNorm(
	input, residual *tensor.TensorNumeric[T],
	weight *tensor.TensorNumeric[T],
	eps float32,
) (normed *tensor.TensorNumeric[T], residualOut *tensor.TensorNumeric[T], scales *tensor.TensorNumeric[T], err error) {
	// FP16 paths — skip entirely for F32 compute.
	if e.dtype != DTypeF32 {
		// Native FP16 path: input and residual already have Float16Storage — no conversion needed.
		inFS, inOK := any(input.GetStorage()).(*tensor.Float16Storage)
		resFS, resOK := any(residual.GetStorage()).(*tensor.Float16Storage)
		if inOK && resOK {
			return fp16FusedAddRMSNormNative(e, inFS, resFS, input, weight, eps)
		}
		// FP16 path: decompose into F32 Add + FP16 RMSNorm.
		return fp16FusedAddRMSNorm(e, input, residual, weight, eps)
	}

	inShape := input.Shape()
	if len(inShape) < 2 {
		return nil, nil, nil, fmt.Errorf("GPUFusedAddRMSNorm: input must be at least 2D, got %v", inShape)
	}
	D := inShape[len(inShape)-1]
	rows := 1
	for i := 0; i < len(inShape)-1; i++ {
		rows *= inShape[i]
	}

	inPtr, inCleanup, err := getDevicePtr(e, input)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("GPUFusedAddRMSNorm input: %w", err)
	}
	defer inCleanup()

	// Residual is updated in-place. We need a mutable device pointer.
	resPtr, resCleanup, err := getDevicePtr(e, residual)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("GPUFusedAddRMSNorm residual: %w", err)
	}
	defer resCleanup()

	wPtr, wCleanup, err := getDevicePtr(e, weight)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("GPUFusedAddRMSNorm weight: %w", err)
	}
	defer wCleanup()

	outBytes := rows * D * f32Size
	e.setDevice()
	devNormed, err := e.pool.Alloc(e.deviceID, outBytes)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("GPUFusedAddRMSNorm alloc normed: %w", err)
	}

	devSum, err := e.pool.Alloc(e.deviceID, outBytes)
	if err != nil {
		e.pool.Free(e.deviceID, devNormed, outBytes)
		return nil, nil, nil, fmt.Errorf("GPUFusedAddRMSNorm alloc sum: %w", err)
	}

	if err := e.kernels.FusedAddRMSNormF32(inPtr, resPtr, wPtr, devNormed, devSum, eps, rows, D, e.stream); err != nil {
		e.pool.Free(e.deviceID, devNormed, outBytes)
		e.pool.Free(e.deviceID, devSum, outBytes)
		return nil, nil, nil, err
	}

	normed, err = makeGPUResult[T](e, inShape, devNormed, rows*D)
	if err != nil {
		e.pool.Free(e.deviceID, devSum, outBytes)
		return nil, nil, nil, err
	}

	residualOut, err = makeGPUResult[T](e, inShape, devSum, rows*D)
	if err != nil {
		return nil, nil, nil, err
	}

	return normed, residualOut, nil, nil
}

// GPUFusedNormAdd computes output = rmsnorm(input, weight, eps) + residual
// in a single GPU kernel launch. Replaces separate RMSNorm + Add (2 launches → 1).
func (e *GPUEngine[T]) GPUFusedNormAdd(input, weight, residual *tensor.TensorNumeric[T], eps float32) (*tensor.TensorNumeric[T], error) {
	inShape := input.Shape()
	if len(inShape) < 2 {
		return nil, fmt.Errorf("GPUFusedNormAdd: input must be at least 2D, got %v", inShape)
	}
	D := inShape[len(inShape)-1]
	rows := 1
	for i := 0; i < len(inShape)-1; i++ {
		rows *= inShape[i]
	}

	inPtr, inCleanup, err := getDevicePtr(e, input)
	if err != nil {
		return nil, fmt.Errorf("GPUFusedNormAdd input: %w", err)
	}
	defer inCleanup()

	wPtr, wCleanup, err := getDevicePtr(e, weight)
	if err != nil {
		return nil, fmt.Errorf("GPUFusedNormAdd weight: %w", err)
	}
	defer wCleanup()

	resPtr, resCleanup, err := getDevicePtr(e, residual)
	if err != nil {
		return nil, fmt.Errorf("GPUFusedNormAdd residual: %w", err)
	}
	defer resCleanup()

	outElems := rows * D
	outBytes := outElems * f32Size
	e.setDevice()
	devOut, err := e.pool.Alloc(e.deviceID, outBytes)
	if err != nil {
		return nil, fmt.Errorf("GPUFusedNormAdd alloc: %w", err)
	}

	if err := e.kernels.FusedNormAddF32(inPtr, wPtr, resPtr, devOut, eps, rows, D, e.stream); err != nil {
		e.pool.Free(e.deviceID, devOut, outBytes)
		return nil, err
	}

	return makeGPUResult[T](e, inShape, devOut, outElems)
}

// GPUFusedQKNormRoPE applies per-head RMSNorm + RoPE to combined Q+K heads
// in a single GPU kernel launch. This replaces 4 kernel launches per GQA layer.
// input: [totalHeads, headDim], weightQ/weightK: [headDim],
// cosAngles/sinAngles: [halfRotary], output: [totalHeads, headDim].
func (e *GPUEngine[T]) GPUFusedQKNormRoPE(
	input *tensor.TensorNumeric[T],
	weightQ, weightK *tensor.TensorNumeric[T],
	cosAngles, sinAngles *tensor.TensorNumeric[T],
	eps float32,
	totalHeads, headDim, numQHeads, halfRotary int,
) (*tensor.TensorNumeric[T], error) {
	inPtr, inCleanup, err := getDevicePtr(e, input)
	if err != nil {
		return nil, fmt.Errorf("GPUFusedQKNormRoPE input: %w", err)
	}
	defer inCleanup()

	wqPtr, wqCleanup, err := getDevicePtr(e, weightQ)
	if err != nil {
		return nil, fmt.Errorf("GPUFusedQKNormRoPE weightQ: %w", err)
	}
	defer wqCleanup()

	wkPtr, wkCleanup, err := getDevicePtr(e, weightK)
	if err != nil {
		return nil, fmt.Errorf("GPUFusedQKNormRoPE weightK: %w", err)
	}
	defer wkCleanup()

	cosPtr, cosCleanup, err := getDevicePtr(e, cosAngles)
	if err != nil {
		return nil, fmt.Errorf("GPUFusedQKNormRoPE cos: %w", err)
	}
	defer cosCleanup()

	sinPtr, sinCleanup, err := getDevicePtr(e, sinAngles)
	if err != nil {
		return nil, fmt.Errorf("GPUFusedQKNormRoPE sin: %w", err)
	}
	defer sinCleanup()

	outElems := totalHeads * headDim
	outBytes := outElems * f32Size
	e.setDevice()
	devOut, err := e.pool.Alloc(e.deviceID, outBytes)
	if err != nil {
		return nil, fmt.Errorf("GPUFusedQKNormRoPE alloc: %w", err)
	}

	if err := e.kernels.FusedQKNormRoPEF32(inPtr, wqPtr, wkPtr, cosPtr, sinPtr, devOut, eps, totalHeads, headDim, numQHeads, halfRotary, e.stream); err != nil {
		e.pool.Free(e.deviceID, devOut, outBytes)
		return nil, err
	}

	return makeGPUResult[T](e, []int{totalHeads, headDim}, devOut, outElems)
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
