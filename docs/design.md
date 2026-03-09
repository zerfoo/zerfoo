# Zerfoo Design Document

## 1. Overview

Zerfoo is a Go 1.25 generics-based machine learning framework. It supports CPU
and CUDA GPU execution through a unified compute engine interface. All layers,
models, and training loops delegate computation to the Engine interface, enabling
transparent hardware acceleration without modifying application code.

The framework targets importing and running open weights models (Gemma 3,
Kimi-VL, SigLIP) from ONNX via the companion ZMF model format.

Module: `github.com/zerfoo/zerfoo`

---

## 2. Core Architecture

### 2.1 Package Layout

```
tensor/               TensorNumeric[T], Storage[T], type constraints (Numeric, Float, Addable)
numeric/              Type-specific arithmetic (float32/64, float8, float16, int8, uint8), quantization
compute/              Engine[T] interface, CPUEngine, GPUEngine (//go:build cuda)
graph/                Computation graph, Node[T] interface, Builder, Parameter, topological execution
model/                Model[T], ZMF loader/exporter, global layer registry, plugin registry
layers/               Neural network layers organized by family (18 sub-packages)
  layers/core/          Add, Sub, Mul, MatMul, MatMulNBits, Cast, Concat, Constant, Conv2d, Dense,
                        FFN, FiLM, GlobalAvgPool, Linear, LMHead, MoE, Pad, Polynomial, Reshape,
                        Resize, RotaryEmbedding, Shape, Slice, SpectralFingerprint, TopK, Unsqueeze, Bias
  layers/activations/   ReLU, LeakyReLU, Sigmoid, Tanh, Gelu, FastGelu, Erf, Softmax, SwiGLU
  layers/attention/     AttentionHead, GlobalAttention, GroupQueryAttention, LocalAttention, QKNorm, SDPA
  layers/normalization/ BatchNorm, LayerNorm, RMSNorm, SimplifiedLayerNorm, SkipSimplifiedLayerNorm
  layers/embeddings/    TokenEmbedding, RotaryPositionalEmbedding
  layers/gather/        Gather (embedding-table lookup)
  layers/transpose/     Transpose
  layers/reducesum/     ReduceSum
  layers/regularization/ Dropout
  layers/components/    GradientComputer, MatrixMultiplier, WeightInitializer
  layers/transformer/   TransformerBlock
  layers/recurrent/     RNN
  layers/sequence/      S4 (structured state space)
  layers/hrm/           HModule, LModule (hierarchical recurrent model)
  layers/features/      SpectralFeature
  layers/tokenizers/    TokenizerNode
  layers/registry/      RegisterAll() -- central wiring of all layers into the model registry
training/             Trainer[T], DefaultTrainer, GradientStrategy, workflow interfaces
  training/optimizer/   Optimizer[T] interface, AdamW[T], SGD[T]
  training/loss/        Loss[T] interface, MSE[T], CrossEntropyLoss[T]
distributed/          gRPC-based distributed training: AllReduce, Barrier, Broadcast, TLS
  distributed/coordinator/ Coordinator gRPC server with worker registry and checkpoint tracking
  distributed/pb/       Generated protobuf/gRPC bindings
device/               Device, Allocator interfaces (CPU + CUDA)
config/               Generic JSON config loader with env var overrides and validation
health/               HTTP health server (/healthz, /readyz, /debug/pprof/)
metrics/              ML evaluation metrics (Pearson, Spearman, MSE, RMSE, MAE)
  metrics/runtime/      Operational instrumentation (Counter, Gauge, Histogram, InMemoryCollector)
log/                  Structured leveled logging (Debug/Info/Warn/Error, text/JSON)
shutdown/             Ordered shutdown coordinator with reverse-order Closer execution
cmd/                  CLI binaries and framework
  cmd/zerfoo/           Main binary (predict, tokenize, worker, pull, run, serve subcommands)
  cmd/cli/              Command interface, CommandRegistry, CLI runner, pull/run/serve commands
  cmd/zerfoo-predict/   Standalone predict binary
  cmd/zerfoo-tokenize/  Standalone tokenize binary
  cmd/bench-compare/    Benchmark comparison tool
  cmd/coverage-gate/    CI coverage enforcement tool
inference/            High-level inference API: Load, Generate, GenerateStream, Chat, Embed
generate/             Autoregressive generation loop, sampling (temp, topK, topP, repetition), streaming
registry/             Model registry with local cache, Pull/Get/List/Delete interface
serve/                OpenAI-compatible HTTP server (chat completions, completions, models, SSE streaming)
pkg/tokenizer/        BPE tokenizer loading from tokenizer.json, WhitespaceTokenizer for testing
data/                 Dataset container (Sample, Batch, normalization)
features/             Time-series feature transformers (Lag, Rolling, FFT)
types/                Shared type definitions (BackwardMode)
internal/xblas/       CPU BLAS wrappers (gonum GEMM for float32/64; upcast for float16/float8)
internal/cuda/        CUDA runtime CGO bindings (//go:build cuda)
internal/cublas/      cuBLAS CGO bindings (//go:build cuda)
internal/cuda/kernels/ CUDA kernel source (.cu) and Go wrappers
testing/testutils/    Test assertion helpers, MockEngine, custom mocks
tests/                Parity tests (env-var gated model forward pass tests)
```

### 2.2 Dependency Graph

```
cmd/* --> model --> graph --> compute --> tensor
           |         |         |           |
           |         |    numeric.Arithmetic[T]
           |         |                  Storage[T]
           |       types                /        \
           |                  CPUStorage[T]  GPUStorage[T]
       layers/*                                    |
           |                               internal/cuda
       graph.Node[T]
```

Key invariant: layers never access tensor data directly for computation. All
arithmetic goes through Engine[T]. This enables transparent CPU/GPU switching.

### 2.3 Architectural Boundaries

- `zerfoo/` must not import `zonnx/` or `onnx/` (verified by `make verify-architecture`).
- `zonnx/` must not import `github.com/zerfoo/zerfoo` (decoupled via `zmf` format).
- CUDA code is gated behind `//go:build cuda`. Non-CUDA builds compile without GPU dependencies.

### 2.4 Type System

Three type constraints govern the generics:

- `tensor.Numeric` -- full union: int types, uint8, float32/64, float8.Float8, float16.Float16, float16.BFloat16
- `tensor.Float` -- standard Go floats only: float32, float64
- `tensor.Addable` -- types supporting native Go operators (+, -, *); excludes custom minifloats

Custom minifloats (float8, float16, bfloat16) require `numeric.Arithmetic[T]`
for all operations since Go operators do not work on defined types.

---

## 3. Key Interfaces

### 3.1 Engine[T] (compute/engine.go)

The compute engine is the central abstraction. Every layer receives an Engine at
construction time and delegates all computation to it.

```go
type Engine[T tensor.Numeric] interface {
    Ops() numeric.Arithmetic[T]

    // Unary
    UnaryOp(ctx, a, op, dst...) (*Tensor, error)

    // Binary elementwise (with broadcasting)
    Add(ctx, a, b, dst...) (*Tensor, error)
    Sub(ctx, a, b, dst...) (*Tensor, error)
    Mul(ctx, a, b, dst...) (*Tensor, error)
    Div(ctx, a, b, dst...) (*Tensor, error)
    Pow(ctx, base, exp, dst...) (*Tensor, error)

    // Scalar ops
    AddScalar(ctx, a, scalar, dst...) (*Tensor, error)
    MulScalar(ctx, a, scalar, dst...) (*Tensor, error)
    DivScalar(ctx, a, scalar, dst...) (*Tensor, error)

    // Matrix
    MatMul(ctx, a, b, dst...) (*Tensor, error)
    Transpose(ctx, a, axes, dst...) (*Tensor, error)

    // Activations and math
    Tanh(ctx, a, dst...) (*Tensor, error)
    TanhPrime(ctx, a, upstream, dst...) (*Tensor, error)
    Softmax(ctx, a, axis, dst...) (*Tensor, error)
    Exp(ctx, a, dst...) (*Tensor, error)
    Log(ctx, a, dst...) (*Tensor, error)
    Sqrt(ctx, a, dst...) (*Tensor, error)
    Rsqrt(ctx, a, dst...) (*Tensor, error)

    // Reductions
    Sum(ctx, a, axis, keepDims, dst...) (*Tensor, error)
    ReduceSum(ctx, a, axis, keepDims, dst...) (*Tensor, error)
    ReduceMean(ctx, a, axis, keepDims, dst...) (*Tensor, error)

    // Tensor manipulation
    Reshape(ctx, a, shape, dst...) (*Tensor, error)
    Split(ctx, a, numSplits, axis) ([]*Tensor, error)
    Concat(ctx, tensors, axis, dst...) (*Tensor, error)
    Repeat(ctx, a, axis, reps, dst...) (*Tensor, error)

    // Initialization and copy
    Zero(ctx, a) error
    Zeros(ctx, a, shape) error
    Fill(ctx, a, value) error
    Copy(ctx, dst, src) error
    RandomUniform(ctx, t, min, max) error

    // Embedding
    Gather(ctx, params, indices, output) error
    ScatterAdd(ctx, dTable, indices, dOut) error

    // Encoding
    OneHot(ctx, input, depth, dst...) (*Tensor, error)
}
```

All methods accept `context.Context` for cancellation and timeout support.
Binary ops support broadcasting. Optional `dst` parameters enable in-place
operation when callers want to reuse existing tensor memory.

**Implementations:**
- `CPUEngine[T]`: Uses `numeric.Arithmetic[T]` for element-wise ops,
  `internal/xblas` (gonum BLAS) for MatMul, `parallelFor()` for multi-core.
  Integrated with `metrics.Collector`, `log.Logger`, and `MemoryTracker`.
- `GPUEngine[T]`: CUDA kernels for float32. CPU fallback for other types.
  20 methods have native GPU implementations; the rest use CPU fallback by design.

### 3.2 Node[T] (graph/node.go)

```go
type Node[T tensor.Numeric] interface {
    OpType() string
    Attributes() map[string]interface{}
    Forward(ctx, inputs...) (*Tensor, error)
    Backward(ctx, mode, outputGradient, inputs...) ([]*Tensor, error)
    Parameters() []*Parameter[T]
    OutputShape() []int
}
```

Every layer implements Node[T]. The graph Builder connects nodes into a DAG.
`Graph.Forward` and `Graph.Backward` are safe for concurrent use; a
`sync.Mutex` serializes access to the internal memo cache.

### 3.3 Storage[T] (tensor/storage.go)

```go
type Storage[T Numeric] interface {
    Len() int
    Slice() []T         // CPU: zero-copy. GPU: copies D2H.
    Set(data []T)       // CPU: direct assign. GPU: copies H2D.
    DeviceType() device.Type
}
```

- **CPUStorage[T]:** Wraps a Go `[]T` slice. `Slice()` returns the underlying
  slice directly (zero copy). `DeviceType()` returns `device.CPU`.
- **GPUStorage[T]:** Wraps a CUDA device pointer (`unsafe.Pointer`). `Slice()`
  allocates a host slice and copies D2H. `Set()` copies H2D. `Ptr()` returns
  the device pointer for kernel dispatch. `TrySlice()`/`TrySet()` return
  errors instead of logging.

### 3.4 Arithmetic[T] (numeric/arithmetic.go)

```go
type Arithmetic[T any] interface {
    Add(a, b T) T
    Sub(a, b T) T
    Mul(a, b T) T
    Div(a, b T) T
    Tanh(a T) T
    Sigmoid(a T) T
    ReLU(a T) T
    LeakyReLU(a T, alpha float64) T
    Exp(a T) T
    Log(a T) T
    Pow(a T, n float64) T
    Sqrt(a T) T
    Sum(a []T) T
    Abs(a T) T
    IsZero(a T) bool
    GreaterThan(a, b T) bool
    FromFloat32(f float32) T
    FromFloat64(f float64) T
    One() T
}
```

Concrete implementations: `Float32Ops`, `Float64Ops`, `Int8Ops`, `Uint8Ops`,
`Float16Ops`, `Float8Ops`.

### 3.5 Distributed Training (distributed/interfaces.go)

```go
type InternalStrategy[T tensor.Numeric] interface {
    Init(rank, size int, coordinatorAddress string) error
    AllReduceGradients(gradients map[string]*Tensor) error
    Barrier() error
    BroadcastTensor(t *Tensor, rootRank int) error
    Rank() int
    Size() int
    Shutdown()
}
```

gRPC-based with coordinator pattern. Workers register with a coordinator,
exchange peer addresses, then communicate directly for gradient reduction.
TLS/mTLS is supported via `distributed.TLSConfig`.

### 3.6 Layer Registration

Layers register with the model package via builder functions:

```go
type LayerBuilder[T tensor.Numeric] func(
    engine compute.Engine[T],
    ops    numeric.Arithmetic[T],
    name   string,
    params map[string]*graph.Parameter[T],
    attrs  map[string]interface{},
) (graph.Node[T], error)
```

`layers/registry.RegisterAll()` is the single entry point that wires all
standard layers (including FFN) into `model.RegisterLayer[T]`. No layer
package uses `init()` for registration. The ZMF model loader uses this
registry to reconstruct graphs from serialized specs.

---

## 4. GPU Engine Architecture

### 4.1 Build Requirements

- CUDA Toolkit 12.x (libcudart, development headers)
- cuBLAS library (libcublas)
- cuDNN library (libcudnn8 or later)
- NVIDIA GPU with Compute Capability >= 7.0 (Volta/Turing or newer)
- GCC/G++ (for CGO linking)

```
go build -tags cuda ./...
go test -tags cuda ./...
```

Compile CUDA kernels:

```
cd internal/cuda/kernels/
make                        # default: sm_75 (GCP T4)
make CUDA_ARCH=sm_120       # DGX Spark (Blackwell GB10)
make CUDA_ARCH=sm_70        # V100
```

This produces `libkernels.a` from `elementwise.cu` using `nvcc -O2 -arch=$(CUDA_ARCH)`.

### 4.2 GPU-Accelerated Operations

| Category | Operations | Backend |
|----------|-----------|---------|
| Matrix | MatMul (2D and batched, float32) | cuBLAS Sgemm |
| Matrix | MatMul (2D and batched, BFloat16) | cuBLAS GemmEx (CUDA_R_16BF) |
| Element-wise | Add, Sub, Mul, Div, Pow | Custom CUDA kernels |
| Scalar | AddScalar, MulScalar, DivScalar | Custom CUDA kernels |
| Activation | Tanh, TanhPrime | Custom CUDA kernels |
| Math | Exp, Log, Sqrt, Rsqrt | Custom CUDA kernels |
| Reduction | Sum, ReduceSum, ReduceMean | Custom CUDA kernels (shared memory) |
| Other | Softmax, Fill | Custom CUDA kernels |

### 4.3 CPU Fallback Operations

These delegate to CPUEngine by design (not compute-bound or require Go runtime):

- UnaryOp (Go function pointers)
- Transpose (metadata-only)
- Zero, Zeros, Copy
- Reshape, Split, Concat, Repeat
- Gather, ScatterAdd (integer indexing)
- OneHot, RandomUniform

### 4.4 Device-Resident Pipeline

GPU operations produce tensors with `GPUStorage`, keeping data on-device
between chained operations. Only the first input (if CPU-backed) does an H2D
copy, and only the final result (when `.Data()` is called) does a D2H copy.

```
CPU Input -> H2D (via pool) -> Kernel -> GPUStorage output
                                            |
                               GPUStorage input (zero-copy) -> Kernel -> GPUStorage output
                                                                            |
                                                               .Data() -> D2H copy -> CPU slice
```

Key helpers:
- `getDevicePtr`: GPUStorage returns device pointer directly (zero-copy).
  CPUStorage allocates from memory pool and copies H2D.
- `makeGPUResult`: Creates output tensors with GPUStorage wrapping device pointer.

### 4.5 CUDA Memory Pool

`internal/cuda/mempool.go`: Size-bucketed free-list allocator. Reuses
previously freed device memory, avoiding per-operation cudaMalloc/cudaFree.
Mutex-synchronized. Drained on `GPUEngine.Close()`.

Supports two allocation modes:
- `Alloc` / `Free`: Standard cudaMalloc (discrete device memory).
- `AllocManaged` / `FreeManaged`: cudaMallocManaged (unified memory). On
  NVLink-C2C hardware (DGX Spark GB10), managed memory avoids explicit H2D
  copies and is 200-5000x faster to allocate (demand paging). Use
  `tensor.NewManagedGPUStorage` to create tensors with managed memory.

### 4.6 CUDA Stream

Dedicated non-default stream for async kernel execution. All kernels and cuBLAS
operations execute on this stream. Created in `NewGPUEngine()`, destroyed in
`Close()`.

### 4.7 OOM Fallback

When `cudaMalloc` fails (OOM), GPU operations fall back to `CPUEngine`
transparently. An atomic counter (`OOMFallbackCount()`) tracks fallback frequency.

### 4.8 cuBLAS Row-Major Strategy

cuBLAS operates in column-major order. To compute C = A * B in row-major:
- Observe that for row-major matrices, A_row = A_col^T
- Call cublasSgemm with B as first argument, A as second, swapping m and n
- This avoids explicit transposition and works for any matrix dimensions.

### 4.9 cuDNN Integration

`internal/cudnn/` provides CGo bindings wrapping libcudnn behind `//go:build cuda`.
GPUEngine gains a `cudnnHandle` field alongside `cublasHandle`. cuDNN-accelerated
operations are non-interface methods on GPUEngine in `compute/gpu_cudnn.go`:

| Operation | cuDNN Function | Notes |
|-----------|---------------|-------|
| Conv2dForward | cudnnConvolutionForward | IMPLICIT_GEMM algo, grouped conv via SetGroupCount |
| BatchNormForwardInference | cudnnBatchNormalizationForwardInference | Spatial mode |
| CudnnActivationForward | cudnnActivationForward | ReLU, Sigmoid, Tanh |
| CudnnPoolingForward | cudnnPoolingForward | Max, AvgIncPad, AvgExcPad |
| CudnnSoftmaxForward | cudnnSoftmaxForward | Channel-mode softmax |
| Conv2dBackwardData | cudnnConvolutionBackwardData | Algo0, workspace-based |
| Conv2dBackwardFilter | cudnnConvolutionBackwardFilter | Algo0, workspace-based |
| BatchNormForwardTraining | cudnnBatchNormalizationForwardTraining | Saves mean/invVar for backward |
| CudnnBatchNormBackward | cudnnBatchNormalizationBackward | Returns dx, dScale, dBias |
| CudnnActivationBackward | cudnnActivationBackward | ReLU, Sigmoid, Tanh |
| CudnnPoolingBackward | cudnnPoolingBackward | Max, AvgIncPad, AvgExcPad |

See [ADR-008](adr/008-cudnn-integration.md) and [ADR-014](adr/014-cudnn-backward-pass.md) for architecture decisions.

### 4.10 TensorRT Integration

`internal/tensorrt/` provides CGo bindings wrapping TensorRT's C++ API via a thin
C shim in `cshim/trt_capi.h/cpp`. Pre-compiled into `libtrt_capi.a` via Makefile.

The inference pipeline supports TensorRT via `WithBackend("tensorrt")`:

1. **Graph conversion** (`inference/tensorrt_convert.go`): Walks the graph in
   topological order, mapping each node to a TRT layer. Supported ops: MatMul,
   Add/Sub/Mul/Div, ReLU/Sigmoid/Tanh, Softmax, Reshape, Transpose, ReduceSum,
   Conv, Dense, Linear, Constant. Unsupported ops return `UnsupportedOpError`.

2. **Engine caching** (`inference/tensorrt_cache.go`): SHA-256 key from
   (modelID, precision, gpuArch). Cached at `~/.cache/zerfoo/tensorrt/`.
   Cache hit skips the expensive TRT build step.

3. **TRT pipeline** (`inference/tensorrt_pipeline.go`): `TRTInferenceEngine`
   wraps the TRT execution context with `Forward()` and `Close()` methods.

FP16 precision via `WithPrecision("fp16")` sets the `FP16` builder flag.

4. **Dynamic shapes** (`DynamicShapeConfig`): Optimization profiles with
   min/opt/max dims per input allow a single engine to handle variable batch
   sizes and sequence lengths. `SetInputShape` is called before each enqueue.
   Cache keys incorporate shape ranges for correctness.

See [ADR-009](adr/009-tensorrt-integration.md) and [ADR-016](adr/016-tensorrt-dynamic-shapes.md) for architecture decisions.

### 4.11 CUTLASS Flash Attention

Flash attention fuses Q*K^T scaling, softmax, and V weighting into a single
tiled CUDA kernel, reducing memory from O(n^2) to O(n) and eliminating three
intermediate kernel launches.

- **Kernel**: `internal/cuda/kernels/flash_attention.cu` -- online softmax
  (log-sum-exp trick), shared memory for K/V tiles, causal masking. BLOCK_SIZE=64,
  MAX_HEAD_DIM=128.
- **Dispatch**: Build-tag-gated pair `layers/attention/flash_cuda.go` /
  `flash_nocuda.go` (`//go:build cuda && cutlass` / `!(cuda && cutlass)`).
  `ScaledDotProductAttention.Forward` calls `tryFlashForward` before the naive
  path when no arbitrary mask is provided.
- **Scope**: Float32 forward only. Backward pass deferred. Head dim > 128 or
  arbitrary masks fall back to naive attention.

See [ADR-010](adr/010-cutlass-flash-attention.md) for architecture decisions.

### 4.12 CUDA File Layout

```
compute/
  gpu_engine.go            GPUEngine (GRAL interfaces, pool, stream) (//go:build cuda)
  gpu_cudnn.go             DNN-accelerated operations via GRAL (//go:build cuda)
  gpu_kernels.go           getDevicePtr, makeGPUResult, kernel dispatch via GRAL (//go:build cuda)

tensor/
  storage.go               Storage[T] interface, CPUStorage[T], NewWithStorage
  gpu_storage.go           GPUStorage[T] with gpuapi.Runtime (//go:build cuda)
  transfer.go              ToGPU/ToCPU helpers via GRAL Runtime (//go:build cuda)

internal/gpuapi/
  doc.go                   Package identity
  runtime.go               Runtime, Stream, MemcpyKind interfaces
  blas.go                  BLAS interface (Sgemm)
  dnn.go                   DNN interface (conv, batchnorm, activation, pooling, softmax)
  kernels.go               KernelRunner interface (17 element-wise/reduction ops)
  mempool.go               MemPool interface
  gpuapi_test.go           Compile-time interface assertions
  cuda_runtime.go          CUDARuntime adapter (//go:build cuda)
  cuda_blas.go             CUDABlas adapter (//go:build cuda)
  cuda_dnn.go              CUDADNN adapter (//go:build cuda)
  cuda_kernels.go          CUDAKernels adapter (//go:build cuda)
  cuda_mempool.go          CUDAMemPool adapter (//go:build cuda)

device/
  cuda_device.go           CUDA device abstraction (//go:build cuda)
  cuda_allocator.go        CUDA memory allocator (//go:build cuda)

internal/cuda/
  runtime.go               CUDA runtime + Stream bindings (//go:build cuda)
  mempool.go               Size-bucketed device memory pool (//go:build cuda)
  kernels/
    elementwise.cu         CUDA kernel source (17 kernels, stream-aware)
    elementwise.go         CGO bindings for kernels (//go:build cuda)
    flash_attention.cu     Tiled flash attention kernel (online softmax)
    flash_attention.h      C function declaration
    flash_attention.go     CGO binding (//go:build cuda && cutlass)
    gemm_int8.cu           INT8 mixed-precision GEMM kernel
    gemm_int8.h            C function declaration
    gemm_int4.cu           INT4 mixed-precision GEMM kernels (left-mul + right-mul)
    gemm_int4.h            C function declarations
    gemm_quantized.go      CGO bindings for quantized GEMM (//go:build cuda && cutlass)
    Makefile               nvcc compilation

internal/cublas/
  cublas.go                cuBLAS + SetStream bindings (//go:build cuda)

internal/cudnn/
  doc.go                   Package identity (no build tag)
  cudnn.go                 cuDNN CGo bindings (//go:build cuda)

internal/tensorrt/
  doc.go                   Package identity (no build tag)
  tensorrt.go              TensorRT Go bindings (//go:build cuda)
  Makefile                 Compiles cshim/ into libtrt_capi.a
  cshim/
    trt_capi.h             C shim header for TensorRT C++ API
    trt_capi.cpp           C shim implementation

inference/
  tensorrt_convert.go      Graph-to-TRT converter (//go:build cuda)
  tensorrt_cache.go        TRT engine caching (//go:build cuda)
  tensorrt_pipeline.go     TRT inference engine wrapper (//go:build cuda)

layers/attention/
  flash_cuda.go            Flash attention GPU dispatch (//go:build cuda && cutlass)
  flash_nocuda.go          Naive fallback (//go:build !(cuda && cutlass))
```

CGO linker flags:

```
internal/cuda/runtime.go:       -lcudart
internal/cublas/cublas.go:      -lcublas
internal/cudnn/cudnn.go:        -lcudnn
internal/tensorrt/tensorrt.go:  -L${SRCDIR} -ltrt_capi -lnvinfer -lstdc++
internal/cuda/kernels/*.go:     -L${SRCDIR} -lkernels -lcudart -lstdc++
```

### 4.13 GPU Runtime Abstraction Layer (GRAL)

`internal/gpuapi/` defines vendor-neutral interfaces that decouple `compute/`
and `tensor/` from any specific GPU SDK. GPUEngine stores five GRAL interfaces
(`Runtime`, `BLAS`, `DNN`, `KernelRunner`, `MemPool`) instead of vendor-specific
handles. GPUStorage stores a `Runtime` for memory operations.

CUDA adapters in the same package implement these interfaces by delegating to
`internal/cuda`, `internal/cublas`, and `internal/cudnn`. ROCm adapters delegate
to `internal/hip`, `internal/rocblas`, and `internal/miopen`. Adding a new
backend requires implementing the five interfaces -- no changes to compute/ or
tensor/ are needed.

The DNN interface abstracts at the operation level: callers pass shapes as
`[4]int` arrays and the adapter manages vendor-specific descriptors internally.
See [ADR-011](adr/011-gpu-runtime-abstraction-layer.md) for details.

### 4.14 AMD ROCm Backend

ROCmEngine mirrors GPUEngine's architecture using HIP/rocBLAS/MIOpen adapters.
All 35 Engine[T] methods delegate to CPUEngine; the GRAL infrastructure is wired
for GPU acceleration when AMD hardware is available.

```
internal/hip/runtime.go:           -lamdhip64
internal/hip/mempool.go:           (pure Go, uses hip.Malloc/Free)
internal/rocblas/rocblas.go:       -lrocblas
internal/miopen/miopen.go:         -lMIOpen
internal/hip/kernels/*.go:         -L${SRCDIR} -lhipkernels -lamdhip64 -lstdc++
```

Integration: `device/rocm_device.go` auto-registers AMD GPUs via init().
`inference/engine_rocm.go` routes "rocm" / "rocm:N" to ROCmEngine.
`layers/attention/flash_rocm.go` dispatches fused attention on AMD GPUs.
See [ADR-012](adr/012-amd-rocm-backend.md) for details.

### 4.15 OpenCL Backend

OpenCLEngine mirrors GPUEngine's architecture using OpenCL/CLBlast adapters.
All 35 Engine[T] methods delegate to CPUEngine; the GRAL infrastructure is wired
for GPU acceleration when OpenCL hardware is available.

```
internal/opencl/runtime.go:                    -lOpenCL
internal/opencl/kernels/kernels.go:            -lOpenCL (embeds elementwise.cl)
internal/clblast/clblast.go:                   -lclblast -lOpenCL
internal/gpuapi/opencl_{runtime,blas,dnn,kernels,mempool}.go
```

No DNN library: OpenCLDNN returns ErrNotSupported for all operations; the
compute engine falls back to CPU. Kernels are compiled from .cl source at
runtime via `clCreateProgramWithSource`.

Integration: `device/opencl_device.go` auto-registers OpenCL GPUs via init().
`inference/engine_opencl.go` routes "opencl" / "opencl:N" to OpenCLEngine.
See [ADR-013](adr/013-opencl-backend.md) for details.

### 4.16 Parity Tolerances

- MatMul: 1e-5 relative error
- Element-wise ops: 1e-6 relative error
- Reductions (Sum, Mean): 1e-5 relative error
- Flash attention: 1e-3 absolute error (online softmax reordering)

### 4.17 Compatible Hardware

| GPU | Arch | CUDA_ARCH | Memory | Platform |
|-----|------|-----------|--------|----------|
| Tesla T4 | Turing | sm_75 | 16 GB | GCP |
| L4 | Ada Lovelace | sm_89 | 24 GB | GCP |
| Tesla V100 | Volta | sm_70 | 16 GB | GCP |
| A100 | Ampere | sm_80 | 40/80 GB | GCP |
| DGX Spark GB10 | Blackwell | sm_120 | 128 GB unified | Local (ARM64) |

---

## 5. Model Import Pipeline

### 5.1 Pipeline Overview

```
HuggingFace ONNX model
    |
    v
zonnx (ONNX-to-ZMF converter, separate repo)
    |
    v
ZMF file (github.com/zerfoo/zmf format)
    |
    v
model.LoadModelFromZMF[T](engine, ops, path) -> Model[T]{Graph, ZMFVersion}
    |
    v
graph.Graph[T].Forward(ctx, inputs...)
```

### 5.2 ZMF Model Format

ZMF (Zerfoo Model Format) is a protobuf-based container storing:
- Graph topology: nodes with op types, input/output edges, attributes
- Parameters: named tensors with shape, dtype, and serialized data
- Metadata: version, source model info

The `model` package deserializes ZMF files and reconstructs `graph.Graph[T]`
using the registered `LayerBuilder[T]` functions from `layers/registry`.

### 5.3 Supported Model Architectures

- **Gemma 3**: Full text decoder pipeline. Parity test: `tests/parity/gemma3_test.go`
- **SigLIP**: Vision encoder (patch embedding + transformer). Parity test: `tests/parity/siglip_test.go`
- **Kimi-VL**: Vision-language connector. Parity test: `tests/parity/siglip_test.go`

### 5.4 Layer Coverage for Open Weights

Core operators implemented for model import:
- MatMul, MatMulNBits (4-bit quantized), Conv2d, Dense, Linear, LMHead
- RMSNorm, LayerNorm, SimplifiedLayerNorm, SkipSimplifiedLayerNorm, BatchNorm
- GroupQueryAttention, ScaledDotProductAttention, QKNorm
- RotaryEmbedding, TokenEmbedding
- Gelu, FastGelu, SwiGLU, Sigmoid, Softmax, Erf
- MoEGate, MixtureOfExperts
- Cast, Concat, Reshape, Slice, Unsqueeze, Shape, Pad, Resize, TopK
- Constant, GlobalAvgPool, Gather, ReduceSum, Transpose, Dropout

---

## 6. Operations

### 6.1 Configuration

Zerfoo loads configuration from a JSON file with optional environment variable
overrides. Use `config.LoadWithEnv[T](path, prefix)` in code, or pass a JSON
file path to the CLI.

Engine configuration:

| Field | JSON key | Env var | Default | Description |
|-------|----------|---------|---------|-------------|
| Device | device | DEVICE | -- | "cpu" or "cuda" |
| MemoryLimitMB | memory_limit_mb | MEMORY_LIMIT_MB | 0 | Max memory in MB (0=unlimited) |
| LogLevel | log_level | LOG_LEVEL | "info" | debug, info, warn, error |

Training configuration:

| Field | JSON key | Env var | Default | Description |
|-------|----------|---------|---------|-------------|
| BatchSize | batch_size | BATCH_SIZE | -- | Training batch size |
| LearningRate | learning_rate | LEARNING_RATE | -- | Learning rate |
| Optimizer | optimizer | OPTIMIZER | -- | "sgd" or "adam" |
| Epochs | epochs | EPOCHS | 0 | Number of epochs |
| CheckpointInterval | checkpoint_interval | CHECKPOINT_INTERVAL | 0 | Steps between checkpoints |

Distributed configuration:

| Field | JSON key | Env var | Default | Description |
|-------|----------|---------|---------|-------------|
| CoordinatorAddress | coordinator_address | COORDINATOR_ADDRESS | -- | Host:port of coordinator |
| TimeoutSeconds | timeout_seconds | TIMEOUT_SECONDS | 0 | RPC timeout in seconds |
| TLSEnabled | tls_enabled | TLS_ENABLED | false | Enable TLS for gRPC |

### 6.2 Startup Sequence

1. Load configuration from file (apply env overrides).
2. Create the compute engine (CPU or GPU).
3. Set memory limit if configured.
4. Register the engine with the shutdown coordinator.
5. Start the health check server (default :8081).
6. If distributed: start the gRPC server, connect to peers.
7. Load the model and begin serving.

### 6.3 Health Checks

The health server exposes:

- `GET /healthz` -- Liveness probe. Returns 200 if the process is alive.
- `GET /readyz` -- Readiness probe. Returns 200 if all checks pass, 503 otherwise.
- `GET /debug/pprof/` -- pprof index for profiling.

### 6.4 Shutdown

Zerfoo handles SIGINT and SIGTERM for graceful shutdown:

1. Signal received.
2. Shutdown coordinator triggered.
3. Closers called in reverse registration order (distributed strategy disconnects, engine releases resources).
4. Root context canceled.
5. Process exits.

### 6.5 Logging

Structured leveled logging via `log.Logger`:

- **DEBUG**: Detailed operation-level information (tensor shapes, timing).
- **INFO**: Startup, configuration, connection events.
- **WARN**: Recoverable errors (e.g. OOM fallback from GPU to CPU).
- **ERROR**: Unrecoverable errors (connection failures, invalid configuration).

Output formats: text or JSON.

### 6.6 Metrics

Operational instrumentation via `metrics/runtime.Collector`:

- Counters: `op_count_<OpName>`, `allreduce_count`
- Histograms: `op_duration_seconds`, `allreduce_duration_seconds`
- Access via `collector.Snapshot()`

ML evaluation metrics via `metrics.CalculateMetrics()`: Pearson, Spearman,
MSE, RMSE, MAE.

### 6.7 TLS Configuration

For production gRPC, configure `distributed.TLSConfig`:

```go
tlsCfg := &distributed.TLSConfig{
    CACertPath: "/path/to/ca.pem",
    CertPath:   "/path/to/server.pem",
    KeyPath:    "/path/to/server-key.pem",
}
```

For mutual TLS (mTLS), provide client cert/key on both sides.

---

## 7. Testing Architecture

### 7.1 Test Coverage

Target: >= 95% statement coverage for all testable packages.

Documented exceptions (unreachable `tensor.New` error paths):
- layers/gather: 93.1%
- layers/embeddings: 93.5%
- layers/features: 93.8%
- testing/testutils: 94.5%

### 7.2 Testing Patterns

- Table-driven tests using standard `testing` package (no testify).
- Parity tests comparing GPU vs CPU output for every GPU-accelerated method.
- Model parity tests gated by env vars (GEMMA3_ZMF_PATH, SIGLIP_ZMF_PATH, KIMI_CONNECTOR_ZMF_PATH,
  LLAMA3_ZMF_PATH, MISTRAL_ZMF_PATH, QWEN25_ZMF_PATH, PHI4_ZMF_PATH, DEEPSEEK_ZMF_PATH).
- Parity tests cover 6 model families: Gemma 3, Llama 3, Mistral, Qwen 2.5, Phi-4, DeepSeek V3.
- Integration tests for cross-package workflows.
- Numerical gradient checking via finite differences.
- MockEngine for unit testing layers in isolation.
- Model parity on DGX Spark: 8 PASS (Llama3, Qwen25, FlashAttentionGQA),
  13 SKIP (no ZMF: Mistral, Phi4, Gemma3, DeepSeek, SigLIP; 1 device: MultiGPU).
  10 ONNX compatibility fixes applied during Phase 21. See [ADR-018](adr/018-model-parity-testing.md).

### 7.3 Excluded from Coverage Target

| Package | Reason |
|---------|--------|
| distributed/pb/ | Generated protobuf code |
| cmd/zerfoo/ | Main entrypoint, no testable logic |
| cmd/zerfoo-predict/ | Main entrypoint; logic in cmd/cli/ |
| cmd/zerfoo-tokenize/ | Main entrypoint; logic in pkg/tokenizer/ |
| types/ | Type definitions only |

### 7.4 GPU Test Execution

```
# 1. Compile CUDA kernels
cd internal/cuda/kernels && make

# 2. Run GPU test suite
go test -tags cuda -count=1 -v \
    ./compute/ ./tensor/ ./internal/cuda/... ./internal/cublas/... ./device/

# 3. Run parity tests (GPU vs CPU)
go test -tags cuda -run Parity -v ./compute/
```

---

## 8. Build and CI

### 8.1 Dependencies

Direct (go.mod):
- gonum.org/v1/gonum (BLAS)
- google.golang.org/grpc + protobuf (distributed training)
- github.com/zerfoo/zmf (model format)
- github.com/zerfoo/float16, float8 (custom numeric types)
- github.com/google/go-cmp (test comparisons)

### 8.2 Makefile Targets

| Target | Purpose |
|--------|---------|
| test | Full test suite |
| test-coverage | Coverage report (coverage.out) |
| coverage-report | HTML coverage visualization |
| proto | gRPC protobuf compilation |
| format | gofmt + goimports + gofumpt |
| lint | golangci-lint check |
| lint-fix | Auto-fix lint issues |
| check | Full QA pipeline |
| ci | Full CI simulation |
| verify-architecture | Enforce zerfoo/zonnx decoupling |

### 8.3 Pre-Commit Hook

- Runs `golangci-lint` on staged packages.
- Runs `go test ./...`.
- Rejects commits touching files in multiple directories.

### 8.4 CI Pipeline (GitHub Actions)

- Unit tests on push/PR to main (excludes parity tests).
- golangci-lint with 5m timeout.
- Parity tests (currently non-blocking, gated by env vars).
- Nightly toy training pipeline.

---

## 9. Troubleshooting

### 9.1 Common Errors

**"memory limit exceeded"**: Total tensor allocation exceeds configured
`memory_limit_mb`. Increase the limit, reduce batch size, or set to 0
(unlimited).

**"input tensor cannot be nil"**: An operation received a nil tensor. Verify
all tensors are initialized with `tensor.New` and check prior operations for
errors.

**"invalid shapes for matrix multiplication"**: Inner dimensions incompatible
for MatMul. For A @ B, A must be [..., m, k] and B must be [..., k, n]. Use
`Reshape` to fix shape mismatches.

**"context canceled" / "context deadline exceeded"**: Operation took too long or
parent context was canceled. Increase the timeout, or note that this is expected
during SIGTERM shutdown.

### 9.2 GPU-Specific Issues

**"CUDA not found" / build fails with cuda tag**: Install CUDA Toolkit 12.x.
Ensure `nvcc` is in PATH. Set `CUDA_HOME` if non-standard location.

**GPU OOM**: Operations fail with CUDA allocation errors or log "GPU OOM
fallback to CPU" at WARN level. Reduce batch size, lower `memory_limit_mb`,
or monitor with `nvidia-smi -l 1`.

**CUDA driver version mismatch**: Check `nvidia-smi` for driver version and
`nvcc --version` for CUDA version. Refer to NVIDIA CUDA compatibility matrix.

### 9.3 Distributed Issues

**"connection refused"**: Verify coordinator is running on the configured
address. Check firewall for port 50051/TCP. For TLS: verify certificates
are valid.

**"transport: authentication handshake failed"**: TLS certificate mismatch.
Verify both sides use the same CA. Check certificate expiry with
`openssl x509 -in cert.pem -noout -dates`.

**Timeout during Barrier/AllReduce**: Check all workers are running. Increase
`timeout_seconds`. Check for network latency or straggler workers.

### 9.4 Performance Profiling

```
# CPU profile (30 seconds)
go tool pprof http://localhost:8081/debug/pprof/profile?seconds=30

# Heap profile
go tool pprof http://localhost:8081/debug/pprof/heap

# Goroutine dump
curl http://localhost:8081/debug/pprof/goroutine?debug=2
```

---

## 10. Known Limitations

1. float32 only for GPU -- other types fall back to CPU transparently.
2. No broadcasting in GPU kernels -- broadcast cases fall back to CPU.
3. Single GPU -- no multi-GPU or distributed GPU support.
4. cuDNN operations (Conv2d, BatchNorm, activations, pooling, softmax) are non-interface methods on GPUEngine -- layers must call them explicitly rather than through Engine[T].
5. No mixed precision -- full float32 throughout.
6. Default device -- always uses cuda:0, no device selection API.
7. Hardware validation pending -- GCP GPU quota request pending.
8. float16/float8 GEMM upcasts to float32 -- no native half-precision kernels.
9. Generics wiring hardcodes float32 -- registry, worker node, CLI all use float32.
10. Embeddings not yet supported -- inference.Embed returns an error (no hidden state access).
11. KV cache is optional -- not all graph architectures support it.

---

## 11. Ecosystem

### 11.1 Companion Repositories

- **zmf** (`github.com/zerfoo/zmf`): Zerfoo Model Format protobuf library.
- **zonnx** (`github.com/zerfoo/zonnx`): ONNX-to-ZMF converter with per-operator builders.
- **float16** (`github.com/zerfoo/float16`): IEEE 754 float16 and bfloat16 types for Go.
- **float8** (`github.com/zerfoo/float8`): E4M3 float8 type for Go.
- **gemma3** (`github.com/zerfoo/gemma3`): Gemma 3 model support and conversion scripts.

### 11.2 Inference Pipeline (Phase 8)

The inference pipeline provides an embeddable Go-native API for model loading and text generation.

**Loading:** `inference.Load(modelID, opts...)` resolves a model via `registry.ModelRegistry`, reads `config.json` (metadata), `tokenizer.json` (BPE tokenizer), and `model.zmf` (weights), then wires a `generate.Generator[float32]` with a `graph.Graph[float32]` and `compute.CPUEngine[float32]`.

**Generation:** `generate.Generator.Generate(ctx, prompt, config)` runs the autoregressive loop:
1. Encode prompt via BPE tokenizer
2. Prefill: single forward pass through the full graph
3. Decode: loop picking one token at a time using the sampling pipeline
4. Sampling: temperature scaling, top-k filtering, top-p (nucleus) sampling, repetition penalty, then softmax + weighted random (or argmax at temperature=0)
5. Stop on EOS token, stop string, or max tokens

**Streaming:** `generate.Generator.GenerateStream(ctx, prompt, config, handler)` delivers tokens incrementally via `generate.TokenStream.OnToken(token, done)`.

**KV Cache:** `generate.KVCache[T]` stores per-layer key/value tensors, passed via context (`generate.WithKVCache`/`generate.GetKVCache`). Avoids recomputing attention over prior tokens during autoregressive decode.

**Serve:** `serve.NewServer(model).Handler()` returns an `http.Handler` implementing:
- `POST /v1/chat/completions` -- OpenAI chat completion (non-streaming and SSE)
- `POST /v1/completions` -- OpenAI text completion (non-streaming and SSE)
- `GET /v1/models` -- model listing

**CLI Commands:**
- `zerfoo pull <model-id>` -- download and cache a model via registry
- `zerfoo run <model-id>` -- interactive chat with streaming output
- `zerfoo serve --port <port>` -- start OpenAI-compatible HTTP server

### 11.3 Data Flow

```
HuggingFace model (ONNX/SafeTensors)
    |
    v (zonnx converter)
ZMF file + config.json + tokenizer.json
    |
    v (registry.Pull / inference.Load)
inference.Model
    |
    +-- Generate(ctx, prompt) -> string
    +-- GenerateStream(ctx, prompt, handler) -> error
    +-- Chat(ctx, messages) -> Response
    |
    v (serve.NewServer)
OpenAI-compatible HTTP API
```

## 12. Multi-Architecture Support (Phase 9)

Phase 9 extends the inference pipeline to support multiple model architectures
beyond Gemma 3. Each architecture has distinct attention, normalization, and
routing patterns that are handled through the config registry and layer builders.

### 12.1 Supported Model Families

| Model Family | Attention | RoPE Variant | Normalization | MoE | Config Parser |
|---|---|---|---|---|---|
| Gemma 3 | GQA | Standard | RMSNorm | No | `gemma3` |
| LLaMA 3 | GQA | Standard | RMSNorm | No | `llama` |
| Mistral | GQA | Standard | RMSNorm | No | `mistral` |
| Qwen 2.5 | GQA | YaRN scaling | RMSNorm | No | `qwen2` |
| Phi-4 | GQA | Partial (0.75) | LayerNorm | No | `phi` |
| DeepSeek V3 | MLA | Standard | RMSNorm | Shared expert | `deepseek` |

### 12.2 Architecture-Specific Features

**YaRN Scaling (Qwen 2.5):** `embeddings.WithYaRNScaling(factor, origMaxLen)`
classifies frequency bands into low/medium/high and applies differential scaling
to extend context beyond the original training length.

**Partial RoPE (Phi-4):** `embeddings.WithRotaryDimFraction(fraction)` rotates
only a fraction of head dimensions, leaving the rest as pass-through. Phi-4
uses fraction=0.75.

**Multi-head Latent Attention (DeepSeek V3):** `attention.MultiHeadLatentAttention`
compresses KV into a low-rank latent via down-projection (`W_DKV`), then
up-projects to keys (`W_UK`) and values (`W_UV`), reducing KV cache size.

**Shared Expert MoE (DeepSeek V3):** `core.MixtureOfExperts.SharedExpert` runs
one expert on every token and adds its output to the weighted routed sum.

**Tied Embeddings (Phi-4, Gemma 3):** `core.NewTiedLMHead` reuses the token
embedding weight matrix (transposed) as the output projection, halving the
parameter count for the LM head.

### 12.3 Config Registry

`inference.ConfigRegistry` maps model family names to config parsers that extract
`ModelMetadata` from `config.json`. Each parser reads architecture-specific fields
(e.g., `rope_scaling`, `partial_rotary_factor`, `n_shared_experts`) and maps them
to the common metadata struct. Global attributes (rope scaling, partial rotation)
are injected via `model.WithGlobalAttributes` during graph construction.

### 12.4 Parameter Name Resolver

`model.ParamResolver` maps architecture-specific weight names (e.g., Llama's
`q_proj.weight` vs DeepSeek's `kv_a_proj.weight`) to canonical names used by
Zerfoo layer builders. Called as a fallback during model building when exact
parameter names are not found. See [ADR-005](adr/005-multi-architecture-support.md).

---

## 13. Multi-GPU Support

Phase 10 adds device affinity to all CUDA-aware components and NCCL-based
collective operations for distributed GPU training.

### 13.1 Device Affinity

Every CUDA component (MemPool, GPUEngine, GPUStorage, cudaAllocator) carries a
`deviceID` field and calls `cuda.SetDevice(deviceID)` before all CUDA operations.
Constructors accept optional device ID parameters (default 0) for backwards
compatibility. Cross-device tensor transfer uses `cudaMemcpyPeer` for peer-to-peer
D2D copy without CPU staging.

### 13.2 Inference Device Selection

`inference.Load(modelID, WithDevice("cuda:N"))` creates a GPUEngine on device N.
Build-tag-gated files (`engine_cuda.go`, `engine_nocuda.go`) handle conditional
compilation. `Model.Close()` releases GPU resources.

### 13.3 NCCL Collective Operations

`internal/nccl/` binds NCCL functions (AllReduce, Broadcast, GroupStart/End) behind
`//go:build cuda`. `NcclStrategy[T]` in `distributed/` implements `InternalStrategy[T]`
for GPU-native gradient exchange, operating directly on device pointers without CPU
round-trips. See [ADR-007](adr/007-multi-gpu-architecture.md).

---

## 14. Architectural Decision Records

Stable design decisions extracted from the implementation plan into self-contained
ADR files in `docs/adr/`.

| ADR | Title | Phase | Key Decision |
|-----|-------|-------|-------------|
| [001](adr/001-enterprise-production-readiness.md) | Enterprise Production Readiness | 4+7 | Logging, metrics, config, health, CI gates, dead code removal, graph thread safety |
| [002](adr/002-distributed-training-protocol.md) | Distributed Training Protocol | 5 | Star-topology AllReduce, counter-based Barrier, WorkerNode lifecycle |
| [003](adr/003-open-weights-model-import.md) | Open Weights Model Import | 6 | 4-bit weights, Conv2d strategy, MoE design, 13 new operators |
| [004](adr/004-embeddable-inference-library.md) | Embeddable Inference Library | 8 | BPE tokenizer, KV cache, generation loop, sampling, streaming, serve |
| [005](adr/005-multi-architecture-support.md) | Multi-Architecture Support | 9 | Config registry, param resolver, YaRN, partial RoPE, MLA, shared MoE |
| [006](adr/006-gpu-engine-architecture.md) | GPU Engine Architecture | 2-3 | CUDA float32, memory pool, cuBLAS row-major, OOM fallback, parity tolerances |
| [007](adr/007-multi-gpu-architecture.md) | Multi-GPU Architecture | 10 | Device affinity, NCCL bindings, NcclStrategy, cross-device transfer, inference device selection |
| [008](adr/008-cudnn-integration.md) | cuDNN Integration | 11 | cuDNN bindings, Conv2d/BatchNorm/activation/pooling GPU acceleration, descriptor management |
| [009](adr/009-tensorrt-integration.md) | TensorRT Integration | 12 | C shim for C++ API, subgraph conversion, engine caching, FP16 precision, inference pipeline |
| [010](adr/010-cutlass-flash-attention.md) | CUTLASS Flash Attention | 13 | Tiled flash attention kernel, CUTLASS templates, causal mask, build-tag-gated dispatch |
| [011](adr/011-gpu-runtime-abstraction-layer.md) | GPU Runtime Abstraction Layer | 14 | GRAL interfaces decouple compute/tensor from vendor SDKs, CUDA adapters, operation-level DNN |
| [012](adr/012-amd-rocm-backend.md) | AMD ROCm Backend | 15 | HIP runtime, rocBLAS, MIOpen adapters, HIP kernels, device registration, inference routing |
| [013](adr/013-opencl-backend.md) | OpenCL Backend | 16 | OpenCL runtime, CLBlast, runtime kernel compilation, DNN stub, cl_mem memory pool |
| [014](adr/014-cudnn-backward-pass.md) | cuDNN Backward Pass | 17 | Backward CGo bindings, CUDA DNN adapter, GPUEngine backward methods for training |
| [015](adr/015-cutlass-quantized-gemm.md) | CUTLASS Quantized GEMM | 18 | INT8/INT4 CUDA kernels, right-multiply variant, MatMulNBits GPU dispatch |
| [016](adr/016-tensorrt-dynamic-shapes.md) | TensorRT Dynamic Shapes | 19 | Optimization profiles, min/opt/max dims, SetInputShape, dynamic cache keys |
| [017](adr/017-dgx-spark-hardware-validation.md) | DGX Spark Hardware Validation | 20 | ARM64 build fixes, sm_121 BLOCK_SIZE=32, TRT 10 API, benchmark results |
| [018](adr/018-model-parity-testing.md) | Model Parity Testing | 21 | 18 ONNX fixes, 18 PASS (6 model families), 4 SKIP, parity automation script |
| [019](adr/019-phase22-bf16-unified-siglip.md) | BF16 GEMM, Unified Memory, SigLIP Fix | 22 | BF16 cuBLAS GemmEx, cudaMallocManaged, Squeeze scalar fix |
| [020](adr/020-q4-quantized-dot-product.md) | Q4 Quantized Dot Product | 29 | NEON nibble-extract + FMA, row-level assembly, B-operand GEMV |
| [021](adr/021-graph-compilation-worker-pool.md) | Graph Compilation and Worker Pool | 30 | Pre-compiled instruction sequence, persistent worker pool, buffer arena |
| [022](adr/022-gpu-first-inference-pipeline.md) | GPU-First Inference Pipeline | 32 | Upload weights to GPU, GPU-resident forward pass, D2H only for final logits |
| [023](adr/023-gpu-scalar-ops-d2h-elimination.md) | GPU Scalar Ops and D2H Elimination | 33 | PowScalar/SubScalar kernels, scalar-broadcast detection, GPU Split/Concat |
| [024](adr/024-cuda-graph-fused-kernels.md) | CUDA Graph and Fused Kernels | 34 | CUDA graph capture, fused attention kernel, kernel launch overhead reduction |
| [025](adr/025-purego-cuda-bindings.md) | Purego CUDA Bindings | 34 | dlopen-based CUDA bindings, no CGo, cross-platform build |
| [026](adr/026-megakernel-decode.md) | Megakernel Decode | 34 | Single-kernel decode, codegen emitter, optable, nvcc JIT, slot-based memory |
| [027](adr/027-composition-prerequisite.md) | Composition Prerequisite | 34 | All layers must compose Engine primitives for tracing compiler |
| [028](adr/028-tracing-compiler.md) | Tracing Compiler | 34 | EngineProxy records primitive ops during Forward(), automatic decomposition |
| [029](adr/029-neon-simd-cpu-acceleration.md) | NEON SIMD CPU Acceleration | 34 | Plan9 assembly for hot-path ops, same-shape fast paths, tensor arena |

---

## 15. DGX Spark Hardware Validation (Phase 20)

The NVIDIA DGX Spark GB10 (Blackwell, sm_121, CUDA 13.0.2, ARM64 aarch64,
128GB unified LPDDR5X) validated the full GPU stack. All 66 packages pass with
`cuda,cutlass` build tags. See [ADR-017](adr/017-dgx-spark-hardware-validation.md).

### 15.1 ARM64 Build Fixes

Nine code fixes were required for aarch64 compatibility:

- Flash attention BLOCK_SIZE reduced 64 -> 32 (48KB shared memory limit on sm_121)
- TensorRT Makefile: auto-detect multiarch include path via `dpkg-architecture`
- TensorRT C shim: `kEXPLICIT_BATCH` -> 0, `setOptimizationProfileShared` -> `setOptimizationProfileAsync`
- Missing includes: `<cstdio>`, `<stdlib.h>` for C.free
- API renames: `tensor.NewTensorNumeric` -> `tensor.New`, `metrics.Collector` -> `runtime.Collector`
- Logger type safety: convert int/error args to string for `log.Logger`
- Test fixes: ARM64 float32 precision (TanhGrad), MemPool reuse, NCCL format strings
- Import cycle: remove `graph` import from `compute/gpu_integration_test.go`

### 15.2 Benchmark Results (NVIDIA DGX Spark GB10)

Hardware: 20-core ARM Cortex-A78AE, Blackwell GPU (sm_121), 128GB LPDDR5X.

#### MatMul (cuBLAS SGEMM) GPU vs CPU

| Size     | GPU (us) | CPU (us) | Speedup |
|----------|----------|----------|---------|
| 128x128  |       32 |      429 |   13.4x |
| 512x512  |      158 |    4,109 |   26.0x |
| 1024x1024|      509 |   23,393 |   45.9x |

#### Softmax GPU vs CPU (shape: 64x128x512)

| Engine | Latency (us) | Speedup |
|--------|-------------|---------|
| GPU    |       1,054 |   47.6x |
| CPU    |      51,516 |   1.0x  |

#### Flash Attention (CUTLASS, head_dim=64, num_heads=8)

| Seq Len | Latency (us) |
|---------|-------------|
| 128     |         147 |
| 512     |       1,035 |
| 1024    |       2,335 |
| 2048    |       8,924 |

#### Quantized GEMM (CUTLASS INT4/INT8)

| Kernel  | Size     | Latency (us) | GOPS     |
|---------|----------|-------------|----------|
| INT4    | 1024     |       3,958 |      545 |
| INT4    | 2048     |      31,998 |      537 |
| INT4    | 4096     |     426,040 |      322 |
| INT8    | 1024     |         941 |    2,289 |
| INT8    | 2048     |       7,933 |    2,166 |
| INT8    | 4096     |      75,380 |    1,822 |

#### TensorRT Inference

TensorRT 10.15.1 engine build, serialization, deserialization, and inference all
work on Blackwell. 15 tests pass in 5.6 seconds (including engine build time).

### 15.3 Model Parity Results (Phase 21)

Phase 21 downloaded, converted (ONNX -> ZMF via zonnx), and validated 7 model
families on DGX Spark. 18 ONNX-compatibility bugs were fixed during the process.
See [ADR-018](adr/018-model-parity-testing.md) for full details.

| Model | Tests | Status | Notes |
|-------|-------|--------|-------|
| Llama 3 (1B) | FP / GD / Gen | PASS | onnx-community/Llama-3.2-1B |
| Qwen 2.5 (0.5B) | FP / GD / Gen | PASS | Qwen/Qwen2.5-0.5B |
| Gemma 3 (1B) | FP / GD / Gen | PASS | google/gemma-3-1b-it (optimum export) |
| Mistral (7B) | FP / GD / Gen | PASS | mistralai/Mistral-7B-Instruct-v0.3 |
| Phi-3 (mini) | FP / GD / Gen | PASS | microsoft/Phi-3-mini-4k-instruct |
| DeepSeek V3 (671B) | — | SKIP | Exceeds 128GB DGX Spark memory |
| SigLIP (base) | FP | PASS | Fixed in Phase 22 (Squeeze scalar + Concat rank alignment) |
| FlashAttentionGQA | 1 | PASS | GQA kernel parity test |
| MultiGPU DualDevice | 1 | SKIP | Requires >= 2 CUDA devices |

**Summary:** 18 PASS, 4 SKIP. FP = ForwardPass, GD = GreedyDecode, Gen = Generation.

### 15.4 BF16 GEMM and Unified Memory (Phase 22)

Phase 22 added BFloat16 GPU compute, unified memory allocation, and fixed the
SigLIP Concat shape mismatch. See [ADR-019](adr/019-phase22-bf16-unified-siglip.md).

#### BF16 vs FP32 GEMM (cuBLAS)

| Size | BF16 (us) | FP32 (us) | Speedup |
|------|-----------|-----------|---------|
| 128  | 123       | 109       | 0.9x    |
| 512  | 212       | 349       | 1.6x    |
| 1024 | 412       | 631       | 1.5x    |
| 2048 | 1262      | --        | --      |

At 512+, BF16 is 1.5-1.6x faster due to doubled Blackwell tensor core throughput.

#### Unified Memory Allocation

| Size | cudaMalloc (us) | cudaMallocManaged (ns) | Speedup |
|------|-----------------|------------------------|---------|
| 1 MB | 132             | 600                    | 220x    |
| 16 MB | 702            | 658                    | 1067x   |
| 64 MB | 3370           | 668                    | 5045x   |

### 15.5 Test Coverage (Phases 23-24)

Phase 23 raised coverage from 8 packages at 100% to 9, and from 6 packages
below 90% to 2. 42 of 50 packages are at >= 95%. See docs/QUALITY.md.

Phase 24 fixed FFN bias detection heuristic, embedding layer loading in
LoadModelFromZMF, and refactored cmd/zerfoo-predict for testability (0% -> 76.6%).

### 15.6 Multi-GPU Test Coverage Gap

Six tests require >= 2 CUDA devices and skip on the single-GPU DGX Spark GB10.
See [ADR-017](adr/017-dgx-spark-hardware-validation.md) Section "Multi-GPU Test
Coverage Gap" for the full inventory and hardware/software prerequisites.

### 15.7 Performance Optimization (Phase 25)

Phase 25 built all core performance primitives for inference throughput:

| Epic | Result |
|------|--------|
| E25: Mmap loading | Zero-copy model load via `syscall.Mmap` |
| E26: Pre-alloc KV cache | 0 allocs/token in decode loop (ring buffer) |
| E27: Q4/Q8 tensor storage | 8x/4.5x compression vs float32 |
| E28: Q4/Q8 CPU MatMul | Fused dequant+SIMD multiply |
| E29: NEON/AVX2 SGEMM | ~2x faster than gonum BLAS (`internal/xblas/`) |
| E30: Parallel graph executor | Concurrent independent branch execution |
| E31: Continuous batching | Channel-based batch scheduler in `serve/` |
| E32: CUDA Q4 kernel | 2383 GFLOPS on DGX Spark GB10 (sm_121) |
| E33: Benchmark suite | tok/s, GFLOPS, memory allocs benchmarks |

Key files: `compute/pool.go`, `internal/xblas/sgemm_neon_arm64.s`,
`internal/xblas/sgemm_avx2_amd64.s`, `internal/cuda/kernels/q4_gemm.cu`.

### 15.8 End-to-End Inference Validation (Phase 26)

Phase 26 validated Q4 inference on real models and added serving primitives:

| Epic | Result |
|------|--------|
| E34: PagedAttention | Block pool + PagedKVCache (46% memory of pre-alloc) |
| E35: Speculative decoding | SpeculativeGenerator with adaptive draft length |
| E36: End-to-End Q4 pipeline | Gemma 3 2B Q4: 1.96 -> 3.60 tok/s (1.84x via blocked transpose) |
| E37: GGUF model import | Parser + loader + arch mapping (llama/gemma) |
| E38: Performance CI | bench.sh + GH Actions workflow + DGX GPU job |

Baseline performance: Gemma 3 2B Q4 at 3.60 tok/s on DGX Spark ARM64 CPU.
CPU profile showed Transpose at 62%, GEMM at 16%, GC/malloc at 5%.

Key files: `generate/paged_kv_cache.go`, `generate/speculative.go`,
`model/gguf/parser.go`, `model/gguf/loader.go`, `model/gguf/arch.go`.

### 15.9 Inference Throughput Optimization (Phase 27)

Phase 27 targeted CPU bottleneck elimination. Kernel-level work complete;
end-to-end benchmarks deferred to Phase 28 (require DGX Spark).

| Epic | Result |
|------|--------|
| E39: Transpose elimination | `FoldConstantTransposes` graph pass + blocked 4D transpose (35x faster) |
| E40: Tensor arena | `TensorPool` with ref-counted release in `graph.Forward` |
| E43: Operator fusion | Fused RMSNorm (single-pass), RoPE (single-pass), SiLU-gate |

Deferred from Phase 27:
- E41 (GPU inference pipeline) -- requires DGX Spark, deferred to Phase 29.
- E42 (GGUF end-to-end) -- superseded by Phase 28 E44 with refined scope.
- T39.3, T40.3, T43.4 (end-to-end benchmarks) -- carried forward to Phase 28 E48.

Key files: `graph/optimize.go` (FoldConstantTransposes), `compute/pool.go`
(TensorPool), `compute/fused_rmsnorm.go`, `compute/fused_rope.go`,
`compute/fused_silugate.go`.

Fused kernel details:
- **FusedRMSNorm**: `x * rsqrt(mean(x^2) + eps) * weight` in one pass. Returns
  per-row scales for backward compatibility. Gated on CPUEngine + float32.
- **FusedRoPE**: Single-pass rotary embedding. Supports full and partial rotation.
  Gated on CPUEngine + float32.
- **FusedSiLUGate**: `silu(gate) * up` in one pass. No engine gating (pure tensor op).

### 15.10 Make It Actually Work (Phase 28)

Phase 28 closed the gap between README promises and reality. All five epics
completed 2026-03-06.

| Epic | Result |
|------|--------|
| E44: GGUF End-to-End Inference | Graph builders for Llama and Gemma (shared transformer loop in `arch_common.go`). GGUF tokenizer extraction. Unified `LoadFile()`. |
| E45: Model Hub & Auto-Download | HF pull wired into default `LocalRegistry`. Model aliases for 6 models. `findGGUF()` auto-detection. |
| E46: Chat Template Engine | Per-architecture formatters (Gemma, LLaMA 3, Mistral, Qwen 2.5, DeepSeek, Phi-4). Auto-detection from GGUF `general.architecture`. |
| E47: K-Quant Dequantization | Q4_K (144B/256val), Q5_K (176B/256val), Q6_K (210B/256val). Wired into GGUF loader. |
| E48: Phase 27 Deferred Benchmarks | DGX Spark GB10: 3.80 tok/s (Gemma 3 2B Q4_0). Transpose folding: 8.8% runtime (was 62%). 15 tok/s NOT met. |

Key files:
- `inference/arch_llama.go`, `inference/arch_gemma.go`, `inference/arch_common.go` -- graph builders
- `inference/inference.go` -- Load(), LoadFile(), model aliases, chat formatters
- `inference/gguf.go` -- GGUF-to-Model metadata conversion, `chatTemplateForArch()`
- `tensor/quantized_kquant.go` -- K-quant dequantization (Q4_K, Q5_K, Q6_K)
- `model/gguf/loader.go` -- GGUF tensor loading with K-quant dispatch
- `registry/pull.go` -- HuggingFace download with progress, auth, .gguf filter

Model alias registry (`inference/inference.go`):
- `gemma-3-{1b,2b}-q4` -> google/gemma-3-*-it-qat-q4_0-gguf
- `llama-3-{1b,8b}-q4` -> meta-llama/Llama-3.*-Instruct-GGUF
- `mistral-7b-q4` -> mistralai/Mistral-7B-Instruct-v0.3-GGUF
- `qwen-2.5-7b-q4` -> Qwen/Qwen2.5-7B-Instruct-GGUF

Performance at Phase 28 end (DGX Spark GB10, Gemma 3 2B Q4_0, CPU ARM64):
- 3.80 tok/s (up from 3.60 in Phase 26, +5.6% from transpose folding)
- 79,537 allocs/token, 39.4 GB/op
- Profile: MatMul 19.6%, Runtime Transpose 8.8%, GC+memclr 6.9%, Element-wise 3.5%
- Bottleneck: Q4 dequantize-to-float32 before every MatMul dominates memory bandwidth
- TensorPool exists but NOT wired into graph forward loop

### 15.11 CPU Throughput Optimization (Phase 29)

Phase 29 targeted 4x CPU throughput (3.80 -> 15 tok/s). Achieved 6.5 tok/s
(1.7x improvement). Compute kernels are well-optimized; remaining bottleneck
is framework overhead (~150ms/token). See [ADR-020](adr/020-q4-quantized-dot-product.md).

| Epic | Result |
|------|--------|
| E49: NEON Q4 Dot Product | q4DotBlockSIMD (per-block) + q4DotRowSIMD (row-level) ARM64 assembly. GemmF32Q4NT for B-operand Q4 GEMV. Parallel GEMV across N dimension. |
| E50: TensorPool Wiring | Pool created in NewGenerator, attached via graph.WithPool(). Allocation benchmark added. |
| E51: KV Cache Decode | Already optimal (single-token Q/K/V projection, append+return). RoPE position offset bug fixed. |
| E52: NEON Fused Ops | Transpose layer data-pointer cache (3.53 -> 5.42 tok/s). Decode dim=1 short-circuit. T52.2/T52.3 (RMSNorm/SiLU NEON) deferred as low priority (<1% CPU each). |
| E53: DGX Benchmark | 6.5 tok/s (100-token avg). 15 tok/s NOT achieved -- framework overhead dominates. |

Key files:
- `internal/xblas/q4dot_arm64.s` -- NEON q4DotBlockSIMD + q4DotRowSIMD assembly
- `internal/xblas/q4dot_arm64.go`, `q4dot_generic.go`, `q4dot.go` -- Go declarations + scalar fallback
- `internal/xblas/gemm_quant.go` -- GemmF32Q4NT, GemmQ4F32Fused, parallel GEMV

Architecture decisions:
- **B-operand Q4 GEMV:** Model weights are always B operand. Transpose layer passes
  Q4 storage through with transposed shape. Engine detects Q4 on B, dispatches to
  GemmF32Q4NT which calls q4DotRowSIMD per output element.
- **Row-level assembly:** q4DotRowSIMD processes entire Q4 row in single call,
  eliminating 4 Go function calls per block x 248K blocks overhead.
- **FCVT encoding fix:** `0x1E22E0E7` was FCVT H7,S7 (wrong direction). Correct
  encoding for FCVT S7,H7 (half->single): `0x1EE240E7`.

Post-optimization profile (DGX Spark GB10, 30 tokens):

| Component | % CPU | Wall ms/token |
|-----------|-------|---------------|
| sgemmAccRowNeon (SGEMM) | 35.2% | ~12 |
| q4DotRowSIMD (Q4 GEMV) | 34.6% | ~12 |
| binaryOp (Mul/Add) | 6.7% | ~2 |
| Transpose | 3.9% | ~3 |
| GC/malloc | ~5% | ~15 |
| Other (graph, scheduling) | ~15% | ~130 |

**Bottleneck analysis:** Compute (GEMM) accounts for ~70% CPU but only ~24ms wall
time per token (parallelized). The remaining ~150ms/token is framework overhead:
graph traversal (~780 node executions/token with interface dispatch, shape validation,
pool acquire/release), goroutine scheduling (130 MatMul calls/token each spawning/
joining 20 goroutines), and memory management (GC despite TensorPool).

To reach 15 tok/s (67ms/token) requires architectural changes: graph compilation,
persistent worker pools, fused operation batching, zero-copy tensor views.

### 15.12 Graph Compilation and Worker Pool (Phase 30)

Phase 30 implemented graph compilation and a persistent worker pool to reduce
framework overhead. Achieved 6.86 tok/s (5% improvement over Phase 29's 6.5).
15 tok/s target NOT achieved -- root cause analysis revealed framework overhead
was only ~5% of total, not the estimated ~50ms. GEMV kernels dominate at 74%.
See [ADR-021](adr/021-graph-compilation-worker-pool.md).

| Epic | Result |
|------|--------|
| E54: Persistent Worker Pool | `internal/workerpool/pool.go`. Eliminated 2600 goroutine create/join per token. Wired into xblas GEMV and CPUEngine parallelFor. |
| E55: Graph Compiler | `graph/compile.go`. ExecutionPlan with pre-compiled instruction sequence. OpType-to-kernel mapping for all Gemma 3 operators. Wired into Generator decode loop via `GenerateStream`. |
| E56: Buffer Arena | Slot-based ExecutionPlan (deviated from pre-allocated arena). Slots store node.Forward() results directly, eliminating memo map + dependency map lookups. Zero-alloc goal deferred. |
| E57: Instruction Fusion | N/A. Gemma 3 graph uses high-level nodes (RMSNorm, Linear, GQA, SwiGLU) that already fuse element-wise ops internally. No bare fusible patterns exist. |
| E58: DGX Benchmark | 6.86 tok/s (100-token avg). Profile: sgemmAccRowNeon 39.86%, q4DotRowSIMD 33.65%, binaryOp 7.26%, ExecutionPlan.Run 9.66% cum. |

Key files:
- `internal/workerpool/pool.go` -- persistent worker pool with Submit/Close
- `graph/compile.go` -- Graph.Compile(), ExecutionPlan, Instruction types
- `generate/generator.go` -- compiled plan wired into GenerateStream decode loop

**Root cause analysis:** The original ~130ms framework overhead estimate was
incorrect. It was measured by subtracting single-thread kernel time from wall
time, not accounting for 20-core parallel execution. Actual framework overhead
is ~5% of total. Next performance improvement requires faster GEMV kernels
(NEON scheduling, cache blocking, memory layout), not framework changes.

Post-Phase 30 profile (DGX Spark GB10, 50 tokens):

| Component | % CPU | Notes |
|-----------|-------|-------|
| sgemmAccRowNeon (F32 GEMV) | 39.86% | Dominant compute kernel |
| q4DotRowSIMD (Q4 GEMV) | 33.65% | Second dominant kernel |
| binaryOp (element-wise) | 7.26% | Mul/Add/Sub |
| Transpose | 2.56% | Residual from non-folded cases |
| ExecutionPlan.Run | 9.66% cum | Compiled path active |
| Graph.Forward | 5.63% cum | Prefill + warmup only |

### 15.13 Inference Validation and Training Improvements (Phase 31)

Phase 31 validated GPU inference, closed PagedAttention gaps, added benchmark
regression detection, and delivered five training improvements for the Audacity
Numerai pipeline. PR #41 merged to main.

| Epic | Result |
|------|--------|
| E59: PagedAttention GQA v2 | Block-table KV reader in GQA (Option B). 8-seq load test: 0 alloc/op in paged path. Benchmark: block-table 40% fewer allocs vs gather-copy. |
| E62: GPU Q4 Profile | CPU 5.94 tok/s, GPU 5.12 tok/s (slower). Root cause: 43% cgocall overhead, only MatMul on GPU, all other ops fall back to CPU. |
| E63: CI Regression (partial) | Benchmark comparison script + regression detection workflow. DGX self-hosted runner NOT set up. |
| E64: Smoothed Early Stopping | Exponential smoothing (alpha=0.3) on val_corr. Patience based on smoothed metric. |
| E65: EMA Optimizer | EMA[T] wrapper with shadow params, SwapWeights for validation. decay=0.999 default. |
| E66: Cosine Warm Restarts | CosineWarmRestarts scheduler. T0/TMult/EtaMin/EtaMax. Wired into Audacity. |
| E67: SWA Optimizer | SWA[T] wrapper with epoch-boundary averaging. startEpoch gating. Wired into Audacity. |
| E68: Feature Dropout | FeatureDropout[T] layer. Column-wise inverted dropout. Registered in layer registry. Wired into Audacity model graph. |

Incomplete (carried to Phase 32 or deferred):
- E60: Speculative decoding validation (deferred -- needs draft model).
- E61: GGUF real-model inference (deferred -- needs HF downloads).
- E62 T62.2-T62.5: GPU fallback fixes (carried to Phase 32 as primary focus).
- E63 T63.2-T63.5: DGX runner, GPU CI (deferred).

Key files:
- `layers/attention/grouped_query_attention.go` -- BlockTableReader interface
- `serve/loadtest_test.go` -- 8-sequence concurrent load test
- `training/optimizer/ema.go` -- EMA[T] optimizer wrapper
- `training/optimizer/swa.go` -- SWA[T] optimizer wrapper
- `layers/regularization/feature_dropout.go` -- FeatureDropout[T] layer
- `audacity/internal/training/lr_schedule.go` -- CosineWarmRestarts
- `audacity/internal/training/early_stopping.go` -- smoothed tracking
- `.github/workflows/benchmark.yml` -- regression comparison step
- `cmd/bench-compare/` -- benchmark result comparison tool

GPU Profile (DGX Spark GB10, Gemma 3 2B Q4, -device cuda):

| Component | % Time | Notes |
|-----------|--------|-------|
| runtime.cgocall | 43% | CUDA kernel launches + H2D/D2H transfers |
| Q4 dequantize | 9.4% | CPU fallback |
| Transpose | 8.1% | CPU fallback |
| Binary ops | 4.4% | CPU fallback (broadcasting path) |
| MatMul (GPU) | ~30% | Only op actually on GPU |

Performance baselines at Phase 31 end:

| Model | Quant | Device | tok/s | Phase |
|-------|-------|--------|-------|-------|
| Gemma 3 2B | Q4_0 | CPU ARM64 | 6.86 | 30 |
| Gemma 3 2B | Q4_0 | CPU ARM64 | 5.94 | 31 (bench_tps) |
| Gemma 3 2B | Q4_0 | GPU (cuda) | 5.12 | 31 (bench_tps) |

Training improvements added in Phase 31 (Audacity pipeline):

| Feature | Config Flag | Default |
|---------|-------------|---------|
| EMA | --ema-decay | 0.999 |
| Smoothed early stopping | --smooth-alpha | 0.3 |
| Cosine warm restarts | --lr-schedule=cosine-warm-restarts, --lr-t0, --lr-tmult | T0=5, TMult=2 |
| SWA | --swa-start | disabled |
| Feature dropout | --feature-dropout | 0.0 |

### 15.14 GPU-First Inference Pipeline (Phase 32)

Phase 32 eliminated the major CPU fallbacks during GPU inference, improving
Gemma 3 2B Q4 throughput from 5.12 to 6.84 tok/s (+33.6%) on DGX Spark GB10.
GPU inference is now faster than CPU (6.61 tok/s).

Decision rationale: docs/adr/022-gpu-first-inference-pipeline.md.

Key implementations:
- GPU tensor residency: intermediate tensors stay GPU-resident via GPUStorage.
  `makeGPUResult`/`getDevicePtr` in compute/gpu_kernels.go.
- Q4 weight pre-upload: `GPUEngine.UploadWeights()` uploads Q4 blocks to GPU
  at model load time. `Q4Storage.SetGPUPtr/GPUPtr` caches device pointers.
  Only Q4 weights uploaded (float32 stays CPU to avoid D2H for non-GPU ops).
- `Graph.ConstantTensors()` collects all weight tensors from Parameter/Constant
  nodes for bulk upload. `parameterNode.Parameters()` returns nil (ONNX constants
  are not trainable), so `ConstantTensors()` calls Forward() on each node.
- GPU Transpose kernel: 2D tiled shared-memory + N-D stride-based in
  `internal/cuda/kernels/transpose.cu`. Wired into GPUEngine.Transpose.
- GPU Gather kernel: embedding lookup on GPU in `internal/cuda/kernels/gather.cu`.
- GPU broadcasting: stride-based 2D broadcasting for Add/Sub/Mul/Div in
  `compute/gpu_kernels.go gpuBroadcastOp`. `broadcastShape()` computes NumPy-style
  output shape preserving N-D leading dimensions.
- Fused GPU RMSNorm kernel: single-pass with shared-memory reduction in
  `internal/cuda/kernels/rmsnorm.cu`. `FusedRMSNormer` interface in compute/engine.go.

Bug fixes:
- N-D broadcast output shape: `flattenTo2D` lost leading dims (e.g. [1,1,1,1]+[2,1]
  gave [2,1] not [1,1,2,1]). Fixed with `broadcastShape()`.
- GPU Reshape zero-copy view for GPUStorage tensors.
- Nil axes handling in GPU Transpose (default to reverse permutation).

Performance at Phase 32 end:

| Model | Quant | Device | tok/s | Phase |
|-------|-------|--------|-------|-------|
| Gemma 3 2B | Q4_0 | CPU ARM64 | 6.61 | 32 (bench_tps) |
| Gemma 3 2B | Q4_0 | GPU (cuda) | 6.84 | 32 (bench_tps) |

Remaining GPU bottlenecks (pprof):
- cgocall: 58% (activation H2D/D2H from CPU fallback ops)
- Pow CPU fallback: 8.9% (no GPU scalar-broadcast Pow)
- binaryOp CPU fallback: 10.4% (unsupported broadcast patterns)
- GPUStorage.Slice D2H: 24% (CPU fallback ops reading GPU tensor data)

### 15.15 GPU Scalar Ops and D2H Elimination (Phase 33)

Phase 33 eliminated the three remaining CPU fallback bottlenecks identified in
Phase 32 profiling: Pow scalar-broadcast (8.9%), binary op scalar patterns (10.4%),
and D2H round-trips from GPU Split/Concat (24%).

Decision rationale: docs/adr/023-gpu-scalar-ops-d2h-elimination.md.

Key implementations:
- CUDA PowScalar kernel: `out[i] = powf(x[i], scalar)` in elementwise.cu.
  Wired into `gpuPow` to detect scalar-exponent pattern (totalElements==1).
- CUDA SubScalar kernel: `c[i] = a[i] - scalar` in elementwise.cu.
  Completes scalar-op coverage (Add/Sub/Mul/Div/PowScalar).
- Scalar-broadcast detection in `gpuBroadcastOp`: when one operand has
  totalElements==1, uses broadcast kernel with stride (0,0) for the scalar side.
- GPU Split/Concat using D2D memcpy: `gpuSplit` and `gpuConcat` in
  compute/gpu_kernels.go use `MemcpyAsync(D2D)` instead of custom CUDA kernels.
  Eliminates all D2H copies from Split/Concat CPU fallback.
- Float32 weight upload: `GPUEngine.UploadWeights` now uploads both Q4 and
  float32 weights to GPU (previously skipped float32 to avoid D2H from CPU
  fallback ops that no longer exist).
- `totalElements()` helper moved to compute/broadcast.go (no build tag).

Performance at Phase 33 end:

| Model | Quant | Device | tok/s | Phase |
|-------|-------|--------|-------|-------|
| Gemma 3 2B | Q4_0 | CPU ARM64 | 6.75 | 33 (bench_tps) |
| Gemma 3 2B | Q4_0 | GPU (cuda) | 10.32 peak / 7.78 median | 33 (bench_tps, 7 runs) |

High variance (7.45-10.32 across 7 runs) attributed to thermal throttling or
background processes on DGX Spark. Peak meets original 10 tok/s target.

Remaining bottlenecks for Phase 34:
- Per-op CGo kernel launch overhead (~100ns per call, 25+ kernels per forward pass)
- No CUDA graph support (each token = individual kernel launches through CGo)
- No fused kernels beyond RMSNorm (attention has separate Scale/Softmax/MatMul)
- All intermediates are float32 (BF16 GEMM exists but elementwise ops are F32)
- Op-by-op ExecutionPlan dispatch with no batching or fusion
- llama.cpp achieves 24-38 tok/s for similar models on same hardware

### 15.16 NEON SIMD CPU Acceleration (Phase 34 Track D)

Phase 34 Track D added ARM64 NEON SIMD assembly for all CPU hot-path operations
beyond matmul, same-shape fast paths to eliminate broadcasting overhead, and a
tensor arena for buffer reuse. Achieved 8.15 tok/s median (+18.8% over 6.86
baseline). Target was 10 tok/s; remaining gap requires GEMM cache tiling.

Decision rationale: docs/adr/029-neon-simd-cpu-acceleration.md.

#### Same-Shape Fast Paths (E101)

- Same-shape binaryOp fast path in `compute/cpu_engine.go:binaryOp()`: when both
  tensors have identical shapes, skips broadcast coordinate-decode loop and uses a
  direct element-wise loop with parallelFor. 7-8x speedup for same-shape ops.
  Commit f733d15.
- Pow x^2 specialization: detects scalar-broadcast exponent with value 2.0 and
  uses x*x instead of math.Pow(). 13-15x speedup for RMSNorm's Pow call.
  Commit c28a529.
- Scalar op baselines verified: MulScalar, AddScalar, DivScalar already use
  parallelFor. Commit 3d8c3d7.

#### NEON Assembly Kernels (E102)

All assembly follows the established pattern:
- `internal/xblas/<name>_arm64.go`: Go declarations with `//go:noescape`
- `internal/xblas/<name>_arm64.s`: ARM64 NEON plan9 assembly
- `internal/xblas/<name>_generic.go`: `//go:build !arm64` pure-Go fallback

Kernels implemented:
- **VexpF32** (`exp_arm64.s`): Degree-5 Horner polynomial with range reduction
  via x * (1/ln2). Max relative error < 2e-7. Shared by Softmax and SiLU.
  Commit 5931298.
- **SoftmaxF32** (`softmax_arm64.s`): 3-pass NEON: (1) FMAXP reduce for max,
  (2) exp(x-max) via VexpF32 + sum, (3) FMUL by reciprocal sum. Commit bc775d8.
- **RmsNormF32** (`rmsnorm_arm64.s`): FRSQRTE + 2 Newton-Raphson iterations
  for rsqrt(mean(x^2) + eps), then scale. Commit a40a2f7.
- **SiluF32/SiluGateF32** (`silu_arm64.s`): Inline exp polynomial + FRECPE for
  1/(1+exp(-x)) * x. SiluGate variant fuses gate multiplication. Commit d766923.
- **RopeF32** (`rope_arm64.s`): 4-wide SIMD rotation with scalar tail and
  passthrough for dimensions beyond rotary_dim. Commit ef099ef.
- **VaddF32/VmulF32/VsubF32/VdivF32** (`elementwise_arm64.s`): NEON 4-wide
  elementwise with scalar tail. Commit a40a2f7.
- **VmulScalarF32/VaddScalarF32/VdivScalarF32** (`scalar_arm64.s`): NEON
  broadcast-scalar ops. Commit c751b5c.

NEON wiring in CPUEngine (`compute/cpu_engine.go`): float32 type assertion
dispatches to NEON for tensors with >= 32 elements. Commits 7ac9a35, 0afe430.

#### Assembly Bug Fixes

Seven critical NEON assembly bugs discovered and fixed during integration:

| Bug | Root Cause | Fix |
|-----|-----------|-----|
| RoPE IP0/IP1 confusion | Temp register aliases collided with Go linkage | Used distinct temp registers |
| FMLS wrong encoding | WORD opcode had incorrect Rm field | Corrected bit encoding |
| Exp output clamping | Missing clamp for extreme negative inputs | Added FMAX with -87.3f |
| q4dot callee-saved | V10-V13 are callee-saved (D8-D15 lower) | Remapped to V24-V27 (commit e6d5f19) |
| RMSNorm lane zeroing | Tail elements not zeroed in partial vector | MOVI zero + masked insert |
| RMSNorm ABI return | Return value not in correct register | Fixed FMOV to S0 |
| Exp threshold | Threshold constant loaded incorrectly | Fixed constant pool entry |

ARM64 ABI note: registers D8-D15 (lower 64 bits of V8-V15) are callee-saved.
NEON assembly must preserve these or remap to V16-V31 (caller-saved scratch).

#### Tensor Arena (E103)

`compute/arena.go`: Power-of-2 bucketed pooling for buffer reuse. Each bucket
has a free list. Arena.Get(size) rounds up to next power of 2 and returns a
pooled buffer. Arena.Put(buf) returns it to the bucket. Wired into CPUEngine
via getOrCreateDest. Run with -race to detect use-after-free. Commit dc97cd7,
e3775a8.

#### Benchmark Results (E104)

| Config | tok/s | Phase | Notes |
|--------|-------|-------|-------|
| CPU ARM64 baseline | 6.86 | 30 | Before Track D |
| CPU ARM64 + Track D | 8.15 median (7.72-8.45) | 34 D4 | +18.8% |

CPU profile breakdown (post Track D):
- sgemmAccRowNeon: 37% (float32 GEMM, already NEON)
- q4DotRowSIMD: 35% (Q4 dot product, already NEON)
- Transpose: 4.4%
- Everything else: < 3% each

GEMM dominates at 72% of CPU time. The NEON hot-path ops (Softmax, RMSNorm,
SiLU, RoPE, elementwise) are now fast enough that they no longer appear in the
top profile entries. Further CPU gains require GEMM cache tiling (L2/L3 aware
blocking for sgemmAccRow and q4DotRow).

#### NEON Exp Polynomial Reference

The vectorized exp() uses range reduction with degree-5 polynomial:

```
Input: x (float32)
1. n = round(x * (1/ln2))        // FMUL + FCVTNS
2. r = x - n * ln2               // FMSUB (fused)
3. p = c0 + r*(c1 + r*(c2 + r*(c3 + r*(c4 + r*c5))))  // Horner's method
   c0=1.0, c1=1.0, c2=0.5, c3=1/6, c4=1/24, c5=1/120
4. result = ldexp(p, n)           // add n to float32 exponent bits:
                                  //   FCVTZS Vn.4S, Vn.4S (float->int)
                                  //   SHL Vn.4S, Vn.4S, #23 (shift to exponent)
                                  //   ADD Vp.4S, Vp.4S, Vn.4S (add exponent)
```

Max relative error: < 2e-7 for x in [-87.3, 88.7] (float32 range).

#### ARM64 NEON Instruction Encoding Cheat Sheet

Common instructions requiring WORD encoding in Go plan9 assembly:

| Instruction | Encoding | Description |
|-------------|----------|-------------|
| FMAXP Vd.4S, Vn.4S, Vm.4S | `6E20F400+...` | Pairwise max |
| FADDP Vd.4S, Vn.4S, Vm.4S | `6E20D400+...` | Pairwise add |
| FRSQRTE Vd.4S, Vn.4S | `6EA1D800+...` | Reciprocal sqrt estimate |
| FRSQRTS Vd.4S, Vn.4S, Vm.4S | `0EA0FC00+...` | RSqrt Newton step |
| FCVTNS Vd.4S, Vn.4S | `4E21A800+...` | Float to int nearest |
| FRINTX Vd.4S, Vn.4S | `6E219800+...` | Round to integral |
| FNEG Vd.4S, Vn.4S | `6EA0F800+...` | Negate |

Note: All register operand fields must be encoded correctly in the immediate.
Use `aarch64-linux-gnu-objdump -d` on a test .o to verify encodings.

#### Known Issues

- DGX Spark Go 1.25.0 linux/arm64 has intermittent segfaults (~10-40%) across
  ALL packages including pure Go packages with no assembly. Confirmed not caused
  by our code. System-level Go runtime issue.
- Pre-existing amd64 asm `vdotf32` missing Go declaration in
  `internal/xblas/gemm_simd_amd64.go`. golangci-lint rejects adding it because
  the function is unused on amd64. Not in Track D scope.
