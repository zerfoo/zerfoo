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
internal/cuda/        CUDA runtime purego bindings (dlopen libcudart.so)
internal/cublas/      cuBLAS purego bindings (dlopen libcublas.so)
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
- All GPU backends use purego (dlopen-based) bindings. No CGo or build tags.
  `go build ./...` compiles everywhere without `-tags cuda`, `-tags rocm`, or
  `-tags opencl`. Runtime detection via `*.Available()` functions.

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
  errors instead of logging. `SubSlice(offsetElems, length)` returns a
  zero-copy GPU view via pointer arithmetic -- no D2H copy for GPU-side slicing.

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

No CGo, no build tags, no C compiler required for building Zerfoo itself:

```
go build ./...
go test ./...
```

GPU acceleration is detected at runtime via `cuda.Available()`, `hip.Available()`,
and `opencl.Available()`. When a GPU library is not present, the framework
gracefully falls back to CPU.

**Runtime requirements** (only needed for GPU acceleration):
- CUDA Toolkit 12.x+ (libcudart.so, libcublas.so, libcudnn.so)
- NVIDIA GPU with Compute Capability >= 7.0 (Volta/Turing or newer)

**Compile CUDA kernels** (produces `libkernels.so` loaded via purego dlopen):

```
cd internal/cuda/kernels/
make shared                        # default: sm_75 (GCP T4)
make shared CUDA_ARCH=sm_121       # DGX Spark (Blackwell GB10)
make shared CUDA_ARCH=sm_70        # V100
```

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
| Transpose | 2D/3D/4D tensors | Custom CUDA kernel (shared memory tiling, precomputed strides) |
| Gather | Embedding lookup (int32 + int64 indices) | Custom CUDA kernel |
| Broadcasting | 4D element-wise Add, Sub, Mul, Div | Custom CUDA kernels (stride-based indexing) |
| Fused | QK RMSNorm+RoPE, SwiGLU, Scale+Softmax, post-FFN norm+add | Custom CUDA kernels |
| Fused | Dequant+GEMV Q4_K | Custom CUDA kernel (warp shuffle reduction) |
| Normalization | RMSNorm | Custom CUDA kernel (single-pass, shared-memory reduction) |

### 4.3 CPU Fallback Operations

These delegate to CPUEngine by design (not compute-bound or require Go runtime):

- UnaryOp (Go function pointers)
- Transpose (>4D tensors only; 2D/3D/4D have GPU kernels)
- Zero, Zeros, Copy
- Reshape, Split, Concat, Repeat
- ScatterAdd
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

### 4.5 CUDA Memory Management

#### Arena Allocator (internal/cuda/arena.go)

Primary allocator for inference. 2GB pre-allocated bump-pointer arena with
256-byte alignment. O(1) reset between tokens. During inference, 100% of
allocations are served by the arena (119K allocations, 0 fallback). Each
allocation is ~5ns (pointer bump) vs ~4us for cudaMalloc.

#### Memory Pool (internal/cuda/mempool.go)

Fallback allocator. Size-bucketed free-list for allocations that exceed the
arena or for permanent storage (model weights). Mutex-synchronized. Drained
on `GPUEngine.Close()`.

#### Managed Memory

Opt-in via `ZERFOO_ENABLE_MANAGED_MEM=1`. Uses `cudaMallocManaged` instead
of `cudaMalloc`. On NVLink-C2C hardware (DGX Spark GB10), managed memory
avoids explicit H2D copies and is 200-5000x faster to allocate (demand
paging). However, benchmarking on GB10 shows a ~13% throughput regression
(145 vs 165 tok/s) due to page fault overhead on first touch. Disabled by
default pending investigation of `cudaMemPrefetchAsync`.

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

`internal/cudnn/` provides purego bindings wrapping libcudnn via dlopen.
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

`internal/tensorrt/` provides purego bindings wrapping TensorRT's C API via
dlopen of `libtrt_capi.so`. The C shim in `cshim/trt_capi.h/cpp` provides a
flat C interface over TensorRT's C++ API, compiled into a shared library.

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
  (log-sum-exp trick), shared memory for K/V tiles, causal masking. BLOCK_SIZE=64
  for sm_121 (Blackwell), MAX_HEAD_DIM=128. Warp shuffle reductions for ScaledSoftmax.
- **Dispatch**: Single file `layers/attention/flash_cuda.go` with `cuda.Available()`
  runtime guard. No build tags. `ScaledDotProductAttention.Forward` calls
  `tryFlashForward` before the naive path when no arbitrary mask is provided.
- **Scope**: Float32 forward only. Backward pass deferred. Head dim > 128 or
  arbitrary masks fall back to naive attention.

See [ADR-010](adr/010-cutlass-flash-attention.md) for architecture decisions.

### 4.12 CUDA File Layout

All GPU files use purego (dlopen) bindings. No build tags. No CGo.

```
compute/
  gpu_engine.go            GPUEngine (GRAL interfaces, pool, stream, arena)
  gpu_cudnn.go             DNN-accelerated operations via GRAL
  gpu_kernels.go           getDevicePtr, makeGPUResult, kernel dispatch via GRAL

tensor/
  storage.go               Storage[T] interface, CPUStorage[T], NewWithStorage
  gpu_storage.go           GPUStorage[T] with gpuapi.Runtime, SubSlice for zero-copy views

internal/gpuapi/
  runtime.go               Runtime, Stream, MemcpyKind interfaces
  blas.go                  BLAS interface (Sgemm, GemmEx)
  dnn.go                   DNN interface (conv, batchnorm, activation, pooling, softmax)
  kernels.go               KernelRunner interface (element-wise, reduction, transpose, gather, broadcast)
  mempool.go               MemPool interface
  factory.go               Registration pattern for BLAS and DNN implementations
  cuda_runtime.go          CUDARuntime adapter (purego)
  cuda_blas.go             CUDABlas adapter (purego)
  cuda_dnn.go              CUDADNN adapter (purego)
  cuda_kernels.go          CUDAKernels adapter (purego)
  cuda_mempool.go          CUDAMemPool adapter (purego)

internal/cuda/
  runtime_purego.go        CUDA runtime purego bindings (dlopen libcudart.so)
  arena.go                 2GB bump-pointer arena allocator
  mempool.go               Size-bucketed device memory pool (fallback)
  kernels/
    elementwise.cu         CUDA kernel source (element-wise, scalar, broadcast)
    flash_attention.cu     Tiled flash attention kernel (online softmax, BLOCK_SIZE=64)
    transpose.cu           N-D GPU transpose (shared memory tiling, precomputed strides)
    gather.cu              GPU embedding lookup (int32 + int64 indices)
    rmsnorm.cu             Fused RMSNorm (single-pass, shared-memory reduction)
    gemv_q4k.cu            Fused dequant+GEMV for Q4_K (warp shuffle reduction)
    kernels_purego.go      Purego dispatch wrappers (dlopen libkernels.so)
    Makefile               nvcc compilation (produces libkernels.so)

internal/cublas/
  cublas_purego.go         cuBLAS purego bindings (dlopen libcublas.so)

internal/cudnn/
  cudnn_purego.go          cuDNN purego bindings (dlopen libcudnn.so)

internal/tensorrt/
  tensorrt_purego.go       TensorRT purego bindings (dlopen libtrt_capi.so)
  cshim/                   C shim for TensorRT C++ API

inference/
  tensorrt_convert.go      Graph-to-TRT converter (runtime Available() guard)
  tensorrt_cache.go        TRT engine caching
  tensorrt_pipeline.go     TRT inference engine wrapper

graph/
  cuda_graph.go            CUDAGraphExecutor (warmup/capture/replay)

layers/attention/
  flash_cuda.go            Flash attention dispatch (runtime cuda.Available() guard)
```

Purego dlopen targets:

```
libcudart.so              CUDA runtime (memory, streams, graphs)
libcublas.so              cuBLAS (SGEMM, GemmEx)
libcudnn.so               cuDNN (conv, batchnorm, activation, pooling, softmax)
libtrt_capi.so            TensorRT C shim
libkernels.so             Custom CUDA kernels
```

### 4.13 GPU Runtime Abstraction Layer (GRAL)

`internal/gpuapi/` defines vendor-neutral interfaces that decouple `compute/`
and `tensor/` from any specific GPU SDK. GPUEngine stores five GRAL interfaces
(`Runtime`, `BLAS`, `DNN`, `KernelRunner`, `MemPool`) instead of vendor-specific
handles. GPUStorage stores a `Runtime` for memory operations.

CUDA adapters in the same package implement these interfaces by delegating to
`internal/cuda`, `internal/cublas`, and `internal/cudnn` (all purego). ROCm
adapters delegate to `internal/hip`, `internal/rocblas`, and `internal/miopen`
(all purego). OpenCL adapters delegate to `internal/opencl` and
`internal/clblast` (all purego). Adding a new backend requires implementing
the five interfaces -- no changes to compute/ or tensor/ are needed.

The DNN interface abstracts at the operation level: callers pass shapes as
`[4]int` arrays and the adapter manages vendor-specific descriptors internally.
See [ADR-011](adr/011-gpu-runtime-abstraction-layer.md) for details.

### 4.14 AMD ROCm Backend

ROCmEngine mirrors GPUEngine's architecture using HIP/rocBLAS/MIOpen adapters,
all via purego dlopen. No build tags. Runtime detection via `hip.Available()`.

Purego dlopen targets:
- `libamdhip64.so` -- HIP runtime
- `librocblas.so` -- rocBLAS GEMM
- `libMIOpen.so` -- MIOpen DNN operations
- `libhipkernels.so` -- Custom HIP kernels

Integration: `device/rocm_device.go` auto-registers AMD GPUs.
`inference/engine_rocm.go` routes "rocm" / "rocm:N" to ROCmEngine.
`layers/attention/flash_rocm.go` dispatches fused attention on AMD GPUs.
See [ADR-012](adr/012-amd-rocm-backend.md) for details.

### 4.15 OpenCL Backend

OpenCLEngine mirrors GPUEngine's architecture using OpenCL/CLBlast adapters,
all via purego dlopen. No build tags. Runtime detection via `opencl.Available()`.

Purego dlopen targets:
- `libOpenCL.so` -- OpenCL runtime
- `libclblast.so` -- CLBlast GEMM

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
| DGX Spark GB10 | Blackwell | sm_121 | 128 GB unified LPDDR5x | Local (ARM64) |

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
- Large-dimension MatMul GPU tests (Llama 3 128K vocab, Gemma 3 262K vocab) with CPU parity.
- Range op edge case tests (16 cases: zero delta, wrong inputs, descending, empty range).
- Multi-model graph forward tests (large LM head, 2-layer transformer, diamond graph).
- CLI pull command tests (16 cases: error paths, nil registry, cached output).
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
# 1. Compile CUDA kernels (shared library for purego dlopen)
cd internal/cuda/kernels && make shared CUDA_ARCH=sm_121

# 2. Run GPU test suite (no build tags needed)
go test -count=1 -v \
    ./compute/ ./tensor/ ./internal/cuda/... ./internal/cublas/... ./device/

# 3. Run parity tests (GPU vs CPU)
go test -run Parity -v ./compute/
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

**"CUDA not found"**: `cuda.Available()` returns false. Ensure libcudart.so is
in the library search path (LD_LIBRARY_PATH or /usr/local/cuda/lib64). No build
tags or compiler needed -- only the runtime shared libraries.

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

1. float32 only for GPU element-wise ops -- BF16 GEMM via cuBLAS GemmEx, other types fall back to CPU.
2. GPU broadcasting supports up to 4D -- >4D cases fall back to CPU.
3. Single GPU -- no multi-GPU or distributed GPU support.
4. cuDNN operations (Conv2d, BatchNorm, activations, pooling, softmax) are non-interface methods on GPUEngine -- layers must call them explicitly rather than through Engine[T].
5. CUDA graph capture enabled for decode (184/185 ops captured). Only EmbeddingLookup excluded.
6. Managed memory disabled by default (ZERFOO_ENABLE_MANAGED_MEM) -- 13% regression on GB10 from page faults.
7. float16/float8 GEMM upcasts to float32 -- no native half-precision element-wise kernels.
8. Generics wiring hardcodes float32 -- registry, worker node, CLI all use float32.
9. KV cache is optional -- not all graph architectures support it.
10. Fused dequant+GEMV Q4_K kernel ready but engine dispatch validation pending.
11. Performance: 234 tok/s F32 with CUDA graph (18.7% faster than Ollama 197.21 tok/s). Phase 6 complete.
12. cuBLAS status 7 FIXED (Phase 10): Root cause was stale GPU tensor caching in Transpose/MatMul layers. Arena ResetPool freed GPU memory, but cached tensors held nil devicePtrs. Fix: removed caching, use MatMulTransposeB (SgemmNT). ZMF models now complete inference but CUDA graph capture fails (H2D copies during capture, error 901) and non-graph fallback produces garbage output.
13. CUDA graph capture for ZMF codegen models FIXED (Phase 11): Multiple D2H sync operations on the legacy stream during capture caused cuda error 901. Fixes applied:
    - PreUploadFrozenWeights: uploads frozen CPU tensors to GPU before capture (69c48af).
    - KV cache snapshot/restore: prevents double KV cache update on capture failure (425e0c6).
    - Scalar constants kept CPU-resident: prevents Pow/Range D2H during capture (ce1e155).
    - Transpose isGPU early exit removed: CPUStorage tensors use GPU kernel path (e5d4f38).
    - Gather, Slice, Reshape, AutoAttentionMask, AutoPositionIds added to nonCapturableOps.
    - Longest contiguous capturable region scan: handles non-capturable ops scattered in the instruction list (replaces edge-trimming logic).
    - EnsureCaptureInputsGPU: uploads frozen scalar constants needed by capture-region instructions.
    Result: ZMF codegen pipeline (fused ops) captures 99.5% of instructions, 232.86 tok/s (+26% vs no-graph).
    ONNX models capture only 1-2% due to decomposed ops (Pow, ReduceMean, Sqrt, Gather, Slice, Reshape).
    ONNX models previously produced garbage output ("!!!") -- root causes fixed:
    - CUDA powf() NaN for negative bases: kernel_pow_scalar used powf(negative, 2.0) which returns NaN. Fixed with a*a for s==2, powf(fabsf(a), s) otherwise.
    - KV cache never accumulated: zeroKVCacheNode returned empty cache every call. Fixed with StatefulInputNode interface + kvCacheIONode that feeds present KV outputs back as past KV inputs.
    - Position IDs stuck at 0: positionIdsNode always generated [0..seqLen-1]. Fixed with offset counter tracking decode step.
    - Attention mask: maskFromInputNode now tracks accumulated sequence length.
    - Greater/Where ops: added N-D broadcasting for causal mask computation.
    Llama 3 ONNX previously produced coherent text with stale .so; with rebuilt .so all ONNX models hit correctness errors.
    Phase 12 fixes: Cast aliasing (Cast.Forward returns new tensor wrapper with View refcount), Gather index clamping for embedding OOB.
    Remaining per-model issues after Phase 12: Qwen 2.5 (poor output quality), Mistral 7B (Or shape mismatch at node[98]),
    Phi 4 (Add size mismatch at node[125]), Llama 3 (MatMul 1D vs 2D at node[106] with rebuilt .so).
    Phase 13 investigation confirmed broadcastShape is correct (NumPy rules preserved). Real issues fixed:
    - Or op: added N-D broadcasting via validatedBroadcast (same pattern as Greater/Where). Fixes Mistral attention mask.
    - gpuBroadcastOp flattenTo2D collapse: when two N-D shapes flatten to identical (M,D) but broadcast to larger output,
      the 2D kernel allocated wrong-size output. Fixed with element-count mismatch guard that falls back to 4D kernel.
    - 42+ broadcast coverage tests added (broadcastShape, broadcastStrides4D, flattenTo2D, trailingDimsMatch, CPU/GPU parity).
    Remaining: ONNX output quality is a float32 precision accumulation issue (not a bug). First 3 tokens match ORT reference
    exactly. Divergence at token 4 due to ~0.001/layer attention score drift compounding through 16 layers.
    Contributing factors: Cos/Sin/ScatterND/Expand ops force CPU/GPU bouncing, decomposed RMSNorm has different reduction
    order vs fused kernel. Repetition in output (fox fox fox) is expected for small models at temp=0 without repetition penalty.
17. Decode kernel flash_attention_decode disabled for GQA models (Phase 10): kernel exceeds time budget at kv_len>=256 (118% of budget). cuBLAS SDPA achieves 233 tok/s vs kernel's 114 tok/s. Dead fast path code removed (-138 lines). Kernel retained in .cu for future optimization.
18. Purego trampoline assembly correct (Phase 10): ARM64 AAPCS64 trampoline verified on DGX. Segfault was in arena managed memory tests (device-only pointer access from CPU). Fixed with IsManaged() guards.
19. FP16 KV cache verified (Phase 10): identical output to F32, +11.2% throughput (138 vs 124 tok/s on DGX).
20. Phase 10 Wave 2 test coverage: getDevicePtr lifecycle tests (compute/gpu_kernels_test.go), CLI pull integration tests with httptest mock HF server (cmd/cli/pull_test.go), debug GPU logging (ZERFOO_DEBUG_GPU=1).
21. EngineProxy.MatMulTransposeB fallback handles both 2D [1,0] and 3D [0,2,1] axes (Phase 10 fix).
13. Serve layer hardened (Phase 9 Wave 1): structured request logging, /metrics endpoint, panic recovery with 503 for OOM.
14. OpenAI API integration tests added (71 tests, serve/integration_test.go with //go:build integration).
15. Lint debt resolved: errcheck, unused, ineffassign issues fixed. CI lint step now strict (no || true).
16. Phase 10 Wave 1 test coverage: large MatMul GPU tests (compute/gpu_engine_matmul_test.go), Range op edge cases (layers/core/range_op_test.go), graph forward tests (graph/forward_test.go), CLI pull tests expanded (cmd/cli/pull_test.go). CI strict lint verified and 3 additional lint fixes applied.

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
- `POST /v1/embeddings` -- embedding generation
- `GET /v1/models` -- model listing
- `GET /v1/models/:id` -- model detail
- `DELETE /v1/models/:id` -- model unloading
- `GET /openapi.yaml` -- OpenAPI 3.1 specification (go:embed)

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
| [030](adr/030-ollama-performance-parity.md) | Ollama Performance Parity | 34 | Performance strategy for matching/surpassing Ollama throughput |
| [031](adr/031-openai-server-in-zerfoo.md) | OpenAI Server in Zerfoo | 34 | Server stays in Zerfoo serve/ package, not in Zonnx |

---

## 15. DGX Spark Hardware Validation (Phase 20)

The NVIDIA DGX Spark GB10 (Blackwell, sm_121, CUDA 13.0.2, ARM64 aarch64,
128GB unified LPDDR5X) validated the full GPU stack. All 66 packages pass with
`cuda,cutlass` build tags. See [ADR-017](adr/017-dgx-spark-hardware-validation.md).

### 15.1 ARM64 Build Fixes

Nine code fixes were required for aarch64 compatibility:

- Flash attention BLOCK_SIZE: initially reduced 64 -> 32 for sm_121, later restored to 64 with optimized shared memory usage
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

---

## ADR-025 Implementation Complete (2026-03-12)

Runtime GPU detection via purego dlopen fully implemented. Key outcomes:

### Build Tags Removed
- internal/codegen/: runner.go and runner_stub.go merged (commit a64d831)
- internal/cuda/kernels/: 5 CGo kernel files deleted, purego wrappers are sole
  implementation (commit d9375fb). flash_attention.go remains behind cuda&&cutlass.
- compute/: gpu_engine.go, gpu_kernels.go, gpu_cudnn.go, gpu_fused_rmsnorm.go
  all use cuda.Available() runtime guard (commits eb7e77e, cd31b73)
- inference/: engine_cuda.go and engine_nocuda.go unified into engine.go with
  cuda.Available() (commit 3bbea76). TensorRT files unchanged (behind cuda tag).
- tensor/: gpu_storage.go, gpu_storage_default_cuda.go, transfer.go use
  cuda.Available() (commit 44c68ba)

### BLAS/DNN Factory Pattern
- gpuapi/factory.go: Registration pattern for BLAS and DNN implementations.
- cuda_blas.go and cuda_dnn.go register via init() (stay behind cuda tag).
- GPUEngine handles nil BLAS/DNN gracefully. MatMul falls back to CPU when BLAS nil.

### 16 Op Emitters Added to codegen.CheckSupport
AutoPositionIds, AutoZeroKVCache, Shape, Unsqueeze, Cast, Equal, Where,
ConstantOfShape, Expand, Range, Cos, Sin, Greater, Trilu, Max, ScatterND.
Commits: 4bc6e9a, 51ea41d.

### Kernel Fixes
- PowScalar SIGSEGV: AAPCS64 float-in-integer-register mismatch. Changed C
  launcher signatures from float to unsigned int, reinterpret via memcpy
  (commits d787888, e20604c).
- gatherOp panic: non-traced Compile gives Gather only 1 InputIdx. Added bounds
  check, megakernel falls back to per-op (commit 22d269b).

### Verification Results
- go build ./... passes without -tags cuda on macOS and DGX Spark (linux/arm64).
- bench_tps: 10 tokens at 0.44 tok/s on DGX Spark (megakernel falls back to per-op).
- CPU plan.Run(): 5.71 tok/s. CUDA plan.Run(): 2.22 tok/s.
- Both produce degenerate output (pre-existing inference correctness issue).

### Known Issues Post ADR-025 (Resolved)
- internal/cublas/ and internal/cudnn/ converted to purego (Waves 1-8).
- All 6 GPU backends (cuBLAS, cuDNN, TensorRT, CUTLASS/flash attention,
  ROCm/HIP/rocBLAS/MIOpen, OpenCL) are now purego. `go build ./...` works
  everywhere without any build tags.
- Megakernel abandoned in favor of CUDA graph + fused kernels (see below).

---

## Ollama Performance Parity Achieved (2026-03-12)

Zerfoo inference on DGX Spark GB10 reaches 188.92 tok/s average (3 runs),
exceeding the 95% target of 187.35 tok/s. Ollama baseline: 197.21 tok/s.
Current performance: 95.8% of Ollama. Model: Gemma 3 1B Q4_K_M GGUF.

### Inference Correctness Fix
- Root cause: weight loading order and RoPE frequency computation mismatch.
- Fix verified: 50 tokens match Ollama output at temperature=0.

### Performance Optimizations Applied
| Optimization | Commit | Impact |
|-------------|--------|--------|
| Arena allocator (2GB bump-pointer, O(1) reset) | 33b0dee | 8.61 -> 80.35 tok/s |
| Pre-allocated KV cache buffers | 7e80e21 | Eliminates malloc per token |
| GQA KV head broadcast (eliminates ~192MB Repeat) | e92a04a | Reduces memory traffic |
| MatMulTransposeB via cuBLAS SgemmNT | 74cac33, bb5e5fd | Saves 18 Transpose/token |
| cublasSgemmStridedBatched (1 call vs 8 per attn) | 2bbbeb1 | Reduces launch overhead |
| Fused QK RMSNorm+RoPE kernel | 42f4008 | Saves 78 kernel launches/token |
| Zero-copy Q+K view (avoids Concat) | 27bf4d3 | Saves 26 kernel launches/token |
| Fused post-FFN RMSNorm+residual Add | 6b22b47 | Saves 26 launches/token |
| Fused SwiGLU kernel | c3835ad | Eliminates 5 ops per FFN layer |
| Fused Scale+Softmax (shared-mem reductions) | (integrated) | Part of SDPA path |
| NVCC -O3 --use_fast_math | d1ed26a | Negligible (bandwidth-bound) |

### Performance Progression
| Phase | tok/s |
|-------|-------|
| Initial GPU Q4 | 8.61 |
| Arena allocator | 80.35 |
| Previous session best | 177.49 |
| Fused QK norm+RoPE | 183.23 |
| Zero-copy Q+K view | 186.54 |
| Fused norm+add | 189.78 |
| 5-run average (session 1) | 188.01 |
| 3-run average (session 2) | 188.92 |

### CUDA Graph Infrastructure (Built, Disabled)
- Runtime API wrappers: StreamBeginCapture, StreamEndCapture, GraphInstantiate,
  GraphLaunch, GraphDestroy, GraphExecDestroy (internal/cuda/runtime_purego.go).
- StreamProvider interface on GPUEngine (compute/engine.go).
- CUDAGraphExecutor with 3-phase warmup/capture/replay (graph/cuda_graph.go).
- Graceful fallback on capture failure.
- Disabled because 3 D2H copy sites in the forward pass conflict with stream capture:
  1. GPUEngine.Gather reads indices.Data() for int64->int32 conversion.
  2. GPUStorage.TrySlice in GQA CPU fallback paths.
  3. tensor_cache appendGPU CPU fallback.

### OpenAI-Compatible Inference Server
- Package: serve/ with server.go.
- Endpoints: POST /v1/chat/completions, POST /v1/completions,
  POST /v1/embeddings, GET /v1/models, GET /v1/models/:id,
  DELETE /v1/models/:id, GET /openapi.yaml.
- OpenAPI 3.1 specification embedded via go:embed and served at /openapi.yaml.
- Usage token counting (prompt_tokens, completion_tokens) on all responses.
- SSE streaming, batch scheduling, speculative decoding support.
- Wired into CLI via cmd/cli/serve.go.
- See ADR-031 for architecture decision (server lives in Zerfoo, not Zonnx).

---

## Megakernel Investigation and Abandonment (2026-03-13)

The megakernel approach (single CUDA kernel executing the entire decode step)
was investigated and abandoned in favor of CUDA graph + fused kernels.

### Failure Modes

1. **CompileTraced validation failure**: The megakernel design requires
   CompileTraced to decompose composite ops (GQA, FFN, SwiGLU) into primitives.
   CompileTraced fails with "input tensors cannot be nil" at instruction 0
   (MatMul) due to frozen slot tensor lifecycle issues. Fallback to Compile
   produces composite op names that have no emitters.

2. **7 composite ops with no emitters**: Without CompileTraced, the standard
   Compile path produces EmbeddingLookup, GroupedQueryAttention, FFN, LMHead,
   and others that codegen.CheckSupport rejects. Writing CUDA device functions
   for each would duplicate the existing fused kernel infrastructure.

3. **Single-thread execution model**: The emitted megakernel uses a single
   `tid` thread model. Different ops require different parallelism (MatMul
   needs 2D thread blocks, reductions need shared memory, etc.). A single
   thread configuration cannot serve all ops efficiently.

4. **No cuBLAS integration**: The megakernel cannot call cuBLAS from device
   code. MatMul would need a hand-written GEMM, which cannot match cuBLAS
   performance.

5. **No Q4_K support**: The codegen emitter has no Q4_K dequantization path.

### Decision

CUDA graph capture + fused kernels is strictly superior:
- Captures existing optimized kernels (cuBLAS, fused QK norm+RoPE, SwiGLU,
  fused dequant+GEMV) as-is with zero code duplication.
- Each op runs with its own optimal grid/block dimensions.
- Near-zero launch overhead (~15us for graph replay vs ~2.37ms for 338
  individual launches).

Code in generate/megakernel.go, internal/codegen/ is retained but not invested in.

---

## CUDA Graph Partial Capture (2026-03-13)

### Architecture

`graph/cuda_graph.go` implements CUDAGraphExecutor with 3-phase execution:

1. **Warmup** (token 1): Normal per-op execution to validate correctness.
2. **Capture** (token 2): `cudaStreamBeginCapture` records all GPU operations
   into a CUDA graph. `cudaGraphInstantiate` creates a replayable executable.
3. **Replay** (tokens 3+): `cudaGraphLaunch` replays the captured graph in a
   single launch, eliminating per-op launch overhead.

The executor splits the execution plan into capturable and non-capturable
regions. EmbeddingLookup runs outside the graph (data-dependent control flow).

Pre-allocated fixed buffer layout (ExecutionPlan) provides fixed memory
addresses required by CUDA graph replay.

### Current Status

Disabled by default. Opt-in via `ZERFOO_ENABLE_CUDA_GRAPH=1`.

Remaining D2H copies in the transformer body prevent full capture:
- GQA forward pass: KV cache management triggers D2H
- FFN: intermediate tensor reads
- KV cache append: source tensor Data() calls

When capture fails, CUDAGraphExecutor falls back gracefully to per-op
execution. Relaxed capture mode (`cudaStreamCaptureModeRelaxed`) is used to
allow synchronous memcpy during capture, but this is insufficient for the
remaining D2H sites.

### Estimated Impact

Eliminating 338 kernel launches/token at ~7us each = ~2.37ms overhead.
Graph replay reduces this to ~15us total. Estimated +20-30 tok/s.

---

## Performance Summary (2026-03-13)

### Current Performance

| Config | tok/s | Notes |
|--------|-------|-------|
| Zerfoo GB10 (clean defaults) | 166 | 84% of Ollama |
| Zerfoo GB10 (managed mem) | 145 | 13% regression from page faults |
| Zerfoo GB10 (previous best) | 188.92 | Before Wave 1-8 code changes |
| Ollama GB10 | 197.21 | Baseline target |
| Theoretical max (Q4 on GB10) | ~350-400 | 273 GB/s bandwidth ceiling |

Output quality is coherent (verified against Ollama at temperature=0).

### Path to Surpassing Ollama

1. Enable CUDA graph capture (eliminate remaining D2H copies) -- +20-30 tok/s estimated
2. Investigate 188->166 tok/s regression from Wave 1-8 code changes
3. Kernel optimization: register tuning, shared memory for sm_121

---

## Phase 2 Completion Summary (2026-03-13)

All 35 Phase 2 tasks across 6 epics completed:

### E401: Regression Bisect -- Complete
- Root cause: managed memory detection (T303.1) caused 12% regression.
- Fix: disabled by default (ZERFOO_ENABLE_MANAGED_MEM=1 to opt in).

### E402: CUDA Graph D2H Elimination -- Complete (partial)
- All 5 D2H sites addressed. GPU paths taken during inference.
- CUDA graph capture still fails due to GQA D2H in conditional fallback.
- Graph replay not faster because capture does not succeed.

### E403: Q4_K Validation -- Complete
- Fused GEMV kernel validated. Q4_K preservation re-enabled.
- Q4_K path matches Q4_0 in throughput.

### E404: Kernel Benchmarks -- Complete
- Wave 8 optimizations verified on DGX. Register tuning confirmed.

### E405: BF16/FP16 Inference Path -- Complete
- FP16 element-wise kernels added (elementwise_fp16.cu).
- cublasGemmEx mixed-precision MatMul wired.
- Full FP16 inference path works but 17% slower than F32 (124.50 vs 149.52).
- Root cause: F32->FP16 conversion round-trips on every operation.

### E406: FP8 Inference Path -- Complete (acceptance partially met)
- FP8E4M3Storage, cublasLt purego wrappers, FP8 MatMul all implemented.
- FP8 inference runs end-to-end but 100x slower than F32 (1.45 tok/s).
- Root cause: arena thrashing (1841 misses, 5GB GPU memory for 1B model).
- FP8 output is degenerate (repetitive text). Scale factor propagation suspect.

### Key Architecture Insights from Phase 2

1. **FP16 conversion overhead:** Current FP16 path converts F32->FP16 and back
   on every operation (Add, Mul, RMSNorm, Softmax, MatMul). Each binary op
   triggers 6 kernel launches (2 F32->FP16, 1 compute, 1 FP16->F32, 2 alloc/free).
   Solution: store weights and activations natively in FP16 throughout inference.

2. **FP8 arena exhaustion:** 2GB pre-allocated arena cannot hold all FP8
   intermediate buffers. Falls back to MemPool for 1841+ allocations per forward
   pass. Each FP8 MatMul allocates FP16 conversion buffers + scales + output.
   Solution: pre-allocate FP8-specific buffers, enlarge arena, or use persistent
   FP16 activation buffers.

3. **FP8 scale propagation:** FP8 output is degenerate, suggesting scale factors
   are not correctly propagated through the compute graph. Per-tensor absmax
   scaling may be insufficient for small models.

4. **GQA D2H copies:** Two fallback paths in grouped_query_attention.go (lines
   436-453 and 889-909) trigger D2H copies when GPU storage type assertion fails.
   Both have GPU fast paths via SubSlice, but the assertion sometimes fails.

### Phase 3 Completion Summary (2026-03-13)

Phase 3 (26 tasks, 6 epics: E501-E506) is complete. Key accomplishments:

1. **Native FP16 activation storage (E502):** Float16Storage type, native FP16 paths
   for element-wise ops, MatMul, RMSNorm, Softmax. Gather output FP16 conversion
   as single entry point. LMHead FP16->F32 conversion for sampling.

2. **FP16 weight pre-conversion (E503):** Attempted but reverted. Converting F32
   weights (norm gains, embedding table) to Float16Storage in UploadWeights caused
   garbage output due to corruption in FP16->F32 round-trip conversions consumed
   by downstream operations. Fix: F32 weights stay as GPUStorage[float32]; per-op
   FP16 compute paths handle F32->FP16 on the fly. Commit efdd87b.

3. **FP8 arena pre-allocation (E504):** fp8Scratchpad with grow-only reusable
   buffers for A/B matrices. Reduces some arena pressure but 1841 arena misses
   persist because output buffers and scale pointers are not covered.

4. **FP8 scale factor fix (E505):** cublasLt FP8 requires sm_89+. GB10 (sm_121)
   supports it but the check was wrong. Added FP16 dequant fallback path:
   DequantFP8E4M3ToFP16 + MixedFP16Gemm. Works on any GPU with FP16.

5. **FP16 is slower than F32 for Q4K models.** Q4K GEMV always produces F32
   output. FP16 activations add per-op F32<->FP16 conversion overhead with no
   compute benefit. For Q4K models, F32 is the optimal activation type.

6. **FP8 remains broken.** 1.48 tok/s with degenerate output. Arena thrashing
   (1841 misses, 5GB GPU memory) not fully resolved by scratchpad.

### Key Architecture Insights from Phase 3

1. **FP16 activations add overhead for Q4K models.** Since Q4K GEMV produces F32
   output and all weight matrices are Q4K-quantized, FP16 activations mean every
   downstream op round-trips F32<->FP16. FP16 only helps with unquantized dense
   weight matrices.

2. **F32 weight storage is correct for norm weights.** Norm gains are tiny
   (model_dim elements). Converting them to FP16 saves negligible memory but
   introduced a garbage output bug that was difficult to diagnose. Per-op
   F32->FP16 conversion in the compute path is safe and simple.

3. **Managed memory arena regression is fixed.** Arena at internal/cuda/arena.go:64
   now gates managed memory behind ZERFOO_ENABLE_MANAGED_MEM env var. Weight
   uploads also gated (gpu_engine.go:148). Default is cudaMalloc (no page faults).

4. **Q4K GEMV kernel characteristics:** 4 warps x 32 = 128 threads/block. Input
   vector loaded to shared memory. Fused dequant in registers. Warp shuffle
   reduction. Uses __ldg for quantized byte loads. Block size 128 may be
   suboptimal for high occupancy -- llama.cpp uses 256.

5. **GQA D2H copies block CUDA graph capture.** Two fallback paths in
   grouped_query_attention.go trigger .Data() D2H copies when tensor storage
   type assertions fail. These must be eliminated before CUDA graph capture.

### Updated Performance (2026-03-13, gemma3 1B Q4_K_M on DGX Spark GB10)

| Config | tok/s | Notes |
|--------|-------|-------|
| Zerfoo F32 (gemma3) | 157.25 | Best path for Q4K models |
| Zerfoo FP16 (gemma3) | 127.23 | Correct output, slower due to per-op overhead |
| Zerfoo FP8 (gemma3) | 1.48 | Arena thrashing, degenerate output |
| Ollama (gemma3) | 197.21 | Target (25% gap from F32) |

### Phase 4 Completion Summary (2026-03-13)

Phase 4 executed 30 tasks across 6 epics in 5 parallel waves (up to 5 agents).
Closed the Ollama gap from 25% to 3% (157.25 -> 191.28 tok/s).

**E601: Q4K GEMV Kernel Optimization -- Reverted.**
Profiled Q4K GEMV: down_proj (K=6144) dominates at 51.3 us/call, 33% occupancy,
28% bandwidth. Shared-memory-limited (24KB/block). Attempted block size 256 +
vectorized uint4 loads + x-vector tiling. Register pressure increased 43->54,
causing 12.2% regression (189->166 tok/s). Reverted to original kernel (128
threads, 43 registers). Lesson: for memory-bound kernels on sm_121, register
pressure from vectorization can eliminate occupancy gains from reduced smem.

**E602: GQA D2H Elimination -- Complete.**
Audited 15 .Data() calls. Only 2 in decode hot path (fused QK and splitMergedQKV).
Both had GPU fast paths that are always taken during GPU decode. Replaced CPU
fallbacks with hard errors. Verified zero D2H copies during decode on DGX.

**E603: CUDA Graph Capture -- Infrastructure built, GQA blocks capture.**
Built full CUDA graph capture/instantiate/replay infrastructure with arena reset
floor, captured slot restore, async KV cache D2D memcpy, Pow scalar async D2H.
GQA is position-dependent (RoPE angles computed on CPU, KV cache offset from CPU).
Graph replay bakes in capture-time kernel args, producing wrong output for
subsequent tokens. GQA appears at instruction 2 interleaved in every layer, so no
contiguous capturable region exists. Graceful fallback to RunInstructions.
To fix: pre-compute RoPE table for all positions on GPU, store KV offset in GPU
memory. This is a significant refactor deferred to Phase 5+.

**E604: FP8 Arena and Output Fix -- Fixed (partially).**
Added grow-only output buffer to fp8Scratchpad (ensureC). Arena misses: 1841->4.
Root cause of degenerate output: (1) fp8Scratchpad cached stale arena pointers
after arena.Reset(); (2) embed_tokens/lm_head were FP8-quantized but used for
gather, corrupting vocabulary mapping. Fixed both. FP8 CUDA: 53.70 tok/s with
coherent output on CPU path. However, cublasLt FP8 is unsupported on sm_121
(status 15), so all FP8 ops use dequant+FP16 fallback, which introduces quality
degradation. FP8 quality on sm_121 remains degenerate (R606 materialized).

**E605: Per-Token Overhead Reduction -- Complete.**
Pre-allocate [1,1] token tensor, update in-place per decode step (both Generate
and GenerateStream). Eliminates per-token tensor creation. Verified identical
output and stable arena stats on DGX.

**Key finding: FP16/FP8 dispatch overhead was the dominant regression.**
The entire 157->189 tok/s recovery came from adding `e.dtype != DTypeF32` guards
to skip ~600 failed Float16Storage/FP8E4M3Storage/BFloat16Storage type assertions
per token on the F32 hot path. This was not in the plan -- discovered during
pre-wave investigation.

### Updated Performance (2026-03-13 post-Phase 4)

| Config | tok/s | Notes |
|--------|-------|-------|
| Zerfoo F32 (gemma3) | 191.28 | 3-run avg (190.57, 191.41, 191.85) |
| Zerfoo FP8 (gemma3) | 53.70 | Fixed but degenerate on sm_121 |
| Ollama (gemma3) | 197.21 | Gap: 3.0% (5.93 tok/s) |

### Phase 5 Completion Summary (2026-03-13)

Phase 5 executed 19 tasks across 6 epics to investigate Go-side optimizations.
All 5 optimization tracks (PGO, GC, BCE, purego, thread pinning) were audited.

**Key finding: The remaining 3.6% gap is entirely in CUDA kernel execution.**
Go runtime is not the bottleneck:
- GC: Zero pauses during decode (arena allocator handles GPU memory).
- BCE: Only 8 hot-path bounds checks out of 928 total (<0.1% overhead).
- Purego FFI: ~395 calls/token, ~20us total (<0.4% of token time). All function
  pointers cached at init via dlsym + sync.Once.
- PGO: No measurable improvement (hot path is in CUDA kernels, not Go code).
- LockOSThread: Caused 2.6% regression (190->185 tok/s). On GB10 with unified
  LPDDR5x, CUDA context migration is negligible but LockOSThread prevents Go
  scheduler from using the pinned thread for other work. Reverted.

### Updated Performance (2026-03-13 post-Phase 5)

| Config | tok/s | Notes |
|--------|-------|-------|
| Zerfoo F32 (gemma3) | ~190 | Stable across PGO/no-PGO |
| Ollama (gemma3) | 197.21 | Gap: 3.6% (7 tok/s) |

### Path to Surpassing Ollama (Phase 6)

The only viable path is making GQA CUDA-graph-compatible. Current blockers:
1. cache.SeqLen() read from CPU (line 393 in grouped_query_attention.go)
2. RoPE angle offset computed on CPU (line 395, calls GetAngles with posOffset)
3. KV cache append at position-dependent offset (kvcache.go line 138, lb.cursor*dim)

Fix approach: GPU-resident position counter + pre-computed RoPE table indexing.
The RoPE table is already GPU-resident; only the offset selection is CPU-side.
CUDA graph infrastructure (capture/replay, arena floor, slot restore) is built.

### Phase 6 Completion Summary (2026-03-14)

Phase 6 executed 20 tasks across 4 epics to enable CUDA graph capture for the
full decode loop. See docs/adr/032-gpu-resident-position-counter.md for the
design decision and docs/adr/033-how-we-beat-ollama.md for the full journey.

**Key changes:**
- 3 new CUDA kernels: increment_counter, offset_memcpy, rope_select.
- GPU-resident int32 position counter in TensorCache and GPUKVCache.
- GQA decode path uses GPU counter for RoPE and KV append (no CPU SeqLen).
- GQA removed from non-capturable ops list. 184/185 instructions captured.
- Bug fix: GPU counter must be synced from CPU after prefill (H2D copy of
  token count before decode starts).

**New kernel files:**
- internal/cuda/kernels/counter.cu -- atomicAdd increment + reset
- internal/cuda/kernels/offset_memcpy.cu -- counter-indexed memcpy
- internal/cuda/kernels/rope_select.cu -- counter-indexed RoPE table lookup

**Key file changes:**
- generate/tensor_cache.go -- gpuCounter field, AppendGPU, SyncCounterFromGPU
- generate/gpu_kv_cache.go -- gpuCounter field, AppendGPU, SyncCounterFromGPU
- layers/embeddings/rotary_positional_embedding.go -- GetAnglesGPU method
- layers/attention/grouped_query_attention.go -- GPU RoPE path in fused + unfused
- graph/cuda_graph.go -- GQA removed from nonCapturableOps
- generate/generator.go, generate/stream.go -- SyncCounterFromGPU after decode

**Known issue:** Graph and no-graph paths produce different (but both coherent
and deterministic) output at temp=0. Likely floating-point ordering difference
from captured vs individual kernel launches.

### Updated Performance (2026-03-14 post-Phase 6)

| Config | tok/s | Notes |
|--------|-------|-------|
| Zerfoo F32 + CUDA graph | 234.30 | 3-run avg: 235.09, 234.42, 233.39 |
| Zerfoo F32 no graph | ~186 | Baseline without graph capture |
| Ollama (gemma3) | 197.21 | Surpassed by 18.8% |

Bandwidth utilization: ~60% of 273 GB/s theoretical. Token time: 4.27ms.
Theoretical max at full bandwidth: ~390 tok/s. Remaining headroom: ~40%.

### Phase 7 Completion Summary (2026-03-14)

Phase 7 investigated three optimization paths. >300 tok/s target not met but
valuable infrastructure and findings delivered. Released as v1.1.0.

**Key findings:**
- cuBLAS is only 8% of decode time (T901.1). Weight matmuls use fused Q4K GEMV.
  Custom SGEMV integration deferred (low ROI).
- GQA-aware flash_attention_decode kernel (E905): functionally correct but 2x
  slower than cuBLAS SDPA path (114 vs 234 tok/s). Disabled for GQA models.
  The kernel needs Blackwell-specific optimizations (TMA, async copy) to compete.
- FP16 KV cache (E902): working with --kv-dtype flag. Correctness bug fixed.
- Graph/no-graph divergence (E903): fixed via GPU-resident kv_len counter.
  Decode-specific flash_attention kernel reads KV length from GPU memory.
- Performance regression tests added: operation count assertions, decode path
  selection tests, benchmark gates.

**New files (Phase 7):**
- internal/cuda/kernels/sgemv_m1.cu -- custom F32 GEMV for M=1 (152 GFLOPS)
- internal/gpuapi/cuda_blas_profile.go -- cuBLAS profiling wrapper
- generate/decode_ops_test.go -- operation count regression tests
- generate/bench_decode_test.go -- decode throughput benchmarks
- layers/attention/gqa_decode_test.go -- GQA decode path selection tests
- docs/adr/034-gqa-aware-flash-attention-decode.md -- GQA kernel decision

**Performance unchanged:** 234.08 tok/s (F32 KV, CUDA graph, GQA via SDPA).

### Known Technical Debt

1. GQA decode fast path disabled (114 tok/s regression). Kernel needs optimization
   or different approach (Blackwell TMA, speculative decoding).
2. purego assembly trampoline segfaults without -race on Go 1.25/arm64 (DGX).
   Tests pass with -race flag. Affects: internal/cuda/kernels/ tests on DGX.
3. 276 pre-existing golangci-lint issues. CI uses --new-from-rev || true to
   avoid blocking on legacy code.
