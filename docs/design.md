# Zerfoo Design Document

## 1. Overview

Zerfoo is a Go 1.25 generics-based machine learning framework designed to be
the best ML framework in the world through superior design, performance, and
ease of use. It supports CPU, CUDA, ROCm, and OpenCL GPU execution through a
unified compute engine interface. All layers, models, and training loops
delegate computation to the Engine[T] interface, enabling transparent hardware
acceleration without modifying application code.

### 1.1 What Zerfoo Is

A general-purpose ML framework with three production-ready capabilities:

1. **Inference engine.** Load and run open-weights transformer models (text,
   vision, vision-language) via the ZMF model format. Models are converted
   from ONNX or GGUF using companion tools (zonnx, native GGUF loader).
   Supports autoregressive decoding with KV cache, sampling (temperature,
   top-k, top-p, repetition penalty), and streaming.

2. **OpenAI-compatible API server.** Full chat completions, completions,
   embeddings, and model management endpoints with SSE streaming. Drop-in
   replacement for OpenAI API clients.

3. **Training framework.** Generic Trainer[T] with AdamW/SGD optimizers,
   MSE/CrossEntropy loss functions, backpropagation through the computation
   graph, and distributed gradient exchange via gRPC.

### 1.2 Design Principles

- **No CGo, no build tags.** `go build ./...` compiles everywhere. GPU
  acceleration is runtime-detected via purego (dlopen). CUDA kernels are
  compiled separately into a shared library.
- **Generics throughout.** Type-safe parametric polymorphism via Go 1.25
  generics. Three type constraints (Numeric, Float, Addable) govern the
  entire type system.
- **Engine abstraction is law.** Layers never access tensor data directly.
  All arithmetic goes through Engine[T]. This enables transparent CPU/GPU
  switching and makes every op testable on CPU.
- **Architectural boundaries.** `zerfoo/` must not import `zonnx/` or
  `onnx/`. The ZMF format is the decoupling boundary between model
  conversion and model execution.

### 1.3 Validated Model Families

| Family | Format | Architecture | Status |
|--------|--------|-------------|--------|
| Gemma 3 | GGUF Q4_K | Text decoder | Production (CUDA graph, highest throughput) |
| Llama 3 | ZMF/ONNX | Text decoder | Working (ONNX decomposed path) |
| Qwen 2.5 | ZMF/ONNX | Text decoder | Working (ONNX decomposed path) |
| Mistral 7B | ZMF/ONNX | Text decoder | Working (ONNX decomposed path) |
| Phi-3/4 | ZMF/ONNX | Text decoder | Working (ONNX decomposed path) |
| DeepSeek V3 | ZMF/ONNX | MoE text decoder | Config parser present, untested (too large) |
| SigLIP | ZMF | Vision encoder | Parity test PASS |
| Kimi-VL | ZMF | Vision-language connector | Parity test PASS |

See docs/benchmarks.md for current throughput numbers per model.

### 1.4 Two Execution Paths

Zerfoo has two distinct execution paths for inference, each with different
performance characteristics:

**ZMF Codegen Path (fused ops).** Used by GGUF models and hand-tuned ZMF
models. The inference layer constructs a graph using fused operations
(GroupedQueryAttention, FusedAddRMSNorm, FFN) that map directly to optimized
CUDA kernels. Achieves near-complete CUDA graph capture and highest throughput.

**ONNX Decomposed Path (individual ops).** Used by models converted from ONNX
via zonnx. The ONNX standard decomposes composite operations into individual
ops (Pow, ReduceMean, Sqrt, Div, Mul for RMSNorm; Reshape, Gather, MatMul for
attention). Each op is a separate instruction with its own kernel launch.
CUDA graph capture is limited because non-capturable ops (Reshape, Gather,
Shape) are scattered throughout the instruction list. A graph fusion pass
(graph/fusion.go) detects decomposed patterns and replaces them with fused
instructions to bridge the performance gap.

See docs/benchmarks.md for current throughput numbers per execution path.

### 1.5 Maturity Levels

| Capability | Maturity | Notes |
|-----------|----------|-------|
| Inference (GGUF) | Production | CUDA graph capture, Q4_K quantization |
| Inference (ONNX) | Beta | All models run, output quality improving |
| OpenAI API server | Production | Full spec compliance, 71 integration tests |
| Training | Implemented | All tests pass, no end-to-end workflow documented |
| Distributed training | Implemented | gRPC + NCCL strategies, not production-tested |
| Multi-GPU | Architecture ready | NCCL bindings present, single GPU validated |
| ROCm/OpenCL backends | Implemented | Purego bindings, not hardware-validated |

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
256-byte alignment. O(1) reset between tokens. During inference, all
allocations are served by the arena with zero fallback. Allocation cost is
a pointer bump (nanoseconds) vs cudaMalloc (microseconds).

#### Memory Pool (internal/cuda/mempool.go)

Fallback allocator. Size-bucketed free-list for allocations that exceed the
arena or for permanent storage (model weights). Mutex-synchronized. Drained
on `GPUEngine.Close()`.

#### Managed Memory

Opt-in via `ZERFOO_ENABLE_MANAGED_MEM=1`. Uses `cudaMallocManaged` instead
of `cudaMalloc`. On NVLink-C2C hardware (DGX Spark GB10), managed memory
avoids explicit H2D copies and is faster to allocate (demand paging).
However, benchmarking shows a throughput regression due to page fault
overhead on first touch. Disabled by default pending investigation of
`cudaMemPrefetchAsync`. See docs/benchmarks.md for specific numbers.

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
  Multiple ONNX compatibility fixes applied. See [ADR-018](adr/018-model-parity-testing.md).

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

### 10.1 Compute Engine

1. **GPU element-wise ops are float32 only.** BF16 GEMM works via cuBLAS
   GemmEx, but element-wise kernels (Add, Mul, Exp, etc.) only support
   float32. Other types fall back to CPUEngine.
2. **GPU broadcasting supports up to 4D.** Tensors with >4 dimensions fall
   back to CPU for broadcast operations.
3. **cuDNN ops are non-interface methods.** Conv2d, BatchNorm, activations,
   pooling, and softmax are methods on GPUEngine, not part of Engine[T].
   Layers must call them explicitly.
4. **Generics hardcoded to float32.** The registry, worker node, CLI, and
   inference pipeline all instantiate float32. Other numeric types work at
   the library level but are not wired through the application layer.

### 10.2 GPU and Memory

5. **Single GPU only.** Multi-GPU inference is not validated. NCCL bindings
   exist for distributed training but have not been tested on multi-GPU
   hardware.
6. **Managed memory disabled by default.** cudaMallocManaged causes 13%
   throughput regression on GB10 due to page faults. Opt-in via
   ZERFOO_ENABLE_MANAGED_MEM=1.
7. **Flash attention decode kernel disabled.** Exceeds time budget at longer
   sequence lengths. cuBLAS SDPA is faster for GQA models. The kernel is
   retained in .cu for future optimization.

### 10.3 ONNX Execution Path

8. **ONNX CUDA graph capture is limited.** Decomposed ops (Pow, ReduceMean,
   Gather, Slice) break the contiguous capture region. Static Reshape ops
   (1 input, target shape from attributes) are capture-safe and no longer
   break the region; only dynamic Reshape (2+ inputs reading shape from a
   tensor) is non-capturable. The `isNonCapturable()` function in
   graph/cuda_graph.go determines capturability per-instruction. The ZMF
   codegen path achieves near-complete capture. See docs/benchmarks.md.
9. **RMSNorm fusion not yet runtime-correct.** Pattern matching works but
   the fused Forward function produces numerically wrong results due to
   input slot resolution. See docs/devlog.md for investigation status.
10. **ONNX output diverges from ORT after initial tokens.** Float32 precision
    accumulation drift compounds through transformer layers. This is inherent
    to float32 with different GEMM accumulation orders, not a bug.

### 10.4 Training

11. **Training infrastructure is implemented but not production-tested.**
    Trainer[T], optimizers (AdamW, SGD), loss functions (MSE, CrossEntropy),
    and distributed gradient exchange all pass unit tests, but no end-to-end
    training workflow is documented or validated.

### 10.5 Backends

12. **ROCm and OpenCL backends are implemented but not hardware-validated.**
    Purego bindings for HIP/rocBLAS/MIOpen and OpenCL/CLBlast exist and
    compile, but have not been tested on actual AMD or Intel GPUs.

---

## 11. Ecosystem

### 11.1 Companion Repositories

- **zmf** (`github.com/zerfoo/zmf`): Zerfoo Model Format protobuf library.
- **zonnx** (`github.com/zerfoo/zonnx`): ONNX-to-ZMF converter with per-operator builders.
- **float16** (`github.com/zerfoo/float16`): IEEE 754 float16 and bfloat16 types for Go.
- **float8** (`github.com/zerfoo/float8`): E4M3 float8 type for Go.
- **gemma3** (`github.com/zerfoo/gemma3`): Gemma 3 model support and conversion scripts.

### 11.2 Inference Pipeline

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

### 11.3 GGUF Inference Pipeline

GGUF models are loaded via `inference/load_gguf.go` using the `model/gguf`
package. The pipeline extracts `gguf.ModelConfig` from GGUF metadata and
selects an architecture builder based on `cfg.Architecture`:

- `buildLlamaGraph` (`inference/arch_llama.go`) -- Llama family
- `buildGemmaGraph` (`inference/arch_gemma.go`) -- Gemma family

Both delegate to `buildTransformerGraph` (`inference/arch_common.go`), which
constructs the decoder-only transformer graph. Architecture-specific
differences are parameterized via `transformerGraphOpts`:

- `embedScale` -- multiply embeddings by a constant (Gemma: sqrt(hidden_size))
- `headDim` -- override computed head dimension (needed when head_dim != hidden_size/num_heads)
- `rmsNormEps` -- RMSNorm epsilon (default 1e-5)
- `activation` -- FFN activation function ("swiglu" or "gelu_pytorch_tanh")
- `attnScale` -- custom attention scaling factor

Key nodes defined in `inference/arch_llama.go`:
- `embeddingLookupNode` -- token ID to embedding lookup with optional scaling
- `lmHeadNode` -- hidden state to vocabulary logit projection

Both implement `graph.EmbeddedFrozenProvider` so the graph compiler registers
their embedded weights as frozen slots for megakernel compilation.

See [ADR-035](adr/035-gemma3-architecture-parameterization.md) for architecture
parameterization decisions.

### 11.4 Data Flow

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

## 12. Multi-Architecture Support

The inference pipeline supports multiple model architectures. Each architecture
has distinct attention, normalization, and
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

Device affinity on all CUDA-aware components and NCCL-based collective
operations enable distributed GPU training.

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
| [030](adr/030-ollama-performance-parity.md) | Inference Performance Strategy | 34 | Performance optimization strategy for competitive throughput |
| [031](adr/031-openai-server-in-zerfoo.md) | OpenAI Server in Zerfoo | 34 | Server stays in Zerfoo serve/ package, not in Zonnx |
| [034](adr/034-gqa-aware-flash-attention-decode.md) | GQA-Aware Flash Attention Decode | 34 | Grouped-query attention in decode kernel |

