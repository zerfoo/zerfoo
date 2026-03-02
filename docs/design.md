# Zerfoo Design Document

## 1. Overview

Zerfoo is a Go-based machine learning framework with 40+ packages. It supports
CPU and CUDA GPU execution through a unified compute engine interface. All
layers, models, and training loops delegate computation to the Engine interface,
enabling transparent hardware acceleration without modifying application code.

---

## 2. Core Architecture

### 2.1 Package Layout

```
compute/              Engine interface + CPUEngine + GPUEngine
tensor/               TensorNumeric[T], Storage[T] abstraction
graph/                Computation graph, Node interface, Builder, Parameter
layers/               18 sub-packages: activations, attention, core, normalization, etc.
training/             TrainingWorkflow, optimizers (SGD, Adam), loss functions
model/                ModelProvider, ModelInstance, serialization, ZMF format
numeric/              Type-specific arithmetic ops (float32/64, float8, float16, int8, uint8)
device/               Device, Allocator interfaces (CPU + CUDA)
distributed/          gRPC-based distributed training (All-Reduce, Barrier, Broadcast)
metrics/              Evaluation metrics (Pearson, Spearman, MSE, RMSE, MAE)
cmd/                  CLI tools (predict, tokenize, train)
data/                 Data loading and processing
internal/xblas/       CPU BLAS wrappers (gonum)
internal/cuda/        CUDA runtime CGO bindings (//go:build cuda)
internal/cublas/      cuBLAS CGO bindings (//go:build cuda)
internal/cuda/kernels CUDA kernel source and Go wrappers
testing/testutils/    Test assertion helpers, MockEngine, custom mocks
tests/                Integration tests: parity, numerics, helpers
types/                Shared type definitions (BackwardMode, etc.)
pkg/tokenizer/        Tokenizer implementation
pkg/prelude/          Common type re-exports
```

### 2.2 Dependency Graph

```
layers/* --> graph.Node[T] --> compute.Engine[T] --> tensor.TensorNumeric[T]
                                    |                        |
                               numeric.Arithmetic[T]    Storage[T]
                                                        /        \
                                                CPUStorage[T]  GPUStorage[T]
                                                                    |
                                                              internal/cuda
```

Key invariant: layers never access tensor data directly for computation.
All arithmetic goes through Engine[T]. This enables transparent CPU/GPU switching.

### 2.3 Architectural Boundaries

- `zerfoo/` must not import `zonnx/` or `onnx/` (verified by `make verify-architecture`).
- `zonnx/` must not import `github.com/zerfoo/zerfoo` (decoupled via `zmf` format).
- CUDA code is gated behind `//go:build cuda`. Non-CUDA builds work without any GPU dependencies.

---

## 3. Key Interfaces

### 3.1 Engine[T] (compute/engine.go)

The compute engine is the central abstraction. It has 34 methods covering
all tensor operations. Every layer receives an Engine at construction time
and delegates all computation to it.

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
- `GPUEngine[T]`: CUDA kernels for float32. CPU fallback for other types.
  20 methods have native GPU implementations; 14 use CPU fallback by design.

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

### 3.3 Storage[T] (tensor/storage.go)

```go
type Storage[T Numeric] interface {
    Len() int
    Slice() []T         // CPU: zero-copy. GPU: copies D2H.
    Set(data []T)       // CPU: direct assign. GPU: copies H2D.
    DeviceType() device.Type
}
```

**CPUStorage[T]:** Wraps a Go `[]T` slice. `Slice()` returns the underlying
slice directly (zero copy). `DeviceType()` returns `device.CPU`.

**GPUStorage[T]:** Wraps a CUDA device pointer (`unsafe.Pointer`). `Slice()`
allocates a host slice and copies D2H. `Set()` copies H2D. Has additional
`TrySlice()`/`TrySet()` methods that return errors instead of logging.
`Ptr()` returns the device pointer for kernel dispatch.

### 3.4 Distributed Training (distributed/interfaces.go)

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

---

## 4. GPU Engine Architecture

### 4.1 Build Requirements

- CUDA Toolkit (libcudart, development headers)
- cuBLAS library (libcublas)
- NVIDIA GPU with Compute Capability >= 7.0 (Volta/Turing or newer)
- GCC/G++ (for CGO linking)

Build: `go build -tags cuda ./...`
Test: `go test -tags cuda ./...`

### 4.2 GPU-Accelerated Operations (float32 only)

| Category | Operations | Backend |
|----------|-----------|---------|
| Matrix | MatMul (2D and batched) | cuBLAS Sgemm |
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
- `getDevicePtr`: Checks storage type. GPUStorage returns device pointer
  directly (zero-copy). CPUStorage allocates from memory pool and copies H2D.
- `makeGPUResult`: Creates output tensors with GPUStorage wrapping device pointer.

### 4.5 CUDA Memory Pool

`internal/cuda/mempool.go`: Size-bucketed free-list allocator. Reuses previously
freed device memory, avoiding per-operation `cudaMalloc`/`cudaFree`. The pool
is mutex-synchronized and drained on `GPUEngine.Close()`.

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
- Therefore: call cublasSgemm with B as first argument, A as second, swapping m and n
- This avoids explicit transposition and works for any matrix dimensions.

### 4.9 CUDA File Layout

```
compute/
  gpu_engine.go            GPUEngine (pool, stream, cuBLAS) (//go:build cuda)
  gpu_kernels.go           getDevicePtr, makeGPUResult, kernel dispatch (//go:build cuda)
  gpu_engine_test.go       Unit tests (//go:build cuda)
  gpu_integration_test.go  Integration + chained-ops tests (//go:build cuda)

tensor/
  storage.go               Storage[T] interface, CPUStorage[T], NewWithStorage
  gpu_storage.go           GPUStorage[T], TrySlice/TrySet (//go:build cuda)
  transfer.go              ToGPU/ToCPU helpers (//go:build cuda)

device/
  cuda_device.go           CUDA device abstraction (//go:build cuda)
  cuda_allocator.go        CUDA memory allocator (//go:build cuda)

internal/cuda/
  runtime.go               CUDA runtime + Stream bindings (//go:build cuda)
  mempool.go               Size-bucketed device memory pool (//go:build cuda)
  kernels/
    elementwise.cu         CUDA kernel source (17 kernels, stream-aware)
    elementwise.go         CGO bindings for kernels (//go:build cuda)
    Makefile               nvcc compilation

internal/cublas/
  cublas.go                cuBLAS + SetStream bindings (//go:build cuda)
```

CGO linker flags:
```
internal/cuda/runtime.go:     -lcudart
internal/cublas/cublas.go:    -lcublas
internal/cuda/kernels/*.go:   -L${SRCDIR} -lkernels -lcudart -lstdc++
```

### 4.10 Parity Tolerances

- MatMul: 1e-5 relative error
- Element-wise ops: 1e-6 relative error
- Reductions (Sum, Mean): 1e-5 relative error

---

## 5. Testing Architecture

### 5.1 Test Coverage

Target: >= 95% statement coverage for all testable packages.

Documented exceptions (unreachable `tensor.New` error paths):
- layers/gather: 93.1%
- layers/embeddings: 93.5%
- layers/features: 93.8%
- testing/testutils: 94.5%

### 5.2 Testing Patterns

- Table-driven tests using standard `testing` package (no testify).
- Parity tests comparing GPU vs CPU output for every GPU-accelerated method.
- Integration tests for cross-package workflows.
- Numerical gradient checking via finite differences.
- MockEngine for unit testing layers in isolation.

### 5.3 Excluded from Coverage Target

| Package | Reason |
|---------|--------|
| distributed/pb/ | Generated protobuf code |
| cmd/zerfoo/ | Main entrypoint, no testable logic |
| cmd/zerfoo-predict/ | Main entrypoint; logic in cmd/cli/ |
| cmd/zerfoo-tokenize/ | Main entrypoint; logic in pkg/tokenizer/ |
| pkg/prelude/ | 1 line, no statements |
| types/ | Type definitions only |

---

## 6. Build and CI

### 6.1 Dependencies

Direct (go.mod):
- gonum.org/v1/gonum (BLAS)
- google.golang.org/grpc + protobuf (distributed training)
- github.com/zerfoo/zmf (model format)
- github.com/zerfoo/float16, float8 (custom numeric types)
- github.com/google/go-cmp (test comparisons)

### 6.2 Makefile Targets

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

### 6.3 Pre-Commit Hook

- Runs `golangci-lint` on staged packages.
- Runs `go test ./...`.
- Rejects commits touching files in multiple directories.

### 6.4 CI Pipeline (GitHub Actions)

- Unit tests on push/PR to main (excludes parity and numerics tests).
- golangci-lint with 5m timeout.
- Parity tests (currently non-blocking).
- Numerics red team tests (currently non-blocking).
- Nightly toy training pipeline.

---

## 7. Completed Work Summary

### Phase 1: Test Coverage (2026-02-24 to 2026-03-01)

Raised 33 testable packages to >= 93.1% coverage (30 at >= 95%). Added
comprehensive tests for all layers, compute, graph, model, training,
distributed, and utility packages.

### Phase 2: GPU Engine (2026-03-01)

Implemented GPUEngine[T] with 20 native CUDA operations for float32.
Created Storage[T] abstraction, CUDA runtime/cuBLAS bindings, GPU kernels,
parity tests, and benchmarks. All behind `//go:build cuda`.

### Phase 3: GPU Production Readiness (2026-03-01)

Device-resident tensor pipeline (E16), CUDA memory pool (E17), stream
management (E18), graceful error recovery with OOM fallback (E19).
Hardware validation (E15) and production benchmarks (E20) blocked on
GCP GPU quota.

---

## 8. Known Limitations

1. float32 only for GPU -- other types fall back to CPU transparently
2. No broadcasting in GPU kernels -- broadcast cases fall back to CPU
3. Single GPU -- no multi-GPU or distributed GPU support
4. No cuDNN -- all kernels are custom CUDA
5. No mixed precision -- full float32 throughout
6. Default device -- always uses cuda:0, no device selection API
7. Hardware validation pending -- GCP GPU quota request pending
8. No structured logging -- basic Printf-style logging only
9. No metrics export -- no Prometheus or similar integration
10. No TLS/mTLS -- gRPC services unprotected
11. No configuration file loading -- programmatic construction only
12. No health checks or graceful shutdown coordination
