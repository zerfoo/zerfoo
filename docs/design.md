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
  cmd/zerfoo/           Main binary (predict, tokenize, worker subcommands)
  cmd/cli/              Command interface, CommandRegistry, CLI runner
  cmd/zerfoo-predict/   Standalone predict binary
  cmd/zerfoo-tokenize/  Standalone tokenize binary
  cmd/bench-compare/    Benchmark comparison tool
  cmd/coverage-gate/    CI coverage enforcement tool
data/                 Dataset container (Sample, Batch, normalization)
features/             Time-series feature transformers (Lag, Rolling, FFT)
types/                Shared type definitions (BackwardMode)
pkg/tokenizer/        Whitespace-splitting tokenizer
internal/xblas/       CPU BLAS wrappers (gonum GEMM for float32/64; upcast for float16/float8)
internal/cuda/        CUDA runtime CGO bindings (//go:build cuda)
internal/cublas/      cuBLAS CGO bindings (//go:build cuda)
internal/cuda/kernels/ CUDA kernel source (.cu) and Go wrappers
testing/testutils/    Test assertion helpers, MockEngine, custom mocks
tests/                Parity tests, numerics tests, integration test helpers
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

`layers/registry.RegisterAll()` wires all standard layers into
`model.RegisterLayer[T]`. The ZMF model loader uses this registry to
reconstruct graphs from serialized specs.

---

## 4. GPU Engine Architecture

### 4.1 Build Requirements

- CUDA Toolkit 12.x (libcudart, development headers)
- cuBLAS library (libcublas)
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
- `getDevicePtr`: GPUStorage returns device pointer directly (zero-copy).
  CPUStorage allocates from memory pool and copies H2D.
- `makeGPUResult`: Creates output tensors with GPUStorage wrapping device pointer.

### 4.5 CUDA Memory Pool

`internal/cuda/mempool.go`: Size-bucketed free-list allocator. Reuses
previously freed device memory, avoiding per-operation cudaMalloc/cudaFree.
Mutex-synchronized. Drained on `GPUEngine.Close()`.

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

### 4.9 CUDA File Layout

```
compute/
  gpu_engine.go            GPUEngine (pool, stream, cuBLAS) (//go:build cuda)
  gpu_kernels.go           getDevicePtr, makeGPUResult, kernel dispatch (//go:build cuda)

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

### 4.11 Compatible Hardware

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
- Model parity tests gated by env vars (GEMMA3_ZMF_PATH, SIGLIP_ZMF_PATH, KIMI_CONNECTOR_ZMF_PATH).
- Integration tests for cross-package workflows.
- Numerical gradient checking via finite differences.
- MockEngine for unit testing layers in isolation.

### 7.3 Excluded from Coverage Target

| Package | Reason |
|---------|--------|
| distributed/pb/ | Generated protobuf code |
| cmd/zerfoo/ | Main entrypoint, no testable logic |
| cmd/zerfoo-predict/ | Main entrypoint; logic in cmd/cli/ |
| cmd/zerfoo-tokenize/ | Main entrypoint; logic in pkg/tokenizer/ |
| pkg/prelude/ | No statements |
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

- Unit tests on push/PR to main (excludes parity and numerics tests).
- golangci-lint with 5m timeout.
- Parity tests (currently non-blocking).
- Numerics red team tests (currently non-blocking).
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
4. No cuDNN -- all kernels are custom CUDA.
5. No mixed precision -- full float32 throughout.
6. Default device -- always uses cuda:0, no device selection API.
7. Hardware validation pending -- GCP GPU quota request pending.
8. float16/float8 GEMM upcasts to float32 -- no native half-precision kernels.
9. graph.Graph is not thread-safe -- concurrent Forward calls race on memo cache.
10. Generics wiring hardcodes float32 -- registry, worker node, CLI all use float32.

---

## 11. Ecosystem

### 11.1 Companion Repositories

- **zmf** (`github.com/zerfoo/zmf`): Zerfoo Model Format protobuf library.
- **zonnx** (`github.com/zerfoo/zonnx`): ONNX-to-ZMF converter with per-operator builders.
- **float16** (`github.com/zerfoo/float16`): IEEE 754 float16 and bfloat16 types for Go.
- **float8** (`github.com/zerfoo/float8`): E4M3 float8 type for Go.
- **gemma3** (`github.com/zerfoo/gemma3`): Gemma 3 model support and conversion scripts.

### 11.2 Data Flow

```
HuggingFace model (ONNX/SafeTensors)
    |
    v (zonnx converter)
ZMF file
    |
    v (zmf library)
model.LoadModelFromZMF[float32](engine, ops, path)
    |
    v
graph.Graph[float32].Forward(ctx, input)
    |
    v
Output tensor
```
