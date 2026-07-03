# Zerfoo Design Document

## 1. Overview

Zerfoo is a Go 1.25 generics-based machine learning framework designed to be
the best ML framework in the world through superior design, performance, and
ease of use. It supports CPU, CUDA, ROCm, and OpenCL GPU execution through a
unified compute engine interface. All layers, models, and training loops
delegate computation to the Engine[T] interface, enabling transparent hardware
acceleration without modifying application code.

> Product direction and roadmap live in [docs/product-strategy-2026-H2.md](product-strategy-2026-H2.md) ([ADR-093](adr/093-h2-2026-trust-then-traction-strategy.md)).

### 1.1 What Zerfoo Is

A general-purpose ML framework with three production-ready capabilities:

1. **Inference engine.** Load and run open-weights transformer models (text,
   vision, vision-language) via GGUF, the sole model format. ONNX and
   safetensors checkpoints are converted to GGUF at build time using zonnx;
   there is no runtime ONNX execution path.
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
- **Architectural boundaries.** `zerfoo/` imports `github.com/zerfoo/ztensor`
  for tensor/compute/graph and `github.com/zerfoo/ztoken` for tokenizer.
  GGUF is the sole model format.

### 1.3 Validated Model Families

20 architecture builders registered. New architectures are auto-detected
from GGUF metadata via the AutoBuilder.

| Family | Format | Architecture | Status |
|--------|--------|-------------|--------|
| Gemma 3 | GGUF Q4_K | Text decoder | Production (CUDA graph, 244 tok/s) |
| Gemma 3n | GGUF | Mobile text decoder | Working |
| Llama 3 | GGUF | Text decoder | Working |
| Llama 4 | GGUF | Text decoder | Working |
| Mistral | GGUF | Text decoder (sliding window) | Working |
| Mixtral | GGUF | MoE text decoder | Working |
| Qwen 2 | GGUF | Text decoder (attention bias) | Working |
| Phi-3/4 | GGUF | Text decoder (partial RoPE) | Working |
| DeepSeek V3 | GGUF | MoE text decoder (MLA) | Working |
| Command R | GGUF | Text decoder | Working |
| Falcon | GGUF | Text decoder (multi-query) | Working |
| RWKV | GGUF | Linear attention | Working |
| Mamba | GGUF | SSM (selective scan) | Working |
| Mamba 3 | GGUF | MIMO SSM | Parity test PASS |
| Jamba | GGUF | Hybrid Mamba-Transformer | Working |
| Whisper | GGUF | Audio encoder-decoder | Working |
| LLaVA | GGUF | Vision-language | Parity test PASS |
| Qwen-VL | GGUF | Vision-language | Parity test PASS |

See docs/benchmarks.md for current throughput numbers per model.

### 1.4 Execution Path

Zerfoo uses a single GGUF execution path. The inference layer constructs a
graph using fused operations (GroupedQueryAttention, FusedAddRMSNorm, FFN)
that map directly to optimized CUDA kernels. Achieves near-complete CUDA
graph capture and highest throughput.

### 1.5 Maturity Levels

| Capability | Maturity | Notes |
|-----------|----------|-------|
| Inference (GGUF) | Production | CUDA graph capture, Q4_K quantization |
| OpenAI API server | Production | Full spec compliance, 71 integration tests |
| Training | Implemented | All tests pass, no end-to-end workflow documented |
| Distributed training | Implemented | gRPC + NCCL strategies, not production-tested |
| Multi-GPU | Architecture ready | NCCL bindings present, single GPU validated |
| ROCm/OpenCL backends | Implemented | Purego bindings, not hardware-validated |

Module: `github.com/zerfoo/zerfoo` (depends on `github.com/zerfoo/ztensor` and `github.com/zerfoo/ztoken`)

### 1.6 Package Maturity

Each sub-package carries a stability label: **stable**, **beta**, or **alpha**.

- **Stable** — Production-tested, API frozen for v1.0. Breaking changes require a new major version.
- **Beta** — Implemented and tested, API may change in minor releases. Not yet production-hardened.
- **Alpha** — Experimental or newly added. API subject to change or removal without notice.

| Package | Stability | Notes |
|---------|-----------|-------|
| `model/` | stable | Model[T] abstraction, GGUF loader, layer registry |
| `model/gguf/` | stable | GGUF parser, production-validated across 6+ architectures |
| `layers/` | stable | Layer interface, registry wiring |
| `layers/core/` | stable | Foundational ops: Add, MatMul, Linear, FFN, MoE, etc. |
| `layers/activations/` | stable | ReLU, GELU, SwiGLU, Softmax, etc. |
| `layers/attention/` | stable | GQA, SDPA, multi-head attention |
| `layers/normalization/` | stable | RMSNorm, LayerNorm, BatchNorm |
| `layers/embeddings/` | stable | Token and rotary positional embeddings |
| `layers/gather/` | stable | Embedding-table lookup |
| `layers/transpose/` | stable | Transpose op |
| `layers/reducesum/` | stable | ReduceSum op |
| `layers/regularization/` | stable | Dropout |
| `layers/transformer/` | stable | TransformerBlock |
| `layers/registry/` | stable | Central layer registration |
| `layers/components/` | stable | GradientComputer, MatrixMultiplier, WeightInitializer |
| `inference/` | stable | GGUF model loading, architecture builders (Llama, Gemma, etc.) |
| `generate/` | stable | Autoregressive decoding, KV cache, sampling, streaming |
| `generate/speculative/` | beta | Speculative decoding strategies |
| `generate/grammar/` | beta | JSON Schema to CFG constrained decoding |
| `generate/agent/` | alpha | Agentic tool-use loop |
| `serve/` | stable | OpenAI-compatible API server, SSE streaming |
| `serve/batcher/` | beta | Continuous batching scheduler |
| `serve/agent/` | alpha | Agentic loop HTTP adapter |
| `serve/registry/` | beta | bbolt-backed model version registry |
| `serve/disaggregated/` | alpha | Disaggregated prefill/decode serving |
| `config/` | stable | JSON config loader with env var overrides |
| `health/` | stable | HTTP health check endpoints (/healthz, /readyz, /debug/pprof/) |
| `shutdown/` | stable | Ordered shutdown coordinator |
| `registry/` | stable | Model registry with local cache |
| `data/` | beta | Dataset container (Sample, Batch, normalization) |
| `features/` | beta | Time-series feature transformers (Lag, Rolling, FFT) |
| `training/` | beta | Trainer[T], DefaultTrainer, gradient strategies |
| `training/optimizer/` | beta | AdamW[T], SGD[T], EMA, SWA |
| `training/loss/` | beta | MSE[T], CrossEntropyLoss[T] |
| `training/lora/` | beta | LoRA/QLoRA fine-tuning adapters |
| `training/fp8/` | alpha | FP8 mixed-precision training |
| `training/nas/` | alpha | Neural architecture search (DARTS) |
| `training/automl/` | alpha | Bayesian hyperparameter optimization, PBT |
| `training/online/` | alpha | Online learning with drift detection |
| `distributed/` | beta | gRPC-based distributed training |
| `distributed/coordinator/` | beta | Coordinator server with worker registry |
| `distributed/fsdp/` | alpha | Fully Sharded Data Parallelism |
| `distributed/pb/` | beta | Generated protobuf/gRPC bindings |
| `inference/multimodal/` | alpha | Vision, audio, and multi-modal inference |
| `inference/parallel/` | alpha | Tensor and pipeline parallelism for multi-GPU |
| `inference/timeseries/` | alpha | Time-series model architecture builders |
| `layers/residual/` | alpha | Attention Residuals (AttnRes, BlockAttnRes) |
| `layers/recurrent/` | beta | RNN layers |
| `layers/ssm/` | alpha | Mamba, RWKV, S4 state space model blocks |
| `layers/hrm/` | alpha | Hierarchical Reasoning Model modules |
| `layers/vision/` | beta | CLIP/SigLIP vision encoder |
| `layers/audio/` | alpha | Whisper-style audio encoder |
| `layers/timeseries/` | alpha | Time-series patch embedding, variable selection |
| `model/hrm/` | alpha | HRM model types (experimental) |
| `model/huggingface/` | beta | HuggingFace config parsing |
| `tabular/` | alpha | Tabular ML model package |
| `internal/cuda/` | stable | CUDA runtime purego bindings |
| `internal/cuda/kernels/` | stable | Custom CUDA kernel wrappers (25+ kernels) |
| `internal/cublas/` | stable | cuBLAS purego bindings |
| `internal/cudnn/` | beta | cuDNN purego bindings |
| `internal/tensorrt/` | alpha | TensorRT purego bindings |
| `internal/gpuapi/` | beta | GPU Runtime Abstraction Layer (GRAL) |
| `internal/xblas/` | stable | ARM NEON + AVX2 SIMD assembly |
| `internal/codegen/` | alpha | Megakernel code generator |
| `internal/workerpool/` | stable | Persistent goroutine pool |
| `internal/nccl/` | beta | NCCL CGo bindings |
| `internal/hip/` | alpha | AMD HIP runtime purego bindings |
| `internal/hip/kernels/` | alpha | HIP kernel wrappers |
| `internal/rocblas/` | alpha | AMD rocBLAS purego bindings |
| `internal/miopen/` | alpha | AMD MIOpen purego bindings |
| `internal/opencl/` | alpha | OpenCL 2.0 runtime purego bindings |
| `internal/opencl/kernels/` | alpha | OpenCL kernel dispatch |
| `internal/clblast/` | alpha | CLBlast BLAS wrappers |
| `cmd/zerfoo/` | stable | Main CLI binary |
| `cmd/cli/` | stable | CLI framework and command registry |

---

## 2. Core Architecture

### 2.1 Package Layout

This is the post-E124 target layout. Migration in progress; see
docs/plan.md E124 for status. New top-level Go packages require an ADR
slug referenced in the package's `doc.go` (enforced by the layout lint
in T124.1.3). Anything not listed here is unsanctioned.

```
# Provided by github.com/zerfoo/ztensor:
# tensor/             TensorNumeric[T], Storage[T], type constraints (Numeric, Float, Addable)
# numeric/            Type-specific arithmetic (float32/64, float8, float16, int8, uint8), quantization
# compute/            Engine[T] interface, CPUEngine, GPUEngine
# graph/              Computation graph, Node[T] interface, Builder, Parameter, topological execution
# device/             Device, Allocator interfaces (CPU + CUDA)
#
# Provided by github.com/zerfoo/ztoken:
# pkg/tokenizer/      BPE tokenizer loading from tokenizer.json

model/                Model[T], GGUF loader, global layer registry, plugin registry
  model/cache/          Model cache (relocated from top-level modelcache/, T124.5.5)
  model/dsl/            Model DSL (relocated from top-level modeldsl/, T124.5.5)
  model/registry/       Model registry: Pull/Get/List/Delete (relocated from top-level registry/, T124.5.5)
layers/               Neural network layers organized by family
  layers/core/          Add, Sub, Mul, MatMul, MatMulNBits, Cast, Concat, Constant, Conv2d, Dense,
                        FFN, FiLM, GlobalAvgPool, Linear, LMHead, MoE, Pad, Polynomial, Reshape,
                        Resize, RotaryEmbedding, Shape, Slice, SpectralFingerprint, TopK, Unsqueeze, Bias
  layers/activations/   ReLU, LeakyReLU, Sigmoid, Tanh, Gelu, FastGelu, Erf, Softmax, SwiGLU
                        (canonical Node registry; layers/functional/ delegates here per T124.2.2)
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
  layers/ssm/           Mamba, RWKV, S4 (state space models)
  layers/hrm/           HModule, LModule (hierarchical recurrent model)
  layers/gnn/           Graph neural network layers (relocated from top-level gnn/, T124.5.1)
  layers/generative/synth/ VAE-based synthetic data generation (relocated from top-level synth/, T124.5.2)
  layers/shared_latent/ Cross-model latent space (relocated from top-level shared/, T124.5.3)
  layers/registry/      RegisterAll() -- central wiring of all layers into the model registry
training/             Trainer[T], DefaultTrainer, GradientStrategy, workflow interfaces
  training/optimizer/   Optimizer[T] interface, AdamW[T], SGD[T]
  training/loss/        Loss[T] interface, MSE[T], CrossEntropyLoss[T]
  training/rl/          Reinforcement learning (relocated from top-level rl/, T124.4.1)
  training/meta/        MAML meta-learning (relocated from top-level meta/, T124.4.2)
  training/gp/          Tree-based genetic programming (relocated from top-level gp/, T124.4.3)
  training/mlops/monitor/  Model monitoring (relocated from top-level monitor/, T124.4.4)
  training/mlops/recover/  Retraining recovery (relocated from top-level recover/, T124.4.4)
  training/provenance/  Hash-chain model lifecycle audit (relocated from top-level provenance/, T124.4.5)
  training/federated/   FedAvg coordinator (relocated from top-level federated/, T124.4.6)
distributed/          gRPC-based distributed training: AllReduce, Barrier, Broadcast, TLS
  distributed/coordinator/ Coordinator gRPC server with worker registry and checkpoint tracking
  distributed/pb/       Generated protobuf/gRPC bindings
config/               Generic JSON config loader with env var overrides and validation
metrics/              ML evaluation metrics (Pearson, Spearman, MSE, RMSE, MAE)
  metrics/runtime/      Operational instrumentation (Counter, Gauge, Histogram, InMemoryCollector)
log/                  Structured leveled logging (Debug/Info/Warn/Error, text/JSON)
cmd/                  CLI binaries and framework
  cmd/zerfoo/           Main binary (predict, tokenize, worker, pull, run, serve subcommands)
  cmd/cli/              Command interface, CommandRegistry, CLI runner, pull/run/serve commands
  cmd/zerfoo-predict/   Standalone predict binary
  cmd/zerfoo-tokenize/  Standalone tokenize binary
  cmd/bench-compare/    Benchmark comparison tool
  cmd/coverage-gate/    CI coverage enforcement tool
inference/            High-level inference API: Load, Generate, GenerateStream, Chat, Embed
  inference/timeseries/causal/   Causal time-series (relocated from top-level causal/, T124.5.4)
  inference/timeseries/features/ TS feature transformers (relocated from top-level features/, T124.5.4)
  inference/timeseries/regime/   Regime detection (relocated from top-level regime/, T124.5.4)
generate/             Autoregressive generation loop, sampling (temp, topK, topP, repetition), streaming
serve/                OpenAI-compatible HTTP server (chat completions, completions, models, SSE streaming)
  serve/health/         HTTP liveness/readiness probes (relocated from top-level health/, T124.3.1)
  serve/shutdown/       Ordered shutdown coordinator (relocated from top-level shutdown/, T124.3.2)
  serve/support/        Customer-support webhook handlers (relocated from top-level support/, T124.3.3)
  serve/security/       Access control, API keys, rate limit (relocated from top-level security/, T124.3.4)
data/                 Dataset container (Sample, Batch, normalization)
internal/xblas/       CPU BLAS wrappers (gonum GEMM for float32/64; upcast for float16/float8)
internal/cuda/        CUDA runtime purego bindings (dlopen libcudart.so)
internal/cublas/      cuBLAS purego bindings (dlopen libcublas.so)
internal/cuda/kernels/ CUDA kernel source (.cu) and Go wrappers
internal/autoopt/     Kernel autotuning + codegen (relocated from top-level autoopt/, T124.5.6)
tests/                Test suites and shared test infrastructure
  tests/testutil/       Test assertion helpers, MockEngine, custom mocks (renamed from top-level testing/, T124.1.1)
  tests/integration/    Production smoke tests (relocated from top-level integration/, T124.1.2)
  tests/mobile/         Mobile target tests (relocated from top-level mobile/, T124.6.1)
  tests/parity/         Parity tests (env-var gated model forward pass tests)
  tests/parity/testutil/ Shared parity helpers: makeTensor, setup, loadGolden, assertClose (T124.6.2)
  tests/architecture/   Layout lint and composition tests (T124.1.3)
sdk/                  External adapters
  sdk/integrations/     LangChain + Weaviate adapters (relocated from top-level integrations/, T124.1.2)
```

The pre-E124 root contained ~47 top-level Go directories. The target
above keeps the count under 20 (excluding `cmd/`, `docs/`, `examples/`,
`scripts/`, `benchmarks/`, `bin/`, `deploy/`, `infra/`). Open-core
placement of `cloud/`, `marketplace/`, and `compliance/` was resolved by
[ADR-090](adr/090-zerfoo-oss-scope-cloud-marketplace-compliance.md): all three
are extracted to the private `feza-ai/zerfoo-enterprise` repository.

### 2.2 Dependency Graph

```
cmd/* --> model --> graph --> compute --> tensor    (ztensor)
           |         |         |           |
           |         |    numeric.Arithmetic[T]
           |         |                  Storage[T]
           |         |                  /        \
           |         |        CPUStorage[T]  GPUStorage[T]
       layers/*                                    |
           |                               internal/cuda
       graph.Node[T]
```

Key invariant: layers never access tensor data directly for computation. All
arithmetic goes through Engine[T]. This enables transparent CPU/GPU switching.

### 2.3 Architectural Boundaries

- `zerfoo/` imports `github.com/zerfoo/ztensor` for tensor/compute/graph and `github.com/zerfoo/ztoken` for tokenizer.
- `zerfoo/` must not import `zonnx/` or `onnx/` (verified by `make verify-architecture`).
- GGUF is the sole model format.
- All GPU backends use purego (dlopen-based) bindings. No CGo or build tags.
  `go build ./...` compiles everywhere without `-tags cuda`, `-tags rocm`, or
  `-tags opencl`. Runtime detection via `*.Available()` functions.

### 2.4 Type System

Three type constraints govern the generics:

- `tensor.Numeric` -- full union: int types, uint8, float32/64, float8.Float8, float16.Float16, float16.BFloat16
- `tensor.Float` -- standard Go floats only: float32, float64
- `tensor.Addable` -- types supporting native Go operators (+, -, *); excludes custom minifloats

### 2.5 Composition Principle

Complex components must be built by composing smaller components. The `layers/`
package provides 56+ neural network operations. The `compute.Engine[T]` interface
provides type-safe tensor arithmetic. Models should compose these building blocks
rather than reimplementing low-level math.

**Enforcement status (2026-04-02 audit):**

| Path | Composes layers/ | Composes Engine[T] | Status |
|------|------------------|--------------------|--------|
| inference/ (arch_common.go) | 70 imports | Yes | Exemplar |
| inference/ (6 arch builders) | Partial (31 custom nodes) | Partial | Remediation planned (E61) |
| timeseries/ | No (shared helpers via E52) | Partial (E50/E53) | Remediation in progress |
| crossasset/ | No | Partial (E60 GPU path) | Remediation planned |
| tabular/ | No | No | Remediation planned (E62) |
| gnn/ | No | No | Remediation planned (E62) |
| modeldsl/ | No | No | Remediation planned (E62) |

The inference path via `arch_common.go` is the exemplar: it composes
`layers/attention.GroupedQueryAttention`, `layers/normalization.RMSNorm`,
`layers/embeddings.RotaryPositionalEmbedding`, and `layers/core.Linear` into a
computation graph that enables CUDA graph capture, fusion passes, and megakernel
codegen.

The training backends predate the layers/ package and were never fully migrated.
E50/E52/E53 extracted shared helpers and moved some operations to engine ops, but
the fundamental pattern of raw-slice math persists. See ADR-082 for the
remediation strategy.

**Justified exceptions:** Fused CUDA kernels (GEMV+dequant, RoPE+QKNorm,
softmax+V), ARM NEON SIMD assembly, and separated quantized GPU layouts are
performance-justified reimplementations that composition cannot achieve.
See docs/adr/027-composition-prerequisite.md for the full list.

**Known god objects (ztensor):** gpu_engine.go (4,318 lines, 94 methods) and
KernelRunner interface (71 methods) are oversized. 16 quantized matmul methods
are copy-paste variants differing only in storage type and block size.
Remediation planned in E63 (consolidation) and E64 (file decomposition).

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
package uses `init()` for registration. The GGUF model loader uses this
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
| Fused | Dequant+GEMV Q4_K (FP32 FMA; dp4a INT8 variant auto-selected when available) | Custom CUDA kernels (warp shuffle reduction) |
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
256-byte alignment. O(1) reset between tokens. A free-list overlay enables
intra-pass intermediate buffer reuse: freed intermediates are tracked with
best-fit allocation, block splitting, and coalescing. Tensor lifetime analysis
in graph/compile.go identifies last-use points so intermediates can be freed
mid-pass. During inference, all allocations are served by the arena with zero
fallback. Allocation cost is a pointer bump (nanoseconds) vs cudaMalloc
(microseconds).

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
HuggingFace model (GGUF / SafeTensors)
    |
    v
inference.Load(modelID, opts...) -> inference.Model
    |
    v
graph.Graph[T].Forward(ctx, inputs...)
```

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
- testing/testutils: 94.5%

### 7.2 Testing Patterns

- Table-driven tests using standard `testing` package (no testify).
- Parity tests comparing GPU vs CPU output for every GPU-accelerated method.
- Model parity tests gated by env vars (GEMMA3_GGUF_PATH, SIGLIP_GGUF_PATH, KIMI_CONNECTOR_GGUF_PATH,
  LLAMA3_GGUF_PATH, MISTRAL_GGUF_PATH, QWEN25_GGUF_PATH, PHI4_GGUF_PATH, DEEPSEEK_GGUF_PATH).
- Parity tests cover 6 model families: Gemma 3, Llama 3, Mistral, Qwen 2.5, Phi-4, DeepSeek V3.
- Integration tests for cross-package workflows.
- Numerical gradient checking via finite differences.
- MockEngine for unit testing layers in isolation.
- Large-dimension MatMul GPU tests (Llama 3 128K vocab, Gemma 3 262K vocab) with CPU parity.
- Range op edge case tests (16 cases: zero delta, wrong inputs, descending, empty range).
- Multi-model graph forward tests (large LM head, 2-layer transformer, diamond graph).
- CLI pull command tests (16 cases: error paths, nil registry, cached output).
- Model parity on DGX Spark: 8 PASS (Llama3, Qwen25, FlashAttentionGQA),
  13 SKIP (no GGUF: Mistral, Phi4, Gemma3, DeepSeek, SigLIP; 1 device: MultiGPU).
  Multiple ONNX-to-GGUF conversion compatibility fixes applied. See [ADR-018](adr/018-model-parity-testing.md).

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

### 7.5 Per-Op Verification Gates (ADR-091)

Every new op — a `graph.Node` implementation or a new engine method — must pass
three complementary harnesses before merge. The harnesses live in ztensor
`testing/` (gradcheck, oracle) and the parity runner; ADR-091 defines the
policy. A new op is not done until all three gates are green.

1. **gradcheck (math correctness).** Register the op in the OpInfo registry:
   constructor, representative shapes, input domains, tolerances. The harness
   checks the node's analytic `Backward` against float64 central differences on
   the CPU engine. Runs in ordinary CI; catches wrong Jacobians as named tests.
2. **Engine parity under arena stress (implementation correctness).** The same
   op sequence runs CPU-f32 vs GPU-f32, forward AND backward, in interleaved
   schedules (A.fwd, B.fwd, ..., A.bwd) with a small arena to force buffer
   reuse. Catches kernel bugs and the cached-intermediate lifetime-corruption
   class that single-op tests cannot see (pairs with ztensor's
   `ZTENSOR_ARENA_POISON` mode and the ADR 006 save-for-backward contract).
   GPU runs execute as Spark pods on the GPU host (CI has no GPU) and
   serialize on the single device.
3. **PyTorch oracle (convention correctness).** The op's forward+backward case
   bundle is dumped through the registry and replayed in torch inside the
   pinned NGC container; both directions are diffed within per-op tolerances
   (`|ztensor − torch| ≤ atol + rtol·|torch|`; NaN always fails). Catches
   numerics-convention divergence (fast-math, reduction ordering, eps
   placement) that both Go engines could share.

The OpInfo registry's `NewRegistryNode[T]` is the single source of truth for
constructor arguments, so registering an op once enrolls it in gradcheck and
the oracle with identical configurations. Each harness encodes at least one
historically-fixed bug as a red-proof regression fixture proving it would have
caught that bug class.

---

## 8. Build and CI

### 8.1 Dependencies

Direct (go.mod):
- gonum.org/v1/gonum (BLAS)
- google.golang.org/grpc + protobuf (distributed training)
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

### 8.5 GPU Validation (standing DGX arm64 gate)

GitHub CI has no GPU, and the purego GPU bindings cannot cross-compile
darwin->linux/arm64 (`runtime.dlopen` linknames require cgo), so all
GPU-dependent acceptance runs natively on the DGX GB10 through one command:

```bash
scripts/dgx-validate.sh [-ref <git-ref>] [-timeout <seconds>] [-dry-run]
```

The script submits `docs/bench/manifests/validate-arm64.yaml` as a Spark pod
that clones the ref, builds and vets from source on arm64, runs the
cuda-tagged unit tests, and runs the model-parity subset when GGUF files are
mounted; it exits 0 only on a clean JSON report. One GPU pod runs at a time
on the host; interactive SSH remains debugging-only. Results are recorded in
docs/devlog.md per run.

Targeted debugging narrows the GPU test scope with
`-pkgs "<go test args>"` (e.g. `-pkgs "-v -run TestKernelAdd ./internal/cuda/kernels/"`),
which the pod passes verbatim to `go test`. The in-pod stage
(`scripts/dgx-validate-inpod.sh`) enforces a zero-match guard: if a `-run`
regex matches no tests, `go test` still exits 0 with "no tests to run", so the
guard treats that output as a FAILURE rather than a silent pass. Note the sed
substitution in `dgx-validate.sh` uses `|` as its delimiter, so a `-run` regex
must not contain `|` (regex alternation).

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

### 10.3 Graph Execution and CUDA Capture

8. **CUDA graph capture is limited by decomposed ops.** Decomposed ops (Pow, ReduceMean,
   Gather, Slice) break the contiguous capture region. Static Reshape ops
   (1 input, target shape from attributes) are capture-safe and no longer
   break the region; only dynamic Reshape (2+ inputs reading shape from a
   tensor) is non-capturable. The `isNonCapturable()` function in
   graph/cuda_graph.go determines capturability per-instruction. The GGUF
   codegen path achieves near-complete capture. See docs/benchmarks.md.
9. **RMSNorm fusion not yet runtime-correct.** Pattern matching works but
   the fused Forward function produces numerically wrong results due to
   input slot resolution. See docs/devlog.md for investigation status.
10. **Float32 output is sensitive to GEMM accumulation order.** Float32 precision
    accumulation drift compounds through transformer layers. This is inherent
    to float32 with different GEMM accumulation orders, not a bug.

### 10.4 Training

11. **Training infrastructure is implemented but not production-tested.**
    Trainer[T], optimizers (AdamW, SGD), loss functions (MSE, CrossEntropy),
    and distributed gradient exchange all pass unit tests, but no end-to-end
    training workflow is documented or validated.
12. **RMSNorm.Backward is nil-safe.** Returns an error (not panic) if called
    before Forward or if Forward returned early. A nil guard at the top of
    Backward checks r.rms and r.inputTensor before any dereference. See
    SimplifiedLayerNormalization for the reference guard pattern.

### 10.5 Backends

12. **ROCm and OpenCL backends are implemented but not hardware-validated.**
    Purego bindings for HIP/rocBLAS/MIOpen and OpenCL/CLBlast exist and
    compile, but have not been tested on actual AMD or Intel GPUs.

---

## 11. Ecosystem

### 11.1 Companion Repositories

- **ztensor** (`github.com/zerfoo/ztensor`): tensor, compute engine, and computation graph.
- **ztoken** (`github.com/zerfoo/ztoken`): BPE tokenizer loading.
- **zonnx** (`github.com/zerfoo/zonnx`): converts ONNX/safetensors checkpoints to GGUF at build time.
- **float16** (`github.com/zerfoo/float16`): IEEE 754 float16 and bfloat16 types for Go.
- **float8** (`github.com/zerfoo/float8`): E4M3 float8 type for Go.
- **gemma3** (`github.com/zerfoo/gemma3`): Gemma 3 model support and conversion scripts.
- **zmf**: removed. GGUF is the sole model format; the former ZMF format is deprecated.

### 11.2 Inference Pipeline

The inference pipeline provides an embeddable Go-native API for model loading and text generation.

**Loading:** `inference.Load(modelID, opts...)` resolves a model via `registry.ModelRegistry`, reads the GGUF file (weights + metadata) and `tokenizer.json` (BPE tokenizer), then wires a `generate.Generator[float32]` with a `graph.Graph[float32]` and `compute.CPUEngine[float32]`.

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
GGUF file + tokenizer.json
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

### 12.3 Residual Connections

Standard transformers use additive residual connections, where each layer's
output is summed with its input. This is simple and effective, but treats all
previous layers as equally important.

**Attention Residuals** (`layers/residual/`, arXiv:2603.15031) replace fixed
addition with learned, softmax-weighted aggregation over depth. Each layer
carries a pseudo-query vector that attends over RMSNorm-projected keys from
preceding layers, dynamically routing information across depth.

Two variants are available:

- **AttnRes** — Full attention residuals. Every layer attends over all previous
  layer outputs. Maximum expressiveness at the cost of O(L*d) memory to retain
  all L layer representations.

- **BlockAttnRes** — Block attention residuals. L layers are partitioned into N
  blocks. Within a block, outputs accumulate via standard addition. At block
  boundaries, softmax attention aggregates block-level representations, reducing
  memory to O(N*d). Using N=8 blocks recovers the majority of full AttnRes
  benefit.

**Configuration.** Architecture graph builders read a `ResidualConfig` to select
the strategy. For GGUF models, the config is derived from two metadata keys:
`general.residual_mode` (`"standard"`, `"attnres"`, or `"block_attnres"`) and
`general.attnres_blocks` (block count, default 8). Models without these keys
default to standard residuals with no extra overhead.

### 12.4 Config Registry

`inference.ConfigRegistry` maps model family names to config parsers that extract
`ModelMetadata` from `config.json`. Each parser reads architecture-specific fields
(e.g., `rope_scaling`, `partial_rotary_factor`, `n_shared_experts`) and maps them
to the common metadata struct. Global attributes (rope scaling, partial rotation)
are injected via `model.WithGlobalAttributes` during graph construction.

### 12.5 Parameter Name Resolver

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
| [042](adr/042-dp4a-int8-q4k-gemv.md) | dp4a INT8 Q4_K GEMV with FP32 FMA Fallback | 27 | dp4a 4 MACs/instr vs scalar FMA, optional purego soft-symbol, zero regression at batch=1 |
| [043](adr/043-arena-free-list-tensor-lifetime.md) | Arena Free-List with Tensor Lifetime Analysis | 27 | Best-fit free-list overlay on bump-pointer arena; lifetime analysis frees intermediates mid-pass |
| [044](adr/044-paged-attention-kv-block-manager.md) | PagedAttention KV Block Manager | Y1 | Block pool, block table, paged attention CUDA kernel, continuous batching |
| [045](adr/045-speculative-decoding.md) | Speculative Decoding | Y1 | External draft, self-draft, rejection sampling, adaptive alpha fallback |
| [046](adr/046-fp8-nvfp4-quantization-roadmap.md) | FP8 and NVFP4 Quantization Roadmap | Y1-Y2 | FP8 E4M3 dynamic inference, NVFP4 E2M1 Blackwell, FP8 mixed-precision training |
| [047](adr/047-disaggregated-prefill-decode-serving.md) | Disaggregated Prefill/Decode Serving | Y1 | gRPC prefill/decode workers, gateway routing, SSE multiplexing |
| [048](adr/048-mamba-ssm-architecture-support.md) | Mamba/SSM Architecture Support | Y1 | Selective scan CUDA kernel, MambaBlock, SSMState, hybrid Jamba builder |
| [049](adr/049-lora-qlora-finetuning.md) | LoRA/QLoRA Fine-Tuning | Y2 | LoraLinear, LoRA injection, NF4 base weights, adapter GGUF checkpoint |
| [050](adr/050-distributed-training-fsdp.md) | Distributed Training FSDP-Equivalent | Y2 | Parameter sharding, NCCL AllGather/ReduceScatter, gradient accumulation, sharded optimizer |
| [051](adr/051-time-series-ml-platform.md) | Time-Series ML Platform | Y3 | PatchTST, TFT, regime detector, feature store, quantile loss |
| [052](adr/052-online-learning-safety-framework.md) | Online Learning Safety Framework | Y3 | Trigger conditions, incremental LoRA, perplexity/KL validators, rollback, audit log |
| [053](adr/053-multimodal-inference-pipeline.md) | Multi-Modal Inference Pipeline | Y4 | VisionEncoder, projection connector, embedding merge, Whisper audio encoder |
| [054](adr/054-agentic-tool-use-loop.md) | Agentic Tool-Use Loop | Y4 | Tool registry, function-call grammar, supervisor loop, OpenAI tools API |
| [055](adr/055-neural-architecture-search.md) | Neural Architecture Search | Y5 | DARTS bilevel optimization, hardware-aware latency estimator, architecture discretization |
| [056](adr/056-zerfoo-cloud-product.md) | Zerfoo Cloud Product | Y5 | Multi-tenant isolation, token metering, GPU LRU eviction, GKE Terraform (Proposed) |

---

## 15. PagedAttention and Continuous Batching

PagedAttention manages KV cache memory as fixed-size blocks (16 tokens each) via
a block pool and per-sequence block tables, eliminating memory fragmentation.

**Components:**
- `ztensor/graph/kv/block_pool.go` -- Zero-alloc warm-path block allocator
- `ztensor/graph/kv/block_table.go` -- Logical-to-physical KV block mapping per sequence
- `ztensor/internal/cuda/paged_attention.cu` -- CUDA kernel accepting block table pointer arrays
- `serve/batcher/scheduler.go` -- Continuous batching scheduler assembling variable-length batches

**Continuous batching** replaces fixed-batch session pools: completed sequences are
freed immediately without waiting for the batch to finish. The scheduler assembles
per-step batches with zero padding tokens. The ragged attention kernel
(`ztensor/internal/cuda/ragged_attention.cu`) handles variable sequence lengths
in the same batch using block-diagonal masking and online softmax.

See [ADR-044](adr/044-paged-attention-kv-block-manager.md).

---

## 16. Quantization Extensions

### 16.1 FP8 Dynamic Inference

FP8 E4M3FN inference uses cublasLtMatmul for GEMM with per-tensor amax-based
dynamic scaling. Requires sm_89+ (Ada Lovelace or newer). The dispatch path
in GPUEngine.MatMul detects FP8-typed tensors and routes to the FP8 GEMM kernel;
output is dequantized to FP16.

- `ztensor/internal/cuda/fp8_gemm.cu` -- cublasLt FP8 GEMM kernel
- `ztensor/compute/quantize.go` -- Per-tensor amax computation

### 16.2 NVFP4 Inference (Blackwell)

NVFP4 E2M1 with block-scale factors (block size 16) provides 3.5x memory reduction
vs FP16. The GEMV kernel uses a LUT-based dequantization approach with warp shuffle
reductions. Requires sm_100+ (Blackwell); falls back to FP8 on older hardware.

- `ztensor/tensor/quantized.go` -- NVFloat4Storage encode/decode
- `ztensor/internal/cuda/fp4_gemv.cu` -- NVFP4 GEMV kernel

### 16.3 FP8 Mixed-Precision Training

FP8 linear layers use FP8 GEMM for forward pass with FP32 gradient computation
during backward. Dynamic loss scaling halves on inf/NaN and doubles every 2000
clean steps. Master weights maintain FP32 copies for optimizer updates.

- `training/fp8/linear.go` -- FP8Linear forward/backward
- `training/fp8/loss_scaler.go` -- Dynamic loss scaling
- `training/fp8/master_weights.go` -- FP32 master weight store

See [ADR-046](adr/046-fp8-nvfp4-quantization-roadmap.md).

---

## 17. Speculative Decoding

Two draft modes accelerate autoregressive generation:

- **External draft** (`generate/speculative/external_draft.go`): Small model (e.g. 1B)
  generates K draft tokens verified by the target model (e.g. 27B). Draft and target
  share Engine[T] and block manager.
- **Self-draft** (`generate/speculative/self_draft.go`): First N/2 layers of the target
  model generate draft tokens. No separate model needed.

Token acceptance uses Leviathan et al. 2023 rejection sampling
(`generate/speculative/sampler.go`) to guarantee output distribution matches the
target model. The generator automatically falls back to standard decode when
acceptance rate (alpha) drops below 0.4.

Prometheus metric `zerfoo_speculative_acceptance_rate` tracks rolling alpha.

See [ADR-045](adr/045-speculative-decoding.md).

---

## 18. RadixAttention Prefix Caching

A radix tree (`ztensor/graph/kv/radix_tree.go`) indexes KV blocks by token prefix.
Sessions with identical system prompts share physical KV blocks via
`PrefixCache` wrapping the radix tree. LRU eviction handles capacity overflow.

See [ADR-044](adr/044-paged-attention-kv-block-manager.md).

---

## 19. Disaggregated Prefill/Decode Serving

Separates prefill and decode into independent gRPC workers behind an API gateway:

- `serve/disaggregated/prefill_worker.go` -- Runs prefill, streams FP16 KV blocks
- `serve/disaggregated/decode_worker.go` -- Receives KV blocks, runs autoregressive decode
- `serve/disaggregated/gateway.go` -- Least-loaded routing, SSE multiplexing, exponential backoff health check

The gateway routes requests to the least-loaded prefill worker, which streams KV
blocks to a decode worker. The decode worker streams tokens back through the
gateway as SSE events.

See [ADR-047](adr/047-disaggregated-prefill-decode-serving.md).

---

## 20. Mamba/SSM Architecture Support

Mamba-3 state space models use O(d_state) recurrence instead of O(seq_len)
attention, enabling linear-time inference at long sequence lengths.

- `ztensor/internal/cuda/selective_scan.cu` -- Parallel scan CUDA kernel
- `layers/ssm/mamba_block.go` -- MambaBlock[T] with input projection, conv1d,
  selective scan, output projection; forward and backward passes
- `generate/ssm_state.go` -- SSMState[T] managing hidden state across decode steps
- `inference/arch_mamba.go` -- Mamba-3 GGUF loader
- `inference/arch_jamba.go` -- Hybrid Jamba builder interleaving Mamba blocks and
  Transformer layers per mamba_layer_indices metadata

See [ADR-048](adr/048-mamba-ssm-architecture-support.md).

---

## 21. Training Infrastructure

### 21.1 Full Backpropagation

Backward passes implemented for all core layers: RMSNorm, GQA/MHA attention,
SwiGLU/SiLU, RotaryEmbedding, DeepSeek MLA, and MoE routing (straight-through
estimator for discrete top-K gating). Gradient checkpointing
(`ztensor/graph/checkpoint.go`) recomputes activations during backward to reduce
peak memory by 40%+.

### 21.2 LoRA/QLoRA Fine-Tuning

- `training/lora/linear.go` -- LoraLinear[T]: y = Wx + (alpha/r)*B*A*x
- `training/lora/inject.go` -- InjectLoRA replaces named Linear layers, freezes base
- `training/lora/qlora.go` -- QLoRATrainer loads NF4 base, trains LoRA adapters in BF16
- `training/lora/checkpoint.go` -- Adapter save/load as GGUF v3 with lora.{layer}.weight_a/b naming
- `training/optimizers/adamw8bit.go` -- 8-bit AdamW with block-wise quantization (4x memory reduction)
- `cmd/finetune/main.go` -- CLI: `zerfoo finetune --model path --dataset jsonl --rank 16 --epochs 3`

See [ADR-049](adr/049-lora-qlora-finetuning.md).

### 21.3 FSDP Distributed Training

ZeRO Stage 2 equivalent: parameter sharding across N ranks with NCCL AllGather
before forward and ReduceScatter after backward.

- `distributed/fsdp/sharded_module.go` -- ShardedModule splits parameters across devices
- `distributed/nccl.go` -- NCCL AllGather/ReduceScatter via purego dlopen
- `distributed/fsdp/grad_accum.go` -- M micro-step gradient accumulation
- `distributed/fsdp/optimizer_shard.go` -- ShardedAdamW[T] with 1/N moment buffers per rank
- `distributed/fsdp/checkpoint.go` -- AllGather + GGUF write on rank-0, scatter on load
- `cmd/train-distributed/` -- CLI: `zerfoo train-distributed --ranks 4 --model path --dataset jsonl`

See [ADR-050](adr/050-distributed-training-fsdp.md).

### 21.4 Gradient Accumulation Policy

Gradient accumulation across samples/micro-batches stays on the device that
produced the gradients, ordered on the graph's own stream. When no accumulation
engine is configured explicitly (`SetEngine`), the accumulator derives the
graph's compute engine via `Graph.Engine()` (ztensor) and performs in-place
dst-form adds (`Add(ctx, acc, grad, acc)`), so device-resident f32 gradients
accumulate as in-place kernels on the same stream as the graph's kernels --
no per-sample device-to-host round-trip. Host-backed, non-f32, and engine-less
graphs keep the host fallback.

Two upstream ztensor contracts make both paths safe:

- **dst-form storage identity**: an op given a `dst` writes into `dst`'s
  existing storage and never re-homes it onto a pool allocation, so a
  persistent accumulator is never silently converted into an arena tensor that
  a per-step reset recycles behind the live reference.
- **host-access synchronization**: any host read/write of device memory (the
  fallback path's `Data()`/`TrySet`) is stream-ordered via per-device
  registered sync hooks (`tensor.RegisterHostAccessSync`), so a host read can
  never observe bytes from before a still-asynchronous kernel write.

### 21.5 Capture-Replay Training (gated off)

`training/capture_replay.go` provides a `CaptureReplayRunner` that records a
training step's forward + loss + backward + gradient-accumulation walk into a
CUDA graph (on engines implementing `compute.GraphCapturer`) and replays it,
replacing the per-step kernel-launch sequence with a single graph launch.

This path is currently **gated off**. Enabling CUDA-graph capture on the
training walk silently produces wrong gradients (losses ascend while the
identical eager configuration converges), tracked as zerfoo#878. Until the
root-cause fix lands, `NewCaptureReplayRunner` refuses to construct a
capture-enabled runner: when the engine is a `GraphCapturer` and capture is not
disabled via `ZERFOO_DISABLE_CUDA_GRAPH`, construction returns an error unless
`ZERFOO_UNSAFE_CAPTURE_TRAINING=1` is set to acknowledge the hazard. The gate is
containment only, decided once at construction (not per-step). Eager/passthrough
construction (no `GraphCapturer` engine, or capture disabled) is unaffected and
needs no override. Inference-side CUDA-graph capture (see 10.3) is a separate,
unaffected path.

---

## 22. Time-Series ML Platform

### 22.1 Architectures

- `layers/timeseries/patch_embed.go` -- PatchEmbed[T]: splits 1D sequences into
  non-overlapping patches with configurable size and embedding dimension
- `inference/wolf/arch_patchtst.go` -- PatchTST: patch embed + Transformer encoder + projection head
- `layers/timeseries/vsn.go` -- Variable Selection Network with Gated Residual Network
- `inference/wolf/arch_tft.go` -- Temporal Fusion Transformer: static covariates, LSTM,
  VSN, multi-head attention, quantile output (Q10/Q50/Q90)
- `inference/wolf/arch_regime.go` -- Regime detector: GRU + 4-class softmax
  (bull/bear/sideways/volatile)

### 22.2 Feature Store

`inference/wolf/features/store.go` provides a point-in-time-correct feature store
with CSV offline loading and ring buffer (capacity 500) for online updates.
No future timestamps permitted.

### 22.3 Training

- `training/loss/quantile.go` -- Pinball loss and differentiable Sharpe ratio metric
- GGUF metadata keys: `ts.signal.patch_len`, `ts.signal.stride`, `ts.signal.input_features`
- `cmd/wolf_train/main.go` -- Training script with time-ordered train/val split and early stopping

See [ADR-051](adr/051-time-series-ml-platform.md).

---

## 23. Online Learning Safety Pipeline

Incremental model updates with safety gates:

- `training/online/trigger.go` -- Fires when data_count >= 500, hours_since_last >= 24
- `training/online/incremental.go` -- 100 gradient steps at base_lr/10 with gradient clip 0.5;
  updates only LoRA A,B matrices (base frozen)
- `training/online/validator.go` -- Perplexity gate (within 5% of champion) and KL gate
  (KL div < 0.1); both must pass before promotion
- `training/online/rollback.go` -- Swaps LoRA adapter in serve path in under 30 seconds
- `training/online/audit.go` -- Append-only NDJSON audit log for all trigger, update,
  validation, and promotion events

See [ADR-052](adr/052-online-learning-safety-framework.md).

---

## 24. Model Versioning and A/B Testing

- `serve/registry/registry.go` -- bbolt-backed model registry with Register, List,
  Promote (champion), Shadow operations
- `serve/registry/shadow.go` -- ShadowRunner runs both champion and shadow models;
  shadow output logged but not returned; async execution with under 5% latency impact
- `serve/registry/ab_router.go` -- ABRouter with configurable traffic split;
  deterministic session_id hash for sticky routing; runtime-adjustable split
- `serve/registry/canary.go` -- CanaryController: starts at 1% traffic, auto-ramps
  10% every 30 min if error rate and P99 within thresholds; auto-rollback on breach
- Prometheus metrics per model version: `zerfoo_model_requests_total{model_id,version}`,
  `zerfoo_model_latency_p99{model_id,version}`

---

## 25. Multi-Modal Inference

### 25.1 Vision-Language

- `inference/multimodal/preprocess.go` -- Pure Go image preprocessing (resize,
  normalize to [-1,1], patch embedding conversion); JPEG and PNG
- `inference/multimodal/vision_encoder.go` -- VisionEncoder[T] interface; SigLIP implementation
- `inference/multimodal/connector.go` -- Linear projection from vision_dim to text_dim
  via GGUF mm.projector.weight
- `inference/multimodal/merge.go` -- Inserts vision embeddings at image token positions;
  supports up to 4 images per request
- `serve/vision.go` -- OpenAI API image_url content type support (base64 and URL)

### 25.2 Audio Pipeline

- `inference/multimodal/audio.go` -- Pure Go mel-spectrogram extraction (80 filterbanks)
- `layers/audio/whisper_encoder.go` -- WhisperEncoder[T]: 2-layer conv1d + Transformer encoder
- `inference/arch_whisper.go` -- Whisper GGUF loader and encoder graph builder
- Audio+text inference session merges Whisper encoder output with language model
- `POST /v1/audio/transcriptions` -- Multipart/form-data audio upload endpoint

See [ADR-053](adr/053-multimodal-inference-pipeline.md).

---

## 26. Agentic Tool-Use Loop

- `generate/agent/tools.go` -- ToolRegistry with Register(name, schema, handler)
- `generate/agent/function_call.go` -- JSON grammar activated on tool_call token;
  parses ToolCall struct from model output
- `generate/agent/supervisor.go` -- Supervisor executes up to MaxIterations=10 steps;
  detects tool call vs EOS; executes tool; appends result to context
- `generate/agent/tools_market.go` -- 6 market tools (GetMarketData, GetOrderBook,
  GetPortfolio, GetEarningsCalendar, SearchNews, SubmitOrder); RiskApprover gate
  required for SubmitOrder
- `serve/agent/openai_adapter.go` -- OpenAI tools parameter activates agentic mode;
  streaming emits tool_calls delta events

See [ADR-054](adr/054-agentic-tool-use-loop.md).

---

## 27. Neural Architecture Search and AutoML

### 27.1 NAS (DARTS)

- `training/nas/search_space.go` -- Discrete ops (Attention, MLP, Conv1D, SSMBlock),
  connectivity patterns, hyperparameter ranges; JSON-serializable
- `training/nas/darts_layer.go` -- DARTSLayer: softmax-weighted mixture of candidate ops
  with learnable architecture parameters alpha
- `training/nas/darts_optimizer.go` -- Bilevel optimizer: alpha updated by validation
  gradient, weights updated by training gradient, alternating per step
- `training/nas/hw_estimator.go` -- OLS linear latency model calibrated against DGX Spark
- `training/nas/discretize.go` -- Argmax op selection per edge with max_params constraint
- `training/nas/export.go` -- Export discovered architecture + weights as GGUF
- `training/nas/signal_search.go` -- RunSignalNAS for time-series signal models

### 27.2 AutoML

- `training/automl/bayesian.go` -- Gaussian Process surrogate with Expected Improvement
  acquisition function
- `training/automl/pbt.go` -- Population-Based Training (N=8 agents, exploit+explore)
- `training/automl/coordinator.go` -- Pluggable strategy coordinator with Worker interface
  and early stopping
- `cmd/automl/` -- CLI: `zerfoo automl --model path --dataset jsonl --trials 50 --metric sharpe`

See [ADR-055](adr/055-neural-architecture-search.md).

---

## 28. Self-Improving Models

- `training/online/feedback.go` -- FeedbackCollector subscribes to P&L events via gRPC;
  labels signal predictions with realized returns for incremental fine-tuning
- `training/online/drift.go` -- DriftDetector computes rolling 30-day Sharpe ratio;
  alerts when current Sharpe < mean - 1 sigma
- Automated NAS trigger on drift event: runs 2h search on latest data; proposes
  replacement to online safety pipeline validators if discovered architecture
  Sharpe >= current + 5%

See [ADR-052](adr/052-online-learning-safety-framework.md).

---

## 29. Cloud Product

The multi-tenant inference-as-a-service layer -- formerly the `cloud/`,
`marketplace/`, and `compliance/` packages -- has been extracted from this
repository to the private `feza-ai/zerfoo-enterprise` repository. Zerfoo
remains a general-purpose, Apache-2.0 framework; the SaaS serving,
cloud-marketplace billing, and compliance tooling live in the enterprise repo.

See [ADR-090](adr/090-zerfoo-oss-scope-cloud-marketplace-compliance.md) for the
extraction rationale and [ADR-056](adr/056-zerfoo-cloud-product.md) for the
original cloud product proposal.

---

## 30. SentencePiece Tokenizer Integration

Mistral and several other model families use SentencePiece BPE tokenization
with greedy longest-match semantics (distinct from the byte-level BPE used by
Llama/GPT). The ztoken library (v0.3.4+) implements this as a configurable
tokenizer mode, detected from GGUF metadata (`tokenizer.ggml.pre`).

Key design decisions:
- Tokenizer selection is driven by GGUF metadata, not model family name.
  `tokenizer.ggml.pre = "default"` selects greedy longest-match BPE.
- Token IDs are validated against HuggingFace reference outputs during parity
  tests to catch silent regressions.
- BOS/EOS token handling is per-model: Mistral requires BOS prepend, some
  models do not. The generation pipeline reads `tokenizer.ggml.add_bos_token`
  from GGUF metadata.

## 31. Guardian Evaluator Pipeline

Granite Guardian uses a prompt-template-driven evaluation pipeline for content
safety classification. The architecture separates concerns into four stages:

1. **Template engine** (`inference/guardian_templates.go`) -- Renders
   risk-specific prompt templates with user/assistant content slots.
2. **Verdict parser** (`inference/guardian_verdict.go`) -- Extracts Yes/No
   verdict and confidence from model logits (softmax over Yes/No token IDs).
3. **Evaluator** (`inference/guardian_evaluator.go`) -- Orchestrates
   single-risk and multi-risk batch evaluation. Reuses KV cache across
   evaluations for the same context.
4. **Middleware** (`serve/guardian_middleware.go`) -- Optional HTTP middleware
   wrapping `/v1/chat/completions` for automatic input/output safety scanning.

Latency: 77ms median single evaluation on DGX Spark (GB10).

## 32. GGUF Writer Consolidation

The shared `ztensor/gguf` package (ADR-061) consolidates five hand-rolled GGUF
writers into a single implementation. Design:

- `gguf.Writer` is append-only: metadata first, then tensors, then finalize.
  This matches the GGUF v3 file layout (header, metadata KV, tensor info,
  padding, tensor data).
- Round-trip testing via minimal `gguf.Reader` ensures write correctness.
- All consumers (zerfoo training checkpoints, zonnx converter, NAS export,
  FSDP checkpoints, timeseries CLI) import the shared package.
- Migration was mechanical: replace local writer calls with `gguf.NewWriter()`
  and `gguf.AddMetadata*()` / `gguf.AddTensor()` methods.
