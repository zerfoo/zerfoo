# GopherCon 2026 Talk Proposal

## Talk Title

**Native ML Inference in Go: Zero CGo, Maximum Performance**

## Format

Full session (25 minutes + 5 minutes Q&A)

## Abstract

Go powers some of the world's largest production systems, yet ML inference
remains outsourced to Python and C++ runtimes. Zerfoo changes that. It is a
production-grade ML inference framework written entirely in Go that loads GGUF
models, runs transformer architectures (Llama 3, Gemma 3, DeepSeek V3, and 12+
others), and delivers 233 tokens/second on Gemma 3 1B — 28% faster than Ollama
on the same hardware.

The key insight: you do not need CGo to talk to GPUs. Zerfoo uses purego and
dlopen to bind CUDA, ROCm, and OpenCL at runtime, keeping `go build` working
everywhere with zero build tags for CPU-only deploys. The framework captures
entire inference passes as CUDA graphs, fuses operations to eliminate kernel
launch overhead, and uses hand-written NEON/AVX2 SIMD assembly for CPU paths.

This talk walks through the architecture decisions that made this possible:
type-safe generics with a `tensor.Numeric` constraint, a unified
`compute.Engine[T]` interface that abstracts CPU and GPU, a computation graph
compiler with fusion passes, and quantized GEMM/GEMV kernels (Q4/Q8) that keep
memory bandwidth saturated. We will live-demo loading a model and running
inference in 7 lines of Go, then profile it to show where the time goes.

Attendees will leave knowing how to embed ML inference directly in their Go
services — no sidecar, no FFI, no Python.

## Target Audience

- Go developers building services that need ML capabilities (recommendations,
  classification, content generation, code completion)
- Infrastructure engineers evaluating how to deploy LLMs without Python/C++
  dependencies
- Anyone curious about GPU programming, SIMD optimization, or generics patterns
  in Go

**Prerequisites:** Intermediate Go knowledge. No ML or GPU experience required.

## Detailed Outline

### 1. The Problem: ML in Go Today (3 min)

- Current landscape: CGo wrappers around libtorch, ONNX Runtime, or HTTP calls
  to Python services
- Why these approaches hurt: build complexity, cross-compilation failures,
  latency overhead, operational burden of sidecar processes
- The question: can Go do native ML inference competitively?

### 2. Architecture Overview (5 min)

- GGUF as the universal model format: mmap-friendly, self-describing, llama.cpp
  compatible
- The three-layer stack: ztensor (tensors + compute + graph), ztoken
  (tokenizer), zerfoo (inference + serving)
- Type-safe generics: `compute.Engine[T]` with `tensor.Numeric` constraint
  covering float32, float16, bfloat16, float8, and quantized types
- Why "Engine is law" — all arithmetic flows through the engine interface,
  enabling transparent CPU/GPU switching

### 3. Zero-CGo GPU Binding (5 min)

- How purego + dlopen replaces CGo for CUDA, ROCm, and OpenCL
- Runtime GPU detection: `go build` always works, GPU acceleration loads
  dynamically
- The GPU Runtime Abstraction Layer (GRAL): one interface, three backends
- Memory management: device allocators, arena-based allocation, host-device
  transfers
- Trade-offs and where CGo still has an edge (negligible in practice)

### 4. Making It Fast (7 min)

- CUDA graph capture: recording entire inference passes, replaying with near-zero
  launch overhead (99.5% instruction coverage)
- Operation fusion: FusedAddRMSNorm, FusedSiluGate — fewer kernel launches,
  fewer memory round-trips
- Quantized GEMM/GEMV: Q4_K_M and Q8 kernels that keep memory bandwidth
  saturated
- CPU path: ARM NEON and x86 AVX2 SIMD assembly for when there is no GPU
- Benchmark results: 241 tok/s on Gemma 3 1B Q4_K_M vs Ollama 204 tok/s (+14%)
- Computation graph compiler: fusion passes and megakernel code generation

### 5. Live Demo (5 min)

- Load a Gemma 3 model and run inference in 7 lines of Go
- Show the OpenAI-compatible API server starting up
- Profile a request: where time is spent (tokenization, prefill, decode, sampling)
- Show CUDA graph replay in action via trace visualization

### 6. Embedding in Your Application (3 min)

- Library-first design: import and call, no daemon process
- Serving layer: built-in OpenAI-compatible HTTP server with SSE streaming
- Production features: Prometheus metrics, health checks, graceful shutdown,
  TLS/mTLS
- Supported architectures: Llama 3/4, Gemma 3, Mistral, Qwen 2, Phi 3/4,
  DeepSeek V3, Mixtral, Command R, Falcon, Whisper, Mamba, RWKV, Jamba

### 7. Wrap-Up and Key Takeaways (2 min)

- Go is a viable — and competitive — platform for ML inference
- purego eliminates the CGo tax while keeping GPU performance
- You can ship a single static binary that does ML inference
- The ecosystem is production-ready today

## Key Takeaways

1. **Go can match C++ inference runtimes in throughput** — the bottleneck is
   memory bandwidth and GPU kernel efficiency, not the host language.
2. **purego/dlopen eliminates CGo** — GPU bindings load at runtime, so `go build`
   works everywhere and cross-compilation is not broken.
3. **Generics enable type-safe numeric computing** — `compute.Engine[T]` with a
   `tensor.Numeric` constraint gives compile-time safety across 6+ numeric types.
4. **You can embed inference in any Go service today** — 7 lines to load a model,
   built-in OpenAI-compatible API, production-ready serving infrastructure.

## Speaker Bio

Daniel Ndungu is the founder of Feza, Inc and the creator of Zerfoo, a
production-grade ML inference framework for Go. He has spent the past several
years building GPU-accelerated systems in pure Go, including tensor libraries,
CUDA kernel bindings via purego, and SIMD-optimized compute paths. Daniel is
passionate about making ML inference accessible to the Go ecosystem without
sacrificing performance or developer experience.

- GitHub: github.com/zerfoo
- Company: feza.ai

---

## Backup Plan A: Lightning Talk (5 min)

**Title:** "241 tok/s in Pure Go: ML Inference Without CGo"

**Outline:**

1. **(1 min)** The problem: ML in Go means CGo wrappers or HTTP sidecars
2. **(1.5 min)** The solution: purego/dlopen for GPU binding, GGUF model loading,
   `compute.Engine[T]` abstraction
3. **(1.5 min)** Live demo: 7 lines of Go to load Gemma 3 and generate text,
   show tok/s output
4. **(1 min)** Results: 241 tok/s, 28% faster than Ollama, 15+ model
   architectures, zero CGo — one `go get` away

## Backup Plan B: Unconference Session

**Title:** "GPU Programming in Go Without CGo"

**Format:** 45-minute open discussion / workshop

**Description:**

An interactive session exploring how to bind GPU runtimes (CUDA, ROCm, OpenCL)
from Go using purego and dlopen — no CGo required. We will walk through real
code from the Zerfoo ML framework: how to load shared libraries at runtime, call
GPU APIs, manage device memory, and capture CUDA graphs for replay. Participants
will see the full stack from `dlopen("libcuda.so")` to running a fused
transformer attention kernel.

**Discussion topics:**

- purego vs CGo: when does each make sense?
- Memory management patterns for GPU allocations in Go
- SIMD assembly in Go: ARM NEON and AVX2 via Plan 9 assembler
- Computation graph design: how to represent and optimize ML workloads
- Benchmarking methodology: fair comparisons against C++ runtimes
- The future of numeric computing in Go (generics, SIMD intrinsics proposals)

**Who should attend:** Go developers interested in GPU programming, systems
performance, or ML infrastructure. No prior GPU experience needed — we start
from first principles.
