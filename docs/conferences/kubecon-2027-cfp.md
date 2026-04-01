# KubeCon + CloudNativeCon 2027 — CFP Submission

## Title

Running ML Inference in Pure Go: How Zerfoo Eliminates CGo for Production GPU Workloads

## Track

AI + ML (alternative: Cloud Native Applications)

## Session Format

Breakout session (35 minutes)

## Abstract

Go services that need ML inference today face a hard choice: embed a CGo-heavy
runtime with fragile build toolchains and opaque crash stacks, or call out to a
Python sidecar and accept the latency, operational cost, and scaling complexity
that comes with it. Zerfoo removes this dilemma entirely.

Zerfoo is a production-grade ML inference framework written in pure Go. It loads
GGUF models, supports 18 transformer architectures (Llama 3/4, Gemma 3, DeepSeek
V3, Mistral, Phi, Qwen, and more), and delivers 233 tokens/second on Gemma 3 1B
Q4_K_M — 28% faster than Ollama on identical hardware. GPU acceleration (CUDA,
ROCm, OpenCL) is loaded at runtime via purego and dlopen, so `go build ./...`
compiles everywhere with no C toolchain required.

This talk explains the architecture that makes zero-CGo GPU inference possible,
demonstrates the Kubernetes operator that turns model deployment into a
`kubectl apply`, and shows production serving patterns — adaptive batching,
multi-model LRU eviction, and prefill/decode disaggregation — that let platform
teams run ML workloads with the same operational tools they already use.

## Detailed Description

### The Problem

The Go ecosystem lacks a native ML inference runtime. Teams building Go services
that need LLM capabilities (chat, embeddings, code completion, document
processing) are forced into one of two compromises:

1. **CGo bindings** to C++ runtimes (llama.cpp, ONNX Runtime). This introduces
   build complexity (gcc/nvcc toolchains, platform-specific shared libraries),
   breaks cross-compilation, produces unreadable crash stacks, and complicates
   container images.

2. **Python sidecars** (vLLM, TGI, Triton). This doubles the operational
   surface: separate container images, separate scaling policies, network hops
   on every inference call, and a fundamentally different debugging and
   deployment model.

Both approaches undermine the simplicity and reliability that drew teams to Go
and Kubernetes in the first place.

### The Solution: Zero-CGo GPU Acceleration via purego

Zerfoo solves this by binding GPU libraries (CUDA, cuBLAS, cuDNN, TensorRT,
ROCm/HIP, rocBLAS, MIOpen, OpenCL) at runtime through Go's purego package and
the operating system's dlopen mechanism. There is no C compiler in the build
chain. The binary is a standard Go executable that discovers GPU capabilities
at startup and falls back to optimized CPU paths (ARM NEON SIMD, x86 AVX2)
when no GPU is present.

This talk covers how this works in practice:

- **Dynamic symbol resolution**: purego loads `.so`/`.dylib` files at runtime
  and resolves function pointers. Zerfoo wraps 25+ CUDA kernels, cuBLAS GEMM
  routines, and memory management APIs behind Go function signatures with
  zero CGo overhead.

- **The `compute.Engine[T]` interface**: all tensor arithmetic flows through a
  single generic interface parameterized by numeric type (float32, float16,
  bfloat16, float8, quantized). Swapping from CPU to GPU is a one-line engine
  change — no model code is modified. This same interface enables CUDA graph
  capture, which eliminates kernel launch overhead and covers 99.5% of
  instructions on the GGUF inference path.

- **Quantized inference**: Q4_K_M, Q8_0, and other GGUF quantization formats
  run natively. The 241 tok/s benchmark on Gemma 3 1B Q4_K_M uses quantized
  GEMM/GEMV kernels written in CUDA PTX, dispatched through purego.

- **Cross-compilation**: `GOOS=linux GOARCH=arm64 go build` produces a static
  binary that runs on ARM64 edge nodes with NEON SIMD acceleration. No runtime
  dependencies, no Python, no shared libraries required on the target.

### Cloud-Native Deployment

The second half of the talk covers how Zerfoo integrates with Kubernetes:

- **ZerfooInferenceService CRD**: a Kubernetes operator that lets platform
  teams declaratively manage model deployments — specifying model, quantization,
  replica count, GPU memory budget, and latency SLO targets in a single YAML
  manifest.

- **Adaptive batching**: a control loop that dynamically adjusts batch size
  (1-32) based on queue depth and latency EMA, hitting p99 SLOs under variable
  load without manual tuning.

- **Multi-model serving**: a single pod serves multiple models within a GPU
  memory budget using LRU eviction — no manual eviction policy needed.

- **Prefill/decode disaggregation**: separate compute profiles across node
  pools via gRPC, right-sizing each tier independently.

- **Observability**: Prometheus metrics (`tokens_per_second`,
  `request_latency_ms`, `queue_depth`), Grafana dashboards, and standard
  Kubernetes health probes (`/health/live`, `/health/ready`).

## Talk Outline

| Time | Section | Content |
|------|---------|---------|
| 0:00 | The CGo Problem | Why CGo and Python sidecars are untenable for Go teams doing ML at scale |
| 4:00 | Zerfoo Architecture | Three-layer stack (ztensor, ztoken, zerfoo), GGUF model format, `compute.Engine[T]` |
| 8:00 | purego Deep Dive | How dlopen/symbol resolution replaces CGo; CUDA kernel dispatch; memory management |
| 14:00 | Performance | 241 tok/s benchmark, CUDA graph capture (99.5% coverage), quantized GEMM, ARM NEON |
| 18:00 | Kubernetes Operator | ZerfooInferenceService CRD, adaptive batching, multi-model LRU, disaggregated serving |
| 26:00 | Live Demo | `kubectl apply` a model, run inference, show Grafana metrics under load |
| 30:00 | Edge Deployment | Cross-compile to ARM64, DaemonSet pattern, Raspberry Pi 5 / Jetson Orin Nano |
| 33:00 | Q&A | |

## Key Takeaways

1. **CGo is not required for GPU acceleration in Go.** purego and dlopen provide
   a production-viable alternative that preserves cross-compilation, readable
   stack traces, and standard Go tooling.

2. **ML inference belongs in your Go binary, not a sidecar.** Embedding inference
   eliminates network hops, simplifies scaling, and reduces operational surface
   by half.

3. **GGUF models deploy like ConfigMaps.** The llama.cpp-compatible GGUF format
   is self-describing and mmap-friendly — models load from a PersistentVolumeClaim
   with no preprocessing step.

4. **241 tok/s in pure Go is competitive with C++ runtimes.** Fused operations,
   CUDA graph capture, and quantized kernels close the performance gap without
   sacrificing Go's developer experience.

5. **A Kubernetes operator makes ML serving declarative.** Platform teams manage
   model deployments with `kubectl apply`, the same tool they use for everything
   else.

## Speaker Bio

**Daniel Ndungu** — Founder, Feza, Inc / Creator of Zerfoo

Daniel Ndungu is the founder of Feza, Inc and creator of Zerfoo, a
production-grade ML inference and training framework for Go. He has spent the
past several years building GPU-accelerated systems in pure Go — including
hand-written CUDA kernels, tensor libraries with type-safe generics, GPU runtime
bindings via purego/dlopen, ARM NEON SIMD assembly, and a Kubernetes operator for
declarative ML serving. Daniel's work on Zerfoo demonstrates that Go can be
competitive with C++ runtimes for ML inference without sacrificing the language's
core strengths: simplicity, cross-compilation, and operational clarity.

Daniel is passionate about making ML infrastructure a first-class citizen of the
cloud-native ecosystem, accessible to the millions of Go developers who currently
have no native option for embedding inference in their services.

- GitHub: [github.com/zerfoo](https://github.com/zerfoo)
- Company: [feza.ai](https://feza.ai)

## Submission Notes

- This proposal targets the **AI + ML** track but is equally relevant to
  **Cloud Native Applications** given the Kubernetes operator and serving
  architecture content.
- The talk complements the GopherCon 2026 proposal ("Native ML Inference in Go:
  Zero CGo, Maximum Performance") by shifting focus from Go language internals
  to the Kubernetes operational story and the purego/dlopen GPU binding
  approach. The audience overlap is minimal: GopherCon attendees are primarily
  language-focused; KubeCon attendees are primarily platform and infrastructure
  focused.
- All demos are self-contained and can run in a KinD cluster on a laptop for
  the operator and multi-model sections, with a pre-recorded fallback for GPU
  benchmarks and edge ARM64 deployment.
- The speaker can provide a longer (45-minute) version if scheduled for a
  tutorial slot, expanding the live demo and adding a hands-on section where
  attendees deploy a model to a shared cluster.
