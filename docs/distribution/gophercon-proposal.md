# GopherCon Talk Proposal

**Target conference**: GopherCon 2026 or GopherCon 2027

## Title

ML Inference Without Leaving Go

## Abstract

Running ML models in Go typically means one of three compromises: shell out
to a Python process, bind to C++ via CGo, or call an external inference
server over HTTP. Each adds operational complexity, breaks cross-compilation,
or introduces network latency.

Zerfoo is an ML inference framework written entirely in Go that eliminates
all three. It loads GGUF model files and runs transformer inference --
Llama 3, Gemma 3, Mistral, and others -- as a library call. GPU acceleration
(CUDA, ROCm, OpenCL) is loaded dynamically at runtime using purego/dlopen,
so `go build` works without a C compiler and GPU support is detected at
startup, not compile time.

In this talk, attendees will learn how purego replaces CGo for GPU bindings,
how Go generics enable type-safe tensor operations without runtime overhead,
and how CUDA graph capture can push decode throughput to 241 tok/s on a
single GPU -- 28% faster than Ollama on the same hardware. The session
includes a live demo running inference on an NVIDIA DGX Spark.

## Outline (25-minute talk)

1. **The problem** (3 min) -- Why ML in Go is hard today: CGo build
   complexity, Python sidecar latency, vendor lock-in to inference APIs.

2. **Architecture overview** (4 min) -- How Zerfoo is structured: GGUF model
   loading, computation graph construction, architecture-specific graph
   builders (Llama, Gemma, Mistral, etc.).

3. **purego instead of CGo** (5 min) -- How purego/dlopen dynamically loads
   CUDA, cuBLAS, and cuDNN at runtime. The GPU Runtime Abstraction Layer
   (GRAL) that unifies CUDA, ROCm, and OpenCL behind a single interface.
   Trade-offs vs CGo.

4. **Type-safe tensors with Go generics** (3 min) -- The `tensor.Numeric`
   constraint, compile-time type safety across float32/float16/bfloat16/
   float8/quantized types, and how generics replace `interface{}` in hot
   paths.

5. **Performance: CUDA graphs and SIMD** (4 min) -- How CUDA graph capture
   eliminates kernel launch overhead (99.5% coverage), ARM NEON assembly for
   CPU paths, and the road to matching C++ throughput.

6. **Live demo** (4 min) -- Load a Gemma 3 1B model and run inference on a
   DGX Spark. Show the library API, token streaming, and real-time tok/s
   measurement.

7. **What's next** (2 min) -- Continuous batching, LoRA fine-tuning, vision-
   language models, and the broader Go ML ecosystem.

## Demo Plan

- SSH into DGX Spark (NVIDIA Grace Blackwell GB10, 128 GB unified memory)
- Run `go build` to show zero-CGo compilation
- Load Gemma 3 1B Q4_K_M via the library API
- Stream tokens to terminal, display live tok/s counter
- Compare side-by-side with Ollama on the same prompt

## Speaker Bio

[Placeholder -- to be filled in before submission]

## Session Format

- **Length**: 25 minutes
- **Level**: Intermediate
- **Audience**: Go developers interested in ML, systems programming, or GPU
  computing
