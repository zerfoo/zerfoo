---
title: "Inside Zerfoo: An Architecture Deep Dive"
date: 2026-03-18
author: Zerfoo Team
tags: [architecture, design, cuda, inference, go]
---

# Inside Zerfoo: An Architecture Deep Dive

Zerfoo runs LLM inference in Go at 245 tokens/second — 20% faster than Ollama. This post walks through the internal architecture that makes that possible, from loading a GGUF file to streaming tokens over an OpenAI-compatible API.

## The Pipeline

When you call `zerfoo.Load("google/gemma-3-4b")` followed by `m.Chat("Hello")`, the following pipeline executes:

```
GGUF file on disk
  -> Parse GGUF header + tensors
  -> Map tensor names to canonical form
  -> Create Engine (CPU or GPU)
  -> Architecture registry lookup
  -> Build computation graph
  -> Upload weights to GPU
  -> Prefill: full prompt forward pass
  -> Compile graph + CUDA graph capture
  -> Decode loop: token-by-token generation
  -> Sample next token
  -> Return generated text
```

Each stage maps to specific code in the Zerfoo codebase. Let's walk through them.

## Engine[T]: The Compute Abstraction

The most important design decision in Zerfoo is that **all tensor arithmetic flows through `compute.Engine[T]`**. No layer ever operates on raw slices directly.

```go
func createEngine(device string) (compute.Engine[float32], error) {
    switch devType {
    case "cpu":
        return compute.NewCPUEngine[float32](numeric.Float32Ops{}), nil
    case "cuda":
        return compute.NewGPUEngine[float32](numeric.Float32Ops{}, deviceID)
    }
}
```

`Engine[T]` is defined in our [ztensor](https://github.com/zerfoo/ztensor) library and provides operations like `MatMul`, `Add`, `Reshape`, `Softmax`, and `Transpose`. There are three implementations:

| Implementation | Backend |
|---------------|---------|
| `CPUEngine[T]` | Pure Go + ARM NEON / AVX2 SIMD |
| `GPUEngine[T]` | CUDA via purego (dlopen, no CGo) |
| `EngineProxy[T]` | Wraps any engine; records op traces for compilation |

This abstraction gives us three critical properties:

1. **Testability.** Every layer can be tested on CPU without a GPU. Write tests with `CPUEngine`, and the same code runs on `GPUEngine` in CI on GPU machines.

2. **CUDA graph capture.** The `EngineProxy` records which engine operations are called during a forward pass. This trace is what enables CUDA graph capture later — without it, we'd have no way to know which GPU operations to capture.

3. **Backend portability.** The same model code runs on CUDA, ROCm, OpenCL, and CPU without changes. Only the engine implementation differs.

## GGUF Loading

GGUF is Zerfoo's sole model format. Loading happens in two phases.

**Phase 1: Parse and Extract.** `LoadGGUF(path)` opens the GGUF file, parses the header, extracts model config (architecture, hidden size, number of layers), and loads all weight tensors. Tensor names are mapped from GGUF conventions (`blk.0.attn_q.weight`) to canonical names (`model.layers.0.self_attn.q_proj.weight`). Merged tensors (some architectures store Q/K/V as a single blob) are split into individual tensors.

**Phase 2: Build Model.** `LoadFile(path, opts...)` orchestrates the full pipeline: parse the GGUF file, extract the tokenizer from metadata, create the compute engine, build the computation graph, upload weights to GPU, create the generator with KV cache, and return a ready-to-use `*Model`.

## Architecture Registry

Zerfoo supports multiple model architectures through a registry pattern. Each architecture registers a builder function:

```go
func init() {
    RegisterArchitecture("llama", buildLlamaGraph)
    RegisterArchitecture("gemma", buildGemmaGraph)
    RegisterArchitecture("qwen2", buildQwenGraph)
    RegisterArchitecture("mistral", buildMistralGraph)
    RegisterArchitecture("phi", buildPhiGraph)
    RegisterArchitecture("deepseek_v3", buildDeepSeekGraph)
    // ...
}
```

The `general.architecture` field in the GGUF metadata determines which builder is invoked. Most decoder-only architectures share the same transformer body through `buildTransformerGraph()`, which constructs:

```
Embed -> [RMSNorm -> GQA -> Add -> RMSNorm -> FFN(SiLU-gate) -> Add] x N -> RMSNorm -> LMHead
```

Architecture-specific builders customize this shared structure through options:

| Option | Architecture | Effect |
|--------|-------------|--------|
| `embedScale` | Gemma | Multiply embeddings by sqrt(hidden_size) |
| `qkNorm` | Gemma 3 | Apply RMSNorm to Q/K after projection |
| `logitSoftcap` | Gemma 3 | Soft-cap output logits |
| `slidingWindowSize` | Mistral | Sliding window attention mask |
| `attnBias` | Qwen 2 | Add bias to Q/K/V projections |
| `partialRotaryFactor` | Phi | Apply RoPE to a fraction of head dims |

Adding a new architecture is straightforward: create a builder function, call `buildTransformerGraph()` with the right options, and register it. Architectures that diverge significantly from the standard transformer (like DeepSeek V3 with MLA and MoE) implement their own graph construction.

## Graph Compilation

After the computation graph is built, it gets compiled into an `ExecutionPlan`. This happens lazily on the first decode step — not during model load — so the graph has correct shapes for the decode path (sequence length 1).

The compilation flow:

1. Run a forward pass through the `EngineProxy`, which records the exact sequence of engine operations
2. Convert the trace into an optimized execution plan
3. If CUDA is available, capture the plan as a CUDA graph
4. Store the compiled plan atomically — all subsequent decode steps use `plan.Run()` instead of `graph.Forward()`

The key insight is lazy compilation. During prefill (processing the full prompt), the graph runs with variable-length sequences. During decode, it always processes exactly one token. By compiling after the first decode step, we capture the steady-state execution pattern.

## CUDA Graph Capture

CUDA graph capture is the single biggest performance optimization in Zerfoo. It eliminates per-kernel launch overhead by recording the entire decode step as a single replayable GPU operation.

Without CUDA graphs, each decode step dispatches hundreds of individual kernel launches — each one costing 5-10 microseconds of CPU-GPU synchronization. With CUDA graphs, the entire decode step is a single graph launch.

The numbers tell the story: 245 tok/s with CUDA graphs vs 174 tok/s without — a 40% throughput increase from this optimization alone.

Zerfoo achieves 99.5% instruction coverage in CUDA graph capture. The remaining 0.5% consists of operations that must run on the host: token sampling and tokenizer lookup.

There are subtle engineering challenges in CUDA graph capture:

- **Memory stability.** CUDA graphs record GPU pointer addresses. If a new inference session allocates different GPU buffers, the captured graph's pointers become invalid. Zerfoo solves this with session pooling — the `Model.sessionPool` reuses sessions to keep GPU addresses stable.

- **Arena protection.** After capture, the GPU memory arena's reset floor is raised so that pool resets between tokens don't reclaim buffers the graph still references.

- **Capture failure recovery.** If capture fails, KV cache state is restored from a snapshot and execution falls back to running instructions directly.

## Autoregressive Generation

The `Generator[T]` implements the core generation loop in two phases:

**Prefill:** Process the entire prompt in a single forward pass. This is compute-bound (matrix-matrix multiply) and runs through the uncompiled graph. K/V values for all prompt positions are stored in the KV cache.

**Decode:** Generate tokens one at a time. Each step runs through the compiled execution plan (or CUDA graph), appends K/V for the new position, and samples the next token. This is memory-bandwidth-bound (matrix-vector multiply).

Key implementation details:

- **Tensor reuse.** The decode loop pre-allocates a `[1,1]` tensor and updates its value in-place each step. Zero per-token allocation.
- **Arena reset.** Between tokens, `engine.ResetPool()` reclaims intermediate GPU buffers while protecting CUDA graph references.
- **Multiple KV cache strategies:** Pre-allocated CPU cache, GPU-resident cache, paged cache with shared block pools, and FP16 cache to halve GPU memory usage.

## Token Streaming

For real-time applications, `GenerateStream()` delivers tokens as they are generated through the `TokenStream` interface:

```go
type TokenStream interface {
    OnToken(token string, done bool) error
}
```

Returning a non-nil error from `OnToken()` stops generation — this is how client disconnects are handled cleanly.

## OpenAI-Compatible API Server

The `serve/` package wraps a loaded model in an HTTP server implementing the OpenAI API specification:

| Endpoint | Purpose |
|----------|---------|
| `POST /v1/chat/completions` | Chat completions (streaming and non-streaming) |
| `POST /v1/completions` | Text completions |
| `POST /v1/embeddings` | Embedding generation |
| `GET /v1/models` | List available models |
| `POST /v1/audio/transcriptions` | Audio transcription |
| `GET /metrics` | Prometheus metrics |

Streaming requests use Server-Sent Events (SSE). Non-streaming requests can be grouped into batches via the `BatchScheduler` for higher throughput in multi-client scenarios.

The server includes speculative decoding support: configure a smaller draft model that proposes tokens greedily, then verify them against the target model in a single batched forward pass.

## The Zero-CGo Approach

Zerfoo's GPU bindings deserve special attention because they break from the standard Go approach of using CGo for C library interop.

Instead of CGo, Zerfoo uses purego — a library that calls into shared libraries via dlopen at runtime. This means:

- `go build ./...` compiles everywhere, with no C toolchain required
- No ~200ns per-call CGo overhead across thousands of CUDA API calls per token
- GPU libraries are loaded dynamically — if CUDA isn't available, the binary still runs on CPU
- No build tags needed for CPU-only builds

The tradeoff is more verbose binding code (manually defining function signatures), but the performance and portability benefits are substantial.

## Putting It All Together

The architecture is designed around a single principle: minimize the time the GPU spends waiting. CUDA graph capture eliminates kernel launch overhead. Fused kernels eliminate memory round-trips. Zero CGo eliminates host-side call overhead. Session pooling eliminates buffer reallocation. The result is a pipeline where the GPU spends nearly all its time on actual computation.

For a complete package-by-package map of the codebase, see our [architecture tour](../architecture-tour.md).
