# Zerfoo Vision

*Last updated: 2026-03-15*

## The Thesis

There is no production-grade ML inference framework written in Go. Python dominates ML, C++ dominates inference runtimes, and Go developers who want to embed model inference into their applications are forced to shell out to Python, bind to C++ via CGo, or use HTTP APIs to external services.

**Zerfoo fills this gap.** It is a Go-native ML framework that makes model inference a library call — as natural as `json.Unmarshal` or `sql.Open`. Import the package, load a model, generate text. No Python runtime, no CGo build complexity, no sidecar processes.

## Where We Are

Zerfoo has grown from a proof-of-concept into a substantial framework:

- **5 repositories**, ~50,000+ lines of Go, ~5,000 lines of CUDA C, ~2,000 lines of ARM/x86 assembly
- **6 model architectures** running: Llama 3, Gemma 3, Mistral, Qwen 2, Phi 3/4, DeepSeek V3
- **234 tok/s** on Gemma 3 1B Q4_K_M — 18.8% faster than Ollama
- **25+ custom CUDA kernels** with zero-CGo purego bindings
- **OpenAI-compatible API server** with streaming, batching, speculative decoding
- **Distributed training** via gRPC and NCCL
- **ARM NEON SIMD assembly** for critical CPU paths (GEMM, RMSNorm, RoPE, SiLU, softmax)

This is real. The framework runs real models at competitive speeds.

## Where We're Going

### Phase 1: Inference Excellence (Current Priority)

**Goal: Be the fastest Go-native inference runtime. Match or exceed llama.cpp throughput across model sizes.**

Key initiatives:

1. **Fix RMSNorm fusion for ONNX path** — Currently the decomposed ONNX path runs at 4-16 tok/s because fused RMSNorm produces wrong numerical output. Fixing this unblocks a 3-6x improvement for all ONNX-imported models.

2. **Expand model coverage** — Add support for Llama 4, Gemma 3n, Phi-4, Command R, and other popular open-weights models as they release. The architecture builder system makes this straightforward.

3. **Continuous batching** — The current batch generation runs requests sequentially. Implement PagedAttention-style continuous batching to serve multiple concurrent requests efficiently, critical for production serving.

4. **Prefill/decode split** — Separate prefill (compute-bound) and decode (memory-bound) phases onto different execution strategies. Prefill benefits from tensor parallelism; decode benefits from CUDA graph replay.

5. **Quantization improvements** — Add GPTQ, AWQ, and GGUF Q5_K/Q6_K native GEMV (currently these are dequantized to Q4_0). Explore W4A16 and W8A8 mixed precision.

### Phase 2: Developer Experience

**Goal: Make Zerfoo the most pleasant ML library any Go developer has ever used.**

1. **One-line inference API**:
   ```go
   model, _ := zerfoo.Load("google/gemma-3-4b")
   response, _ := model.Chat("What is the meaning of life?")
   ```

2. **Embedding API**:
   ```go
   embeddings, _ := model.Embed([]string{"hello world", "goodbye world"})
   similarity := embeddings[0].CosineSimilarity(embeddings[1])
   ```

3. **Structured output** — JSON schema-constrained generation using grammar-guided decoding.

4. **Tool calling** — Native function calling support in the chat API.

5. **Documentation overhaul** — Getting started guide, architecture tour, API reference, troubleshooting guide, and examples repository.

### Phase 3: Training & Fine-tuning

**Goal: Enable LoRA fine-tuning of loaded models directly in Go.**

1. **LoRA adapters** — Load base model, train lightweight rank-decomposition adapters, merge or hot-swap at inference time.

2. **Gradient checkpointing** — Trade compute for memory to fine-tune larger models on consumer GPUs.

3. **Mixed precision training** — FP16/BF16 forward pass with FP32 master weights and loss scaling.

4. **Learning rate scheduling** — Cosine annealing, warmup, and one-cycle policies.

### Phase 4: Ecosystem

**Goal: Build the tools and integrations that make Zerfoo the default choice for Go ML.**

1. **Model hub CLI** — `zerfoo pull`, `zerfoo push`, `zerfoo list` with HuggingFace and custom registry support.

2. **GGUF-first strategy** — GGUF is the community standard for quantized models. Invest in GGUF loading performance and compatibility over ONNX.

3. **Multimodal** — Vision-language models (LLaVA, Gemma 3 with SigLIP encoder). The SigLIP architecture support is already partially built.

4. **Edge deployment** — ARM-optimized builds for Raspberry Pi, Jetson, and mobile. The zero-CGo design already enables this; invest in ARM-specific kernel optimization.

5. **Go ecosystem integrations** — LangChain-Go adapter, Weaviate plugin, standard `database/sql`-style interface patterns.

## Design Principles (Ranked)

1. **Inference throughput** — Tok/s is the north star metric. Every design decision is evaluated against it.
2. **Embeddability** — Zerfoo must work as a Go library import, not just a CLI or server.
3. **Zero-CGo default** — `go build` must work without a C compiler. GPU support is runtime-detected.
4. **Type safety** — Go generics for compile-time correctness. No `interface{}` in hot paths.
5. **Production readiness** — Metrics, logging, health checks, graceful shutdown, TLS. Not afterthoughts.
6. **Simplicity** — Fewer abstractions are better. Don't over-engineer for hypothetical futures.

## Competitive Landscape

| Framework | Language | Strengths | Zerfoo's Advantage |
|-----------|----------|-----------|-------------------|
| llama.cpp | C++ | Raw performance, huge community | Go embeddability, no CGo, cleaner API |
| Ollama | Go (wraps llama.cpp) | Easy CLI, Docker | Native Go (no C++ subprocess), library-first |
| vLLM | Python | Continuous batching, PagedAttention | No Python runtime, embeddable in Go services |
| TensorRT-LLM | C++/Python | NVIDIA-optimized | Vendor-neutral (CUDA + ROCm + OpenCL), simpler |
| ONNX Runtime | C++ | Broad model support | Go-native, no CGo, better DX for Go developers |

## Non-Goals

- **Pre-training at scale** — Zerfoo is not for training GPT-5. Focus is inference + fine-tuning.
- **Python API** — Go-first. If Python users want Zerfoo, they can use the OpenAI-compatible API.
- **Custom hardware backends** — Support NVIDIA, AMD, and OpenCL. Don't chase TPU, Gaudi, or custom ASICs.
- **ONNX runtime replacement** — zonnx converts ONNX to ZMF at build time. Runtime ONNX execution is not a goal.
