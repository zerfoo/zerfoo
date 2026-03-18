---
title: "Introducing Zerfoo: A Production-Grade ML Inference Framework for Go"
date: 2026-03-18
author: Zerfoo Team
tags: [announcement, go, ml, inference, llm]
---

# Introducing Zerfoo: A Production-Grade ML Inference Framework for Go

We are excited to announce Zerfoo, a production-grade ML inference and training framework written entirely in Go. Zerfoo lets you run large language models directly inside your Go applications — no Python, no CGo, no external processes. Just `go get` and start generating.

## Why We Built Zerfoo

Running LLMs in production Go services today typically means one of two things: shell out to a Python process, or proxy requests to an external inference server. Both approaches add latency, operational complexity, and failure modes that Go developers shouldn't have to accept.

We built Zerfoo because we believed Go deserved a first-class ML inference runtime. One that compiles with `go build`, embeds as a library, and delivers throughput competitive with C++ runtimes like llama.cpp.

## What Zerfoo Does

Zerfoo is three things:

1. **An inference engine** for transformer models (Llama 3, Gemma 3, Mistral, Qwen 2, Phi 3/4, DeepSeek V3) with GPU acceleration via CUDA, ROCm, and OpenCL.

2. **A training framework** with backpropagation, AdamW/SGD optimizers, and distributed gradient exchange over gRPC/NCCL.

3. **An OpenAI-compatible serving layer** with SSE streaming, request batching, speculative decoding, and Prometheus metrics.

All built on a shared foundation: the `Engine[T]` compute abstraction from our [ztensor](https://github.com/zerfoo/ztensor) library, which provides type-safe tensor operations across CPU and GPU backends.

## Getting Started in 7 Lines

```go
package main

import (
    "fmt"
    "log"

    "github.com/zerfoo/zerfoo"
)

func main() {
    m, err := zerfoo.Load("google/gemma-3-4b")
    if err != nil {
        log.Fatal(err)
    }
    defer m.Close()

    reply, err := m.Chat("Explain quicksort in one sentence.")
    if err != nil {
        log.Fatal(err)
    }
    fmt.Println(reply)
}
```

`zerfoo.Load` accepts a HuggingFace model ID or a local GGUF file path. If the model isn't cached locally, it downloads automatically. The default quantization is Q4_K_M.

## Streaming Tokens

Print tokens as they arrive:

```go
stream, err := m.ChatStream(context.Background(), "Write a haiku about Go.")
if err != nil {
    log.Fatal(err)
}
for tok := range stream {
    if tok.Done {
        break
    }
    fmt.Print(tok.Text)
}
```

## OpenAI-Compatible API Server

Serve any model behind a drop-in replacement for the OpenAI API:

```bash
go install github.com/zerfoo/zerfoo/cmd/zerfoo@latest
zerfoo pull gemma-3-1b-q4
zerfoo serve gemma-3-1b-q4 --port 8080
```

Query it with any OpenAI client:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="unused")
print(client.chat.completions.create(
    model="gemma-3-1b-q4",
    messages=[{"role": "user", "content": "Hello!"}],
).choices[0].message.content)
```

This means every tool, library, and application built for the OpenAI API works with Zerfoo out of the box — LangChain, LlamaIndex, Cursor, and anything else that speaks the OpenAI protocol.

## Performance

On an NVIDIA DGX Spark with Gemma 3 1B Q4_K_M, Zerfoo achieves **245 tokens/second** decode throughput — 20% faster than Ollama (204 tok/s) on the same hardware. This comes from three key optimizations:

- **CUDA graph capture** with 99.5% instruction coverage eliminates per-kernel launch overhead
- **Fused kernels** (FusedAddRMSNorm, FusedSiluGate, FusedQKNormRoPE) reduce memory round-trips
- **Zero CGo overhead** — GPU bindings use purego/dlopen instead of CGo, avoiding the ~200ns per-call overhead

See our [benchmark comparison](../benchmark-comparison.md) for full methodology and reproduction instructions.

## Zero CGo by Default

Zerfoo compiles everywhere Go compiles. GPU acceleration is loaded dynamically at runtime via purego/dlopen — no build tags, no C toolchain, no CUDA SDK at compile time. A plain `go build ./...` produces a working binary. If a CUDA-capable GPU is available at runtime, Zerfoo uses it automatically. If not, it falls back to CPU with ARM NEON and x86 AVX2 SIMD acceleration.

## Type-Safe Generics

Zerfoo uses Go generics throughout. The `Engine[T]` interface, tensor types, and layer implementations are all parameterized over `tensor.Numeric`, which covers float32, float64, float16, bfloat16, float8, and quantized types. This gives you compile-time type safety without sacrificing performance.

## Supported Model Architectures

| Architecture | Special Features |
|-------------|-----------------|
| Llama 3 | RoPE theta=500K |
| Gemma 3 | Tied embeddings, QK norms, logit softcap |
| Mistral | Sliding window attention |
| Qwen 2 | Attention bias, RoPE theta=1M |
| Phi 3/4 | Partial rotary factor |
| DeepSeek V3 | Multi-head Latent Attention, Mixture of Experts |

## The Zerfoo Ecosystem

Zerfoo is part of a family of composable Go libraries:

| Package | Purpose |
|---------|---------|
| [ztensor](https://github.com/zerfoo/ztensor) | GPU-accelerated tensor, compute engine, and computation graph |
| [ztoken](https://github.com/zerfoo/ztoken) | BPE tokenizer with HuggingFace compatibility |
| [float16](https://github.com/zerfoo/float16) | IEEE 754 half-precision and BFloat16 arithmetic |
| [float8](https://github.com/zerfoo/float8) | FP8 E4M3FN arithmetic for quantized inference |
| [zonnx](https://github.com/zerfoo/zonnx) | ONNX-to-GGUF converter CLI |

Each library is independently versioned and usable on its own. If you only need tensors and GPU compute, import ztensor. If you only need tokenization, import ztoken.

## What's Next

We're working toward a v1.0 release with stabilized APIs, expanded model support, and community benchmark contributions. We'd love your feedback — try Zerfoo, run some benchmarks, and let us know what you think.

```bash
go get github.com/zerfoo/zerfoo@latest
```

Welcome to ML inference in Go.
