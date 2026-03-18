---
title: "Migrating from Ollama to Zerfoo"
date: 2026-03-18
author: Zerfoo Team
tags: [migration, ollama, tutorial, go, inference]
---

# Migrating from Ollama to Zerfoo

Ollama is a popular tool for running LLMs locally. If you're using Ollama today and want to switch to Zerfoo — whether for the 20% throughput improvement, Go-native embedding, or OpenAI API compatibility — this guide walks you through the migration step by step.

## Why Migrate?

Before diving into the how, here's what Zerfoo offers over Ollama:

| Feature | Ollama | Zerfoo |
|---------|--------|--------|
| Decode throughput (Gemma 3 1B Q4_K_M) | 204 tok/s | **245 tok/s** (+20%) |
| Language | Go + CGo (wraps llama.cpp) | Pure Go (zero CGo) |
| Embeddable as library | No (separate process) | **Yes** (`go get` and import) |
| OpenAI-compatible API | Yes | Yes |
| CUDA graph capture | No | **Yes** (99.5% coverage) |
| Build requirements | C/C++ toolchain | `go build` only |
| Model format | GGUF | GGUF |
| Deployment | Daemon process | Single binary or library |

The biggest difference is architectural: Ollama runs as a separate daemon process that you communicate with via HTTP. Zerfoo can run as a library embedded directly in your Go application, eliminating the process boundary entirely.

## Step 1: Install Zerfoo

### As a CLI

```bash
go install github.com/zerfoo/zerfoo/cmd/zerfoo@latest
```

This gives you the `zerfoo` command, analogous to the `ollama` command.

### As a Library

```bash
go get github.com/zerfoo/zerfoo@latest
```

This adds Zerfoo to your Go module for use as an embedded library.

## Step 2: Pull Your Models

Zerfoo uses GGUF model files, the same format Ollama uses under the hood. You can either download models through Zerfoo's CLI or use GGUF files you already have.

### Using the CLI

```bash
# Ollama:
ollama pull gemma3:1b

# Zerfoo equivalent:
zerfoo pull gemma-3-1b-q4
```

### Using Existing GGUF Files

If you already have GGUF files (from Ollama, llama.cpp, or HuggingFace), Zerfoo can load them directly:

```bash
zerfoo run ./models/gemma-3-1b-it-Q4_K_M.gguf
```

Ollama stores its models in `~/.ollama/models/`. The GGUF files within that directory work with Zerfoo without modification.

## Step 3: Migrate Your API Calls

Both Ollama and Zerfoo support the OpenAI chat completions API, so migration is often as simple as changing the base URL.

### Starting the Server

```bash
# Ollama:
ollama serve
# Listens on http://localhost:11434

# Zerfoo:
zerfoo serve gemma-3-1b-q4 --port 8080
# Listens on http://localhost:8080
```

### curl

```bash
# Ollama (OpenAI-compatible endpoint):
curl http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"gemma3:1b","messages":[{"role":"user","content":"Hello!"}]}'

# Zerfoo:
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"gemma-3-1b-q4","messages":[{"role":"user","content":"Hello!"}]}'
```

The request and response formats are identical — both follow the OpenAI specification.

### Python (OpenAI SDK)

```python
from openai import OpenAI

# Ollama:
client = OpenAI(base_url="http://localhost:11434/v1", api_key="unused")

# Zerfoo — just change the URL:
client = OpenAI(base_url="http://localhost:8080/v1", api_key="unused")

# Same code from here on:
response = client.chat.completions.create(
    model="gemma-3-1b-q4",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response.choices[0].message.content)
```

### Streaming

```python
# Works identically with both Ollama and Zerfoo:
stream = client.chat.completions.create(
    model="gemma-3-1b-q4",
    messages=[{"role": "user", "content": "Tell me a story."}],
    stream=True,
)
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

## Step 4: Embed in Your Go Application (Optional)

This is where Zerfoo diverges most from Ollama. Instead of running a separate server process, you can embed inference directly in your Go application:

```go
package main

import (
    "fmt"
    "log"
    "net/http"

    "github.com/zerfoo/zerfoo"
)

var model *zerfoo.Model

func main() {
    var err error
    model, err = zerfoo.Load("google/gemma-3-1b")
    if err != nil {
        log.Fatal(err)
    }
    defer model.Close()

    http.HandleFunc("/chat", handleChat)
    log.Fatal(http.ListenAndServe(":9090", nil))
}

func handleChat(w http.ResponseWriter, r *http.Request) {
    reply, err := model.Chat("What is the capital of France?")
    if err != nil {
        http.Error(w, err.Error(), 500)
        return
    }
    fmt.Fprint(w, reply)
}
```

No sidecar process. No HTTP calls to a local inference server. The model runs in-process with your application.

### With Streaming

```go
func handleStreamChat(w http.ResponseWriter, r *http.Request) {
    flusher, ok := w.(http.Flusher)
    if !ok {
        http.Error(w, "streaming not supported", 500)
        return
    }

    stream, err := model.ChatStream(r.Context(), "Tell me a joke.")
    if err != nil {
        http.Error(w, err.Error(), 500)
        return
    }
    for tok := range stream {
        if tok.Done {
            break
        }
        fmt.Fprint(w, tok.Text)
        flusher.Flush()
    }
}
```

### With Generation Options

```go
result, err := model.Generate(ctx, "Explain quicksort.",
    zerfoo.WithGenTemperature(0.7),
    zerfoo.WithGenMaxTokens(256),
    zerfoo.WithGenTopP(0.9),
)
```

## Step 5: GPU Acceleration

Both Ollama and Zerfoo support GPU acceleration, but the mechanism differs.

**Ollama** bundles llama.cpp with CGo bindings. GPU support requires a C/C++ toolchain at build time and the appropriate CUDA/ROCm libraries.

**Zerfoo** loads GPU libraries dynamically at runtime via purego/dlopen. No C toolchain is needed at build time. If CUDA is available at runtime, Zerfoo uses it automatically:

```go
import "github.com/zerfoo/zerfoo/inference"

m, err := inference.LoadFile("model.gguf",
    inference.WithDevice("cuda"),
    inference.WithDType("fp16"),
)
```

To verify GPU detection:

```bash
zerfoo serve gemma-3-1b-q4 --port 8080
# Logs will show: "using CUDA device 0" or "using CPU"
```

## Step 6: Production Configuration

For production deployments, Zerfoo supports features that go beyond what Ollama offers out of the box:

### Prometheus Metrics

```bash
zerfoo serve gemma-3-1b-q4 --port 8080 --metrics
# Metrics available at http://localhost:8080/metrics
```

Exposes request counts, latency histograms, tokens generated, GPU memory usage, and more.

### Speculative Decoding

Use a smaller draft model to accelerate generation:

```bash
zerfoo serve gemma-3-4b-q4 --port 8080 \
  --draft-model gemma-3-1b-q4
```

The draft model proposes tokens greedily, and the target model verifies them in a single batched forward pass. When the acceptance rate is high, this can significantly increase throughput.

### Structured Output (JSON Mode)

Zerfoo supports grammar-constrained decoding for structured output:

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma-3-1b-q4",
    "messages": [{"role": "user", "content": "List 3 countries as JSON"}],
    "response_format": {"type": "json_object"}
  }'
```

## Model Compatibility

Both frameworks use GGUF, so model compatibility is straightforward:

| Architecture | Ollama | Zerfoo |
|-------------|--------|--------|
| Llama 3 | Yes | Yes |
| Gemma 3 | Yes | Yes |
| Mistral | Yes | Yes |
| Qwen 2 | Yes | Yes |
| Phi 3/4 | Yes | Yes |
| DeepSeek V3 | Yes | Yes |

Any GGUF model that works with Ollama should work with Zerfoo. Both frameworks support the same quantization formats (Q4_K_M, Q4_0, Q8_0, FP16, etc.).

## Quick Reference: Command Mapping

| Task | Ollama | Zerfoo |
|------|--------|--------|
| Install | `curl -fsSL ollama.com/install.sh \| sh` | `go install github.com/zerfoo/zerfoo/cmd/zerfoo@latest` |
| Pull model | `ollama pull gemma3:1b` | `zerfoo pull gemma-3-1b-q4` |
| Run interactively | `ollama run gemma3:1b` | `zerfoo run gemma-3-1b-q4` |
| Start server | `ollama serve` | `zerfoo serve gemma-3-1b-q4 --port 8080` |
| List models | `ollama list` | `zerfoo list` |
| API base URL | `http://localhost:11434/v1` | `http://localhost:8080/v1` |
| Embed in Go | Not supported | `go get github.com/zerfoo/zerfoo` |

## Summary

Migrating from Ollama to Zerfoo is straightforward because both use GGUF models and implement the OpenAI API. The main steps are:

1. Install the Zerfoo CLI or add the library to your Go module
2. Pull or reuse your existing GGUF model files
3. Change the base URL in your API clients
4. Optionally, embed inference directly in your Go application

The reward: 20% faster decode throughput, zero-CGo builds, in-process inference, and a single-binary deployment model.

```bash
go install github.com/zerfoo/zerfoo/cmd/zerfoo@latest
zerfoo pull gemma-3-1b-q4
zerfoo serve gemma-3-1b-q4 --port 8080
```
