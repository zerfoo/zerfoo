# Getting Started with Zerfoo

Run your first LLM inference in under 10 minutes.

## Prerequisites

- **Go 1.25 or later** -- [Download Go](https://go.dev/dl/)
- A machine with at least 4 GB of RAM (8 GB recommended for 7B models)
- **CUDA toolkit** (optional) -- only needed for GPU acceleration. CPU works out of the box.

Verify your Go installation:

```bash
go version
# go version go1.25.0 linux/amd64
```

## Installation

### As a library

Add Zerfoo to your Go module:

```bash
go get github.com/zerfoo/zerfoo@latest
```

### CLI

Install the `zerfoo` CLI:

```bash
go install github.com/zerfoo/zerfoo/cmd/zerfoo@latest
```

Verify the installation:

```bash
zerfoo version
```

### Build from source

```bash
git clone https://github.com/zerfoo/zerfoo.git
cd zerfoo
go build -o zerfoo ./cmd/zerfoo
```

Zerfoo builds with zero CGo by default. GPU acceleration is loaded dynamically at runtime via purego/dlopen, so you do not need CUDA headers or build tags to compile.

### Platform support

Zerfoo compiles on any platform supported by Go 1.25. GPU acceleration is available on:

- **CUDA** -- NVIDIA GPUs (Linux, Windows)
- **ROCm** -- AMD GPUs (Linux)
- **OpenCL** -- Cross-vendor (Linux, macOS)

## Quick Start

### Minimal Go example

Load a model and generate text in 7 lines of code:

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

`zerfoo.Load` accepts a HuggingFace model ID (e.g. `"google/gemma-3-4b"`) or a
local GGUF file path (e.g. `"./model.gguf"`). If the model is not cached
locally it is downloaded automatically. The default quantization is Q4_K_M.

To request a specific quantization, append it to the ID:

```
google/gemma-3-4b/Q8_0
```

### Streaming

Print tokens as they arrive:

```go
package main

import (
	"context"
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
	fmt.Println()
}
```

### Generation options

Control temperature, token limits, and nucleus sampling:

```go
result, err := m.Generate(context.Background(), "Tell me a joke.",
	zerfoo.WithGenTemperature(0.7),
	zerfoo.WithGenMaxTokens(128),
	zerfoo.WithGenTopP(0.9),
)
if err != nil {
	log.Fatal(err)
}
fmt.Println(result.Text)
fmt.Printf("Tokens: %d, Duration: %s\n", result.TokenCount, result.Duration)
```

## Your First Inference

This section walks you through creating a Go project, downloading a model, and running inference end-to-end.

### Download a model

Zerfoo uses GGUF as its sole model format -- the same format used by llama.cpp. Pull a small quantized model to get started:

```bash
zerfoo pull gemma-3-1b-q4
```

This downloads the quantized GGUF file to a local cache directory (`~/.cache/zerfoo` by default). You can also pull by full HuggingFace repo ID:

```bash
zerfoo pull meta-llama/Llama-3.2-1B-Instruct-GGUF
```

To see which models are cached locally:

```bash
zerfoo list
```

To remove a cached model:

```bash
zerfoo rm gemma-3-1b-q4
```

#### Model aliases

Zerfoo ships with built-in aliases for popular models:

| Alias | HuggingFace Repo |
|-------|-----------------|
| `gemma-3-1b-q4` | `google/gemma-3-1b-it-qat-q4_0-gguf` |
| `gemma-3-2b-q4` | `google/gemma-3-2b-it-qat-q4_0-gguf` |
| `llama-3-1b-q4` | `meta-llama/Llama-3.2-1B-Instruct-GGUF` |
| `llama-3-8b-q4` | `meta-llama/Llama-3.1-8B-Instruct-GGUF` |
| `mistral-7b-q4` | `mistralai/Mistral-7B-Instruct-v0.3-GGUF` |
| `qwen-2.5-7b-q4` | `Qwen/Qwen2.5-7B-Instruct-GGUF` |

You can also pass any HuggingFace repo ID directly, or a local file path.

### CLI usage

Start an interactive chat session:

```bash
zerfoo run gemma-3-1b-q4
```

The model loads and drops you into a prompt:

```
Model loaded. Type your message (Ctrl-D to quit).

> What is the capital of France?
The capital of France is Paris.
>
```

Run a single prompt with `predict`:

```bash
zerfoo predict --model gemma-3-1b-q4 --prompt "Explain what a tensor is in one paragraph."
```

Both commands accept sampling parameters:

| Flag | Description | Default |
|------|-------------|---------|
| `--temperature` | Sampling temperature | 1.0 |
| `--top-k` | Top-K sampling | disabled |
| `--top-p` | Nucleus sampling | 1.0 |
| `--repetition-penalty` | Penalize repeated tokens | 1.0 |
| `--max-tokens` | Maximum tokens to generate | 256 |
| `--system` | System prompt for context | none |
| `--cache-dir` | Override model cache directory | `~/.cache/zerfoo` |
| `--device` | Device to run on (`cpu`, `cuda`) | `cpu` |

### Inference from Go code

Zerfoo is designed to be embedded as a library. Create a new Go project:

```bash
mkdir my-llm-app && cd my-llm-app
go mod init my-llm-app
go get github.com/zerfoo/zerfoo@latest
```

Write `main.go`:

```go
package main

import (
	"context"
	"fmt"
	"log"

	"github.com/zerfoo/zerfoo/inference"
)

func main() {
	// Load a quantized Gemma 3 1B model.
	// On first run, Zerfoo pulls the GGUF file from HuggingFace and caches it.
	mdl, err := inference.Load("gemma-3-1b-q4")
	if err != nil {
		log.Fatal(err)
	}
	defer mdl.Close()

	// Generate text from a prompt.
	result, err := mdl.Generate(
		context.Background(),
		"Explain what a tensor is in one paragraph.",
		inference.WithMaxTokens(128),
	)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(result)
}
```

Run it:

```bash
go run main.go
```

### Chat completion

For multi-turn conversations, use the `Chat` method with structured messages:

```go
resp, err := mdl.Chat(context.Background(), []inference.Message{
	{Role: "system", Content: "You are a helpful assistant."},
	{Role: "user", Content: "What is the capital of France?"},
},
	inference.WithTemperature(0.5),
	inference.WithMaxTokens(64),
)
if err != nil {
	log.Fatal(err)
}
fmt.Println(resp.Content)
fmt.Printf("Tokens used: %d (prompt: %d, completion: %d)\n",
	resp.TokensUsed, resp.PromptTokens, resp.CompletionTokens)
```

The `Chat` method formats messages using the model's built-in chat template and returns a `Response` with token usage statistics.

### Token streaming

Stream tokens as they are generated using the lower-level `inference` package:

```go
err = mdl.GenerateStream(ctx, "Tell me a joke.",
	generate.TokenStreamFunc(func(token string, done bool) error {
		if !done {
			fmt.Print(token)
		}
		return nil
	}),
	inference.WithMaxTokens(128),
)
```

### Loading a local GGUF file

Skip the registry and load a GGUF file directly:

```go
mdl, err := inference.LoadFile("/path/to/model.gguf",
	inference.WithDevice("cuda"),    // use GPU
	inference.WithDType("fp16"),     // FP16 compute precision
	inference.WithKVDtype("fp16"),   // FP16 KV cache
)
```

The `inference.LoadFile` function parses the GGUF file, extracts the tokenizer from its metadata, builds the computation graph for the model's architecture, and returns a ready-to-use `*inference.Model`.

### Custom cache directory

Override the default model cache directory:

```go
mdl, err := inference.Load("gemma-3-1b-q4",
	inference.WithCacheDir("/data/models"),
)
```

From the CLI:

```bash
zerfoo pull gemma-3-1b-q4 --cache-dir /data/models
```

## GPU Acceleration

To use a CUDA GPU, pass the `WithDevice` option:

```go
mdl, err := inference.LoadFile("model.gguf",
	inference.WithDevice("cuda"),
	inference.WithDType("fp16"),
)
```

From the CLI:

```bash
zerfoo run gemma-3-1b-q4 --device cuda
```

No build tags are needed. Zerfoo discovers and loads CUDA libraries at runtime. If CUDA is not available, the call returns an error so your application can fall back to CPU gracefully:

```go
mdl, err := inference.LoadFile("model.gguf", inference.WithDevice("cuda"))
if err != nil {
	log.Println("CUDA not available, falling back to CPU")
	mdl, err = inference.LoadFile("model.gguf")
	if err != nil {
		log.Fatal(err)
	}
}
```

## OpenAI-Compatible API Server

Serve any loaded model behind an OpenAI-compatible HTTP API:

```bash
zerfoo serve gemma-3-1b-q4 --port 8080
```

Send requests using `curl` or any OpenAI client:

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma-3-1b-q4",
    "messages": [{"role": "user", "content": "Hello!"}],
    "temperature": 0.7,
    "max_tokens": 256
  }'
```

Or use streaming with SSE:

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma-3-1b-q4",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": true
  }'
```

Any OpenAI-compatible client library works -- just point it at `localhost:8080`:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="not-needed")
response = client.chat.completions.create(
    model="gemma-3-1b-q4",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response.choices[0].message.content)
```

Available API endpoints:

| Method | Path | Description |
|--------|------|-------------|
| POST | `/v1/chat/completions` | Chat completion |
| POST | `/v1/completions` | Text completion |
| POST | `/v1/embeddings` | Text embeddings |
| GET | `/v1/models` | List loaded models |
| GET | `/v1/models/{id}` | Get model info |
| DELETE | `/v1/models/{id}` | Unload a model |
| GET | `/metrics` | Prometheus metrics |
| GET | `/openapi.yaml` | OpenAPI spec |

## Next Steps

- **[GPU Setup](gpu-setup.md)** -- configure CUDA, ROCm, or OpenCL for hardware-accelerated inference
- **[Design](design.md)** -- architecture overview and key design decisions
- **[Benchmarks](benchmarks.md)** -- throughput numbers across models and hardware
