# Getting Started with Zerfoo

Run your first LLM inference in under 10 minutes.

## Prerequisites

- **Go 1.25 or later** -- [Download Go](https://go.dev/dl/)
- **CUDA toolkit** (optional) -- only needed for GPU acceleration. CPU-only works out of the box.

Verify your Go installation:

```bash
go version
# go version go1.25.0 linux/amd64
```

## Install

Install the `zerfoo` CLI:

```bash
go install github.com/zerfoo/zerfoo/cmd/zerfoo@latest
```

Or build from source:

```bash
git clone https://github.com/zerfoo/zerfoo.git
cd zerfoo
go build -o zerfoo ./cmd/zerfoo
```

## Download a Model

Zerfoo uses GGUF as its sole model format. Pull a quantized model from HuggingFace:

```bash
zerfoo pull google/gemma-3-1b-it-qat-q4_0-gguf
```

This downloads the Gemma 3 1B Q4 GGUF file and caches it locally. You can also use short aliases:

```bash
zerfoo pull gemma-3-1b-q4
```

Available aliases:

| Alias | HuggingFace Repo |
|-------|-----------------|
| `gemma-3-1b-q4` | `google/gemma-3-1b-it-qat-q4_0-gguf` |
| `gemma-3-2b-q4` | `google/gemma-3-2b-it-qat-q4_0-gguf` |
| `llama-3-1b-q4` | `meta-llama/Llama-3.2-1B-Instruct-GGUF` |
| `llama-3-8b-q4` | `meta-llama/Llama-3.1-8B-Instruct-GGUF` |
| `mistral-7b-q4` | `mistralai/Mistral-7B-Instruct-v0.3-GGUF` |
| `qwen-2.5-7b-q4` | `Qwen/Qwen2.5-7B-Instruct-GGUF` |

To use a custom cache directory:

```bash
zerfoo pull gemma-3-1b-q4 --cache-dir /data/models
```

## Run Inference via CLI

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

Customize generation with sampling parameters:

```bash
zerfoo run gemma-3-1b-q4 --temperature 0.7 --max-tokens 512 --top-k 40
```

All `run` options:

| Flag | Description | Default |
|------|-------------|---------|
| `--temperature` | Sampling temperature | 1.0 |
| `--top-k` | Top-K sampling | disabled |
| `--top-p` | Nucleus sampling | 1.0 |
| `--repetition-penalty` | Penalize repeated tokens | 1.0 |
| `--max-tokens` | Maximum tokens to generate | 256 |
| `--system` | System prompt for context | none |
| `--cache-dir` | Override model cache directory | `~/.cache/zerfoo` |

## Run Inference via Library

Import `github.com/zerfoo/zerfoo/inference` to run inference from your own Go code:

```go
package main

import (
	"context"
	"fmt"
	"log"

	"github.com/zerfoo/zerfoo/inference"
)

func main() {
	// Load a model by alias or HuggingFace repo ID.
	// Pulls from HuggingFace automatically if not cached.
	mdl, err := inference.Load("gemma-3-1b-q4")
	if err != nil {
		log.Fatal(err)
	}
	defer mdl.Close()

	// Generate text from a prompt.
	result, err := mdl.Generate(context.Background(), "Explain quicksort in one paragraph.",
		inference.WithTemperature(0.7),
		inference.WithMaxTokens(256),
	)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(result)
}
```

For chat-style multi-turn conversations:

```go
resp, err := mdl.Chat(context.Background(), []inference.Message{
	{Role: "system", Content: "You are a helpful assistant."},
	{Role: "user", Content: "What is the capital of France?"},
},
	inference.WithTemperature(0.7),
	inference.WithMaxTokens(256),
)
if err != nil {
	log.Fatal(err)
}
fmt.Println(resp.Content)
fmt.Printf("Tokens used: %d (prompt: %d, completion: %d)\n",
	resp.TokensUsed, resp.PromptTokens, resp.CompletionTokens)
```

To stream tokens as they are generated:

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

To load a local GGUF file directly (skip the registry):

```go
mdl, err := inference.LoadFile("/path/to/model.gguf",
	inference.WithDevice("cuda"),    // use GPU
	inference.WithDType("fp16"),     // FP16 compute precision
	inference.WithKVDtype("fp16"),   // FP16 KV cache
)
```

## Start the OpenAI-Compatible API Server

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

## Next Steps

- **[GPU Setup](gpu-setup.md)** -- configure CUDA, ROCm, or OpenCL for hardware-accelerated inference
- **[Design](design.md)** -- architecture overview and key design decisions
- **[Benchmarks](benchmarks.md)** -- throughput numbers across models and hardware
