# Part 1: Getting Started with Zerfoo

This tutorial walks you through installing Zerfoo, downloading a model, and running your first inference -- all from the command line and from Go code.

## Prerequisites

- Go 1.25 or later
- A machine with at least 4 GB of RAM (8 GB recommended for 7B models)
- Optional: an NVIDIA GPU with CUDA drivers for hardware-accelerated inference

## Installing Zerfoo

Install the Zerfoo CLI with a single `go install`:

```bash
go install github.com/zerfoo/zerfoo/cmd/zerfoo@latest
```

Verify the installation:

```bash
zerfoo version
```

Zerfoo builds with zero CGo by default. GPU acceleration is loaded dynamically at runtime via purego/dlopen, so you do not need CUDA headers or build tags to compile.

## Downloading a Model

Zerfoo uses the GGUF model format -- the same format used by llama.cpp. Models are available from HuggingFace and can be downloaded with the `pull` command.

Zerfoo ships with built-in aliases for popular models. For example, `gemma-3-1b-q4` resolves to `google/gemma-3-1b-it-qat-q4_0-gguf`. Pull a small model to get started:

```bash
zerfoo pull gemma-3-1b-q4
```

This downloads the quantized GGUF file to a local cache directory. You can also pull by full HuggingFace repo ID:

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

## First Inference with the CLI

The `run` command starts an interactive chat session:

```bash
zerfoo run gemma-3-1b-q4
```

You can also run a single prompt with `predict`:

```bash
zerfoo predict --model gemma-3-1b-q4 --prompt "Explain what a tensor is in one paragraph."
```

Both commands accept sampling parameters:

```bash
zerfoo predict \
  --model gemma-3-1b-q4 \
  --prompt "Write a haiku about Go." \
  --temperature 0.7 \
  --max-tokens 64
```

## First Inference from Go Code

Zerfoo is designed to be embedded as a library. Here is a minimal Go program that loads a GGUF model and generates text:

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

You can also load from a local GGUF file directly:

```go
mdl, err := inference.LoadFile("gemma-3-1b-it-q4_0.gguf")
```

The `inference.LoadFile` function parses the GGUF file, extracts the tokenizer from its metadata, builds the computation graph for the model's architecture, and returns a ready-to-use `*inference.Model`.

## Running on GPU

To use a CUDA GPU, pass the `WithDevice` option:

```go
mdl, err := inference.LoadFile("gemma-3-1b-it-q4_0.gguf",
	inference.WithDevice("cuda"),
)
```

From the CLI, add the `--device` flag:

```bash
zerfoo run gemma-3-1b-q4 --device cuda
```

No build tags are needed. Zerfoo discovers and loads CUDA libraries at runtime. If CUDA is not available, the call to `WithDevice("cuda")` returns an error, so your application can fall back to CPU gracefully:

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

## Chat Completion

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

## Using a Custom Cache Directory

By default, Zerfoo caches downloaded models in a platform-specific directory. You can override this:

```go
mdl, err := inference.Load("gemma-3-1b-q4",
	inference.WithCacheDir("/data/models"),
)
```

From the CLI:

```bash
zerfoo pull gemma-3-1b-q4 --cache-dir /data/models
```

## Model Aliases

`inference.Load` accepts short aliases that map to HuggingFace repo IDs:

| Alias | Model |
|-------|-------|
| `gemma-3-1b-q4` | Gemma 3 1B Q4_0 |
| `gemma-3-2b-q4` | Gemma 3 2B Q4_0 |
| `llama-3-1b-q4` | Llama 3.2 1B Instruct |
| `llama-3-8b-q4` | Llama 3.1 8B Instruct |
| `mistral-7b-q4` | Mistral 7B Instruct v0.3 |
| `qwen-2.5-7b-q4` | Qwen 2.5 7B Instruct |

You can also pass any HuggingFace repo ID directly, or a local file path via `inference.LoadFile`.

## What is Next

- [Part 2: Model Loading and Architecture Support](02-model-loading.md) -- learn about GGUF, supported architectures, and quantization options.
- [Part 3: Text Generation Deep Dive](03-text-generation.md) -- explore sampling strategies, streaming, and KV cache tuning.
- [Part 4: Running the OpenAI-Compatible API Server](04-api-server.md) -- serve models over HTTP with `zerfoo serve`.
- [Part 5: Tabular and Time-Series ML](05-tabular-timeseries.md) -- use Zerfoo for structured data prediction and forecasting.
