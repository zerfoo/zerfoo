# Tutorial 1: Getting Started with Zerfoo

This tutorial walks you through installing Zerfoo, loading a GGUF model, and running your first inference — all in Go, with no CGo required.

## Prerequisites

- **Go 1.25 or later** — [Download Go](https://go.dev/dl/)
- Internet access to pull a model from HuggingFace on first run

Verify your Go installation:

```bash
go version
# go version go1.25.0 linux/amd64
```

GPU acceleration (CUDA) is optional. All examples in this tutorial run on CPU.

## Step 1: Create a New Module

```bash
mkdir hello-zerfoo && cd hello-zerfoo
go mod init example.com/hello-zerfoo
go get github.com/zerfoo/zerfoo@latest
```

## Step 2: Load a Model and Generate Text

Create `main.go`:

```go
package main

import (
	"context"
	"fmt"
	"log"

	"github.com/zerfoo/zerfoo/inference"
)

func main() {
	ctx := context.Background()

	// Load a quantized Gemma 3 1B model.
	// On first run, Zerfoo pulls the GGUF file from HuggingFace and caches it.
	// Subsequent runs load from cache instantly.
	mdl, err := inference.Load("gemma-3-1b-q4")
	if err != nil {
		log.Fatal(err)
	}

	// Generate text from a prompt.
	response, err := mdl.Generate(ctx, "The capital of France is")
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println(response)
}
```

Run it:

```bash
go run .
# Loading model google/gemma-3-1b-it-qat-q4_0-gguf...
# Paris, which is also the largest city in France...
```

## Step 3: Use Chat Format

Most instruction-tuned models expect a structured conversation format. Use `Chat` instead of `Generate`:

```go
package main

import (
	"context"
	"fmt"
	"log"

	"github.com/zerfoo/zerfoo/inference"
)

func main() {
	ctx := context.Background()

	mdl, err := inference.Load("gemma-3-1b-q4")
	if err != nil {
		log.Fatal(err)
	}

	messages := []inference.Message{
		{Role: "system", Content: "You are a concise assistant. Answer in one sentence."},
		{Role: "user", Content: "What is a transformer model?"},
	}

	resp, err := mdl.Chat(ctx, messages)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println(resp.Content)
	fmt.Printf("Tokens used: %d (prompt: %d, completion: %d)\n",
		resp.TokensUsed, resp.PromptTokens, resp.CompletionTokens)
}
```

## Step 4: Control Sampling

Pass `GenerateOption` values to adjust how the model samples:

```go
response, err := mdl.Generate(ctx, "Explain quantum computing in simple terms",
	inference.WithTemperature(0.7),   // lower = more deterministic
	inference.WithTopK(50),           // sample from top 50 tokens
	inference.WithTopP(0.9),          // nucleus sampling threshold
	inference.WithMaxTokens(200),     // cap output length
)
```

## Step 5: Stream Tokens as They Are Generated

For interactive applications, stream tokens instead of waiting for the full response:

```go
package main

import (
	"context"
	"fmt"
	"log"
	"os"

	"github.com/zerfoo/zerfoo/generate"
	"github.com/zerfoo/zerfoo/inference"
)

func main() {
	ctx := context.Background()

	mdl, err := inference.Load("gemma-3-1b-q4")
	if err != nil {
		log.Fatal(err)
	}

	err = mdl.GenerateStream(ctx, "Write a haiku about Go programming",
		generate.TokenStreamFunc(func(token string, done bool) error {
			if !done {
				fmt.Fprint(os.Stdout, token)
			} else {
				fmt.Fprintln(os.Stdout)
			}
			return nil
		}),
		inference.WithTemperature(0.8),
	)
	if err != nil {
		log.Fatal(err)
	}
}
```

## Model Aliases

`inference.Load` accepts short aliases that map to HuggingFace repo IDs:

| Alias | Model |
|-------|-------|
| `gemma-3-1b-q4` | Gemma 3 1B Q4_0 (1.0 GB) |
| `gemma-3-2b-q4` | Gemma 3 2B Q4_0 (1.5 GB) |
| `llama-3-1b-q4` | Llama 3.2 1B Instruct Q4 |
| `llama-3-8b-q4` | Llama 3.1 8B Instruct Q4 |
| `mistral-7b-q4` | Mistral 7B Instruct v0.3 Q4 |
| `qwen-2.5-7b-q4` | Qwen 2.5 7B Instruct Q4 |

You can also pass any HuggingFace repo ID directly, or a local file path via `inference.LoadFile`.

## Using a Custom Cache Directory

```go
mdl, err := inference.Load("gemma-3-1b-q4",
	inference.WithCacheDir("/data/models"),
)
```

## Next Steps

- [Tutorial 2: Building a Chatbot API](02-chatbot-api.md) — serve your model over HTTP with an OpenAI-compatible API
- [Tutorial 3: Tabular ML](03-tabular-ml.md) — train and predict with `tabular.Model`
