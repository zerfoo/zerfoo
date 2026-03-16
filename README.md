# Zerfoo

**234.30 tok/s** on Gemma 3 1B Q4_K_M -- 18.8% faster than Ollama.

A production-grade ML inference framework written entirely in Go. Pure Go with
zero CGo -- GPU acceleration (CUDA, ROCm, OpenCL) is loaded dynamically at
runtime via purego/dlopen. Import it as a library and run inference directly
from your Go application, or use the CLI and OpenAI-compatible API server.

## Install

```bash
go install github.com/zerfoo/zerfoo/cmd/zerfoo@latest
```

## Quickstart: CLI

Pull a model and run inference:

```bash
zerfoo pull gemma-3-1b-q4
zerfoo run gemma-3-1b-q4 "The quick brown fox"
```

Or start the API server:

```bash
zerfoo serve gemma-3-1b-q4 --port 8080
```

## Quickstart: Library

Import `github.com/zerfoo/zerfoo/inference` to load models and generate text
from your own Go code:

```go
package main

import (
	"context"
	"fmt"
	"log"

	"github.com/zerfoo/zerfoo/inference"
)

func main() {
	// Load a GGUF model by alias or HuggingFace repo ID.
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
	inference.WithDevice("cuda"),
	inference.WithDType("fp16"),
	inference.WithKVDtype("fp16"),
)
```

## Supported Models

| Model | Format | Status |
|-------|--------|--------|
| Gemma 3 | GGUF Q4_K | Production (CUDA graph, highest throughput) |
| Llama 3 | GGUF | Working |
| Qwen 2.5 | GGUF | Working |
| Mistral 7B | GGUF | Working |
| Phi-3/4 | GGUF | Working |
| SigLIP | GGUF | Vision encoder (parity tested) |
| Kimi-VL | GGUF | Vision-language (parity tested) |

## Links

- **[Getting Started](docs/getting-started.md)** -- full walkthrough: install, pull a model, run inference via CLI and library
- **[GPU Setup](docs/gpu-setup.md)** -- configure CUDA, ROCm, or OpenCL for hardware-accelerated inference
- **[Benchmarks](docs/benchmarks.md)** -- throughput numbers across models and hardware
- **[Design](docs/design.md)** -- architecture overview and key design decisions
- **[Blog](docs/blog/)** -- development updates and deep dives
- **[CONTRIBUTING.md](CONTRIBUTING.md)** -- how to contribute

## License

Apache 2.0
