# Quickstart

Run LLM inference in Go with 7 lines of code.

## Prerequisites

- **Go 1.25+** -- [go.dev/dl](https://go.dev/dl/)
- No GPU required (CPU works out of the box)

## Install

```bash
go get github.com/zerfoo/zerfoo@latest
```

## Minimal Example

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

## Streaming

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

## Generation Options

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

## OpenAI-Compatible API Server

Serve any model behind an OpenAI-compatible HTTP endpoint:

```bash
# Install the CLI
go install github.com/zerfoo/zerfoo/cmd/zerfoo@latest

# Pull a model
zerfoo pull gemma-3-1b-q4

# Start the server
zerfoo serve gemma-3-1b-q4 --port 8080
```

Query it with curl or any OpenAI client:

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"gemma-3-1b-q4","messages":[{"role":"user","content":"Hello!"}]}'
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="unused")
print(client.chat.completions.create(
    model="gemma-3-1b-q4",
    messages=[{"role": "user", "content": "Hello!"}],
).choices[0].message.content)
```

## GPU Acceleration

To use a CUDA GPU, load from the lower-level `inference` package:

```go
import "github.com/zerfoo/zerfoo/inference"

m, err := inference.LoadFile("model.gguf",
	inference.WithDevice("cuda"),
	inference.WithDType("fp16"),
)
```

No build tags are needed -- GPU libraries are loaded dynamically at runtime.

## Next Steps

- **[Getting Started](getting-started.md)** -- CLI usage, model aliases, full API server docs
- **[GPU Setup](gpu-setup.md)** -- CUDA, ROCm, and OpenCL configuration
- **[Benchmarks](benchmarks.md)** -- throughput numbers across models and hardware
