# zerfoo

Pure Go ML inference framework -- embed any GGUF model in your Go application with `go build ./...`.

[![CI](https://github.com/zerfoo/zerfoo/actions/workflows/ci.yml/badge.svg)](https://github.com/zerfoo/zerfoo/actions/workflows/ci.yml)
[![Go 1.25+](https://img.shields.io/badge/Go-1.25+-00ADD8.svg)](https://go.dev/)
[![Go Reference](https://pkg.go.dev/badge/github.com/zerfoo/zerfoo.svg)](https://pkg.go.dev/github.com/zerfoo/zerfoo)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

**245 tok/s** on Gemma 3 1B Q4_K_M -- 20% faster than Ollama.

## Quick Start

```go
m, _ := zerfoo.Load("google/gemma-3-4b")  // downloads from HuggingFace
defer m.Close()
response, _ := m.Chat("Explain Go interfaces in one sentence.")
fmt.Println(response)
```

## Installation

```bash
go get github.com/zerfoo/zerfoo
```

## HuggingFace Download

`Load` accepts HuggingFace model IDs. Models are downloaded and cached automatically:

```go
// Download by repo ID (defaults to Q4_K_M quantization)
m, err := zerfoo.Load("google/gemma-3-4b")

// Specify a quantization variant
m, err := zerfoo.Load("google/gemma-3-4b/Q8_0")

// Or load a local GGUF file
m, err := zerfoo.Load("./models/gemma-3-1b.gguf")
```

## Streaming

Stream tokens as they are generated via a channel:

```go
m, _ := zerfoo.Load("google/gemma-3-4b")
defer m.Close()

ch, err := m.ChatStream(context.Background(), "Tell me a joke.")
if err != nil {
    log.Fatal(err)
}
for tok := range ch {
    if !tok.Done {
        fmt.Print(tok.Text)
    }
}
fmt.Println()
```

## Embeddings

Extract L2-normalized embeddings and compute similarity:

```go
m, _ := zerfoo.Load("google/gemma-3-4b")
defer m.Close()

embeddings, _ := m.Embed([]string{
    "Go is a statically typed language.",
    "Rust has a borrow checker.",
})
score := embeddings[0].CosineSimilarity(embeddings[1])
fmt.Printf("similarity: %.4f\n", score)
```

## Structured Output

Constrain model output to valid JSON matching a schema:

```go
import "github.com/zerfoo/zerfoo/generate/grammar"

m, _ := zerfoo.Load("google/gemma-3-4b")
defer m.Close()

schema := grammar.JSONSchema{
    Type: "object",
    Properties: map[string]*grammar.JSONSchema{
        "name": {Type: "string"},
        "age":  {Type: "number"},
    },
    Required: []string{"name", "age"},
}

result, _ := m.Generate(context.Background(),
    "Generate a person named Alice who is 30.",
    zerfoo.WithSchema(schema),
)
fmt.Println(result.Text) // {"name": "Alice", "age": 30}
```

## Tool Calling

Detect tool/function calls in model output (OpenAI-compatible):

```go
import "github.com/zerfoo/zerfoo/serve"

m, _ := zerfoo.Load("google/gemma-3-4b")
defer m.Close()

tools := []serve.Tool{{
    Type: "function",
    Function: serve.ToolFunction{
        Name:        "get_weather",
        Description: "Get the current weather for a city",
        Parameters:  json.RawMessage(`{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}`),
    },
}}

result, _ := m.Generate(context.Background(),
    "What is the weather in Paris?",
    zerfoo.WithTools(tools...),
)

for _, tc := range result.ToolCalls {
    fmt.Printf("call %s(%s)\n", tc.FunctionName, tc.Arguments)
}
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

## CLI

```bash
go install github.com/zerfoo/zerfoo/cmd/zerfoo@latest

zerfoo pull gemma-3-1b-q4          # download a model
zerfoo run gemma-3-1b-q4 "Hello"   # generate text
zerfoo serve gemma-3-1b-q4         # OpenAI-compatible API server
zerfoo list                         # list cached models
```

## Examples

See the [`examples/`](examples/) directory for runnable programs:

- **[chat](examples/chat/)** -- interactive chatbot CLI
- **[rag](examples/rag/)** -- retrieval-augmented generation with embeddings
- **[json-output](examples/json-output/)** -- grammar-guided structured JSON output
- **[embedding](examples/embedding/)** -- embed inference in an HTTP server
- **[api-server](examples/api-server/)** -- standalone API server
- **[inference](examples/inference/)** -- basic text generation

## Links

- **[Getting Started](docs/getting-started.md)** -- full walkthrough: install, pull a model, run inference via CLI and library
- **[GPU Setup](docs/gpu-setup.md)** -- configure CUDA, ROCm, or OpenCL for hardware-accelerated inference
- **[Benchmarks](docs/benchmarks.md)** -- throughput numbers across models and hardware
- **[Design](docs/design.md)** -- architecture overview and key design decisions
- **[Blog](docs/blog/)** -- development updates and deep dives
- **[CONTRIBUTING.md](CONTRIBUTING.md)** -- how to contribute

## License

Apache 2.0
