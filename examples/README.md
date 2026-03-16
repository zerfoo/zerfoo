# Zerfoo Examples

These examples demonstrate Zerfoo's core value: embeddable ML inference in pure Go. Each example is a standalone program you can build and run with `go build`.

## Prerequisites

- **Go 1.25 or later** -- [Download Go](https://go.dev/dl/)
- **A GGUF model file** -- download one from [HuggingFace](https://huggingface.co/models?library=gguf&sort=trending). For a quick start, pull Gemma 3 1B Q4:

```bash
zerfoo pull google/gemma-3-1b-it-qat-q4_0-gguf
```

Or download directly:

```bash
# The model file will be cached in ~/.cache/zerfoo/
zerfoo pull gemma-3-1b-q4
```

- **CUDA toolkit** (optional) -- only needed for GPU acceleration. All examples work on CPU out of the box.

## Available Examples

| Example | Description | Prerequisites |
|---------|-------------|---------------|
| [`inference/`](inference/) | Load a GGUF model and generate text from a prompt. Demonstrates the core `inference.LoadFile` and `model.Generate` API with sampling options (temperature, top-K, top-P) and token streaming. | GGUF model file |
| [`chat/`](chat/) | Interactive chatbot CLI. Demonstrates the `zerfoo.Load` and `model.Chat` one-line API with a readline loop. | GGUF model file |
| [`embedding/`](embedding/) | Embed inference inside a custom Go HTTP handler. Demonstrates the pattern of loading a model once at startup and serving many concurrent requests through your own routing and request/response types. | GGUF model file |
| [`api-server/`](api-server/) | Start an OpenAI-compatible HTTP server backed by a GGUF model. Demonstrates `serve.NewServer` with graceful shutdown. Drop-in replacement for any OpenAI client. | GGUF model file |
| [`json-output/`](json-output/) | Grammar-guided decoding that constrains model output to valid JSON matching a predefined schema. Useful for structured data extraction and tool-calling pipelines. | GGUF model file |
| [`rag/`](rag/) | Retrieval-augmented generation pattern: embed a document corpus, retrieve the most relevant documents via cosine similarity, and generate answers grounded in those facts using `model.Embed` and `model.Chat`. | GGUF model file |

## Running an Example

```bash
# Build and run the inference example
go build -o inference ./examples/inference/
./inference path/to/model.gguf "What is the capital of France?"

# With GPU acceleration (automatic if CUDA is available)
./inference --device cuda path/to/model.gguf "What is the capital of France?"
```

## Further Reading

See [docs/getting-started.md](../docs/getting-started.md) for a full tutorial covering CLI usage, library API, and the OpenAI-compatible server.
