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
| [`chat/`](chat/) | Interactive chatbot CLI. Demonstrates the `zerfoo.Load` and `model.Chat` one-line API with a readline loop. | GGUF model file |
| [`embedding-search/`](embedding-search/) | Semantic search over a document corpus using model embeddings and cosine similarity. | GGUF model file |
| [`rag/`](rag/) | Retrieval-augmented generation: embed documents, retrieve relevant ones, and generate grounded answers. | GGUF model file |
| [`code-completion/`](code-completion/) | Generate code completions from partial code snippets using `inference.LoadFile` and `model.Generate`. | GGUF model file |
| [`summarization/`](summarization/) | Summarize text from a string or file using a language model. | GGUF model file |
| [`translation/`](translation/) | Translate text between languages using a multilingual model. | GGUF model file |
| [`classification/`](classification/) | Text classification with grammar-constrained JSON output using `inference.WithGrammar`. | GGUF model file |
| [`vision-analysis/`](vision-analysis/) | Analyze images using a vision-capable model with `inference.Message.Images`. | Vision GGUF model + image |
| [`audio-transcription/`](audio-transcription/) | Speech-to-text using the OpenAI-compatible `/v1/audio/transcriptions` endpoint. | Whisper GGUF model + audio file |
| [`agentic-tool-use/`](agentic-tool-use/) | Function calling (tool use) with `zerfoo.WithTools` for agentic AI patterns. | GGUF model file |

### Additional Examples

| Example | Description | Prerequisites |
|---------|-------------|---------------|
| [`inference/`](inference/) | Load a GGUF model and generate text with sampling options and token streaming. | GGUF model file |
| [`streaming/`](streaming/) | Streaming chat generation using `model.ChatStream` with per-token output. | GGUF model file |
| [`embedding/`](embedding/) | Embed inference inside a custom Go HTTP handler for concurrent request serving. | GGUF model file |
| [`api-server/`](api-server/) | Start an OpenAI-compatible HTTP server with `serve.NewServer` and graceful shutdown. | GGUF model file |
| [`json-output/`](json-output/) | Grammar-guided decoding that constrains output to valid JSON matching a schema. | GGUF model file |
| [`fine-tuning/`](fine-tuning/) | LoRA fine-tuning of a tabular model: pre-train, adapt, merge, save/load. | None (synthetic data) |

## Running an Example

```bash
# Build and run the chat example
go build -o chat ./examples/chat/
./chat --model path/to/model.gguf

# Build and run the code completion example
go build -o code-completion ./examples/code-completion/
./code-completion --model path/to/model.gguf --code "func fibonacci(n int) int {"

# With GPU acceleration
./code-completion --model path/to/model.gguf --device cuda --code "func add(a, b int) int {"
```

## Further Reading

See [docs/getting-started.md](../docs/getting-started.md) for a full tutorial covering CLI usage, library API, and the OpenAI-compatible server.
