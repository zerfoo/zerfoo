# Embedding Example

Embed Zerfoo inference directly inside a Go HTTP handler. This pattern is useful when you want to add ML inference to an existing Go service without running a separate server process.

## Prerequisites

- Go 1.25+
- A GGUF model file (e.g., Gemma 3 1B or Llama 3.2 1B)

### Downloading a test model

```bash
pip install huggingface-hub

huggingface-cli download google/gemma-3-1b-it-qat-q4_0-gguf \
  --local-dir ./models
```

## Build

```bash
go build -o embedding ./examples/embedding/
```

## Run

```bash
./embedding ./models/gemma-3-1b-it-qat-q4_0.gguf
```

With a custom port and GPU:

```bash
./embedding -port 9090 -device cuda ./models/gemma-3-1b-it-qat-q4_0.gguf
```

## Testing with curl

### Generate text

```bash
curl -s http://localhost:8080/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Explain what a tensor is in one sentence.",
    "max_tokens": 128,
    "temperature": 0.7
  }' | jq .
```

### Health check

```bash
curl http://localhost:8080/health
```

## How it works

The model is loaded once at startup. Each incoming HTTP request calls `model.Generate()` with the provided prompt and options. This is the simplest way to add inference to any Go application — just import `github.com/zerfoo/zerfoo/inference` and call `LoadFile` / `Generate`.
