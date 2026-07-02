# API Server Example

Start an OpenAI-compatible HTTP inference server powered by Zerfoo.

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
go build -o api-server ./examples/api-server/
```

## Run

```bash
./api-server ./models/gemma-3-1b-it-qat-q4_0.gguf
```

With a custom port:

```bash
./api-server -port 9090 ./models/gemma-3-1b-it-qat-q4_0.gguf
```

With GPU acceleration:

```bash
./api-server -device cuda ./models/gemma-3-1b-it-qat-q4_0.gguf
```

## Testing with curl

### Chat completion

```bash
curl -s http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma-3-1b-it",
    "messages": [{"role": "user", "content": "What is 2+2?"}],
    "temperature": 0.7,
    "max_tokens": 128
  }' | jq .
```

### Text completion

```bash
curl -s http://localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma-3-1b-it",
    "prompt": "The capital of France is",
    "max_tokens": 64
  }' | jq .
```

### Streaming

```bash
curl -N http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma-3-1b-it",
    "messages": [{"role": "user", "content": "Write a haiku about Go."}],
    "stream": true
  }'
```

### List models

```bash
curl -s http://localhost:8080/v1/models | jq .
```

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/v1/chat/completions` | Chat completion (OpenAI-compatible) |
| POST | `/v1/completions` | Text completion |
| POST | `/v1/embeddings` | Text embeddings |
| GET | `/v1/models` | List loaded models |
| GET | `/openapi.yaml` | OpenAPI specification |
| GET | `/metrics` | Prometheus metrics |
