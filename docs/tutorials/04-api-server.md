# Part 4: Running the OpenAI-Compatible API Server

This tutorial shows how to serve a model over HTTP using the Zerfoo API server, which implements the OpenAI API specification. Any client library or tool that works with the OpenAI API can connect to Zerfoo with a one-line base URL change.

## Starting the Server

The simplest way to start serving is with the `serve` CLI command:

```bash
zerfoo serve gemma-3-1b-q4
```

This downloads the model (if not already cached), loads it, and starts an HTTP server on `localhost:8080`. You can customize the host and port:

```bash
zerfoo serve gemma-3-1b-q4 --host 0.0.0.0 --port 3000
```

For GPU inference:

```bash
zerfoo serve gemma-3-1b-q4 --device cuda
```

## Available Endpoints

The server exposes these OpenAI-compatible endpoints:

| Method | Path | Description |
|--------|------|-------------|
| POST | `/v1/chat/completions` | Chat completion (multi-turn conversation) |
| POST | `/v1/completions` | Text completion (single prompt) |
| POST | `/v1/embeddings` | Text embeddings |
| POST | `/v1/audio/transcriptions` | Audio transcription (when a transcriber is configured) |
| GET | `/v1/models` | List loaded models |
| GET | `/v1/models/{id}` | Get model information |
| DELETE | `/v1/models/{id}` | Unload a model |
| GET | `/metrics` | Prometheus metrics |
| GET | `/openapi.yaml` | OpenAPI specification |

## Making Requests with curl

### Chat Completion

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma-3-1b-q4",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "What is the capital of France?"}
    ],
    "temperature": 0.7,
    "max_tokens": 64
  }'
```

### Streaming

Add `"stream": true` to receive server-sent events (SSE) as tokens are generated:

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma-3-1b-q4",
    "messages": [
      {"role": "user", "content": "Write a poem about Go."}
    ],
    "stream": true,
    "max_tokens": 128
  }'
```

Each SSE event contains a JSON chunk with the delta token. The stream ends with `data: [DONE]`.

### Text Completion

```bash
curl http://localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma-3-1b-q4",
    "prompt": "The Go programming language is",
    "max_tokens": 64,
    "temperature": 0.5
  }'
```

### List Models

```bash
curl http://localhost:8080/v1/models
```

## Using with the OpenAI Python Client

Any OpenAI-compatible client library works. Here is an example with the official Python client:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="not-needed",  # Zerfoo does not require an API key by default
)

response = client.chat.completions.create(
    model="gemma-3-1b-q4",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain transformers in ML."},
    ],
    temperature=0.7,
    max_tokens=256,
)

print(response.choices[0].message.content)
```

For streaming:

```python
stream = client.chat.completions.create(
    model="gemma-3-1b-q4",
    messages=[{"role": "user", "content": "Write a haiku."}],
    stream=True,
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

## Starting the Server from Go Code

You can embed the server directly in your Go application:

```go
package main

import (
	"log"
	"net/http"

	"github.com/zerfoo/zerfoo/inference"
	"github.com/zerfoo/zerfoo/serve"
)

func main() {
	model, err := inference.LoadFile("gemma-3-1b-it-q4_0.gguf",
		inference.WithDevice("cuda"),
	)
	if err != nil {
		log.Fatal(err)
	}
	defer model.Close()

	srv := serve.NewServer(model)

	log.Println("Listening on :8080")
	log.Fatal(http.ListenAndServe(":8080", srv.Handler()))
}
```

### Server Options

The `serve.NewServer` function accepts options for logging, metrics, batch scheduling, speculative decoding, and multi-GPU distribution:

```go
srv := serve.NewServer(model,
	serve.WithLogger(logger),
	serve.WithMetrics(metricsCollector),
	serve.WithBatchScheduler(batchScheduler),
	serve.WithDraftModel(draftModel),
	serve.WithGPUs([]int{0, 1}),
)
```

**Speculative decoding**: When a draft model is set with `WithDraftModel`, the server uses speculative decoding for all completion requests. A smaller, faster model proposes tokens and the target model verifies them in a single batched forward pass, improving decode throughput.

**Batch scheduling**: When a `BatchScheduler` is attached with `WithBatchScheduler`, incoming non-streaming requests are grouped into batches for higher throughput under load.

## Prometheus Metrics

The server exposes a `/metrics` endpoint in Prometheus format. Key metrics include:

- Request count and latency per endpoint
- Token generation rate (tokens per second)
- Speculative decoding acceptance rate (when enabled)

Point your Prometheus scrape config at `http://localhost:8080/metrics` to collect these metrics.

## Monitoring and Health

The `/v1/models` endpoint serves as a lightweight health check. If the model is loaded and ready, it returns model metadata. After a `DELETE /v1/models/{id}` call, the model is unloaded and subsequent inference requests return an error.

The server includes built-in recovery middleware that catches panics during request handling and returns a 500 response instead of crashing the process.

## Using with the OpenAI Go Client

You can also use any Go HTTP client. Here is an example using the standard library:

```go
package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
)

func main() {
	body := map[string]interface{}{
		"model": "gemma-3-1b-q4",
		"messages": []map[string]string{
			{"role": "user", "content": "What is Go?"},
		},
		"max_tokens": 64,
	}
	data, _ := json.Marshal(body)

	resp, err := http.Post("http://localhost:8080/v1/chat/completions",
		"application/json", bytes.NewReader(data))
	if err != nil {
		log.Fatal(err)
	}
	defer resp.Body.Close()

	out, _ := io.ReadAll(resp.Body)
	fmt.Println(string(out))
}
```

This works because the server speaks the same JSON schema as the OpenAI API. Any HTTP client in any language can send requests to Zerfoo without a dedicated SDK.

## What is Next

- [Part 5: Tabular and Time-Series ML](05-tabular-timeseries.md) -- use Zerfoo for structured data prediction and forecasting.
