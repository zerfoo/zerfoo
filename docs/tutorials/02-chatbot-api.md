# Tutorial 2: Building a Chatbot API

This tutorial shows how to expose a Zerfoo model as an OpenAI-compatible HTTP API using the `serve` package. Any client that speaks the OpenAI chat completions protocol — `curl`, the OpenAI Go SDK, LangChain, etc. — works out of the box.

## Prerequisites

Complete [Tutorial 1](01-getting-started.md) and ensure you have a model cached locally.

## Step 1: Create the Server

```bash
mkdir chatbot-api && cd chatbot-api
go mod init example.com/chatbot-api
go get github.com/zerfoo/zerfoo@latest
```

Create `main.go`:

```go
package main

import (
	"context"
	"log"
	"net/http"

	"github.com/zerfoo/zerfoo/inference"
	"github.com/zerfoo/zerfoo/serve"
)

func main() {
	// Load the model (pulled from HuggingFace on first run).
	mdl, err := inference.Load("gemma-3-1b-q4")
	if err != nil {
		log.Fatal("load model:", err)
	}

	// Create an OpenAI-compatible server.
	srv := serve.NewServer(mdl)

	log.Println("Listening on :8080")
	log.Fatal(http.ListenAndServe(":8080", srv.Handler()))
}
```

Run it:

```bash
go run .
# 2026/03/18 10:00:00 Listening on :8080
```

## Step 2: Send a Chat Request

The server exposes `POST /v1/chat/completions` with the same request/response schema as the OpenAI API.

```bash
curl -s http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma-3-1b-q4",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "What is the Go programming language?"}
    ],
    "max_tokens": 200
  }' | jq .choices[0].message.content
```

## Step 3: Streaming Responses

Set `"stream": true` to receive server-sent events (SSE):

```bash
curl -N http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma-3-1b-q4",
    "messages": [{"role": "user", "content": "Count to five."}],
    "stream": true
  }'
```

Each SSE event is a `data: {...}` line with a partial token delta, ending with `data: [DONE]`.

## Step 4: Add Speculative Decoding

Speculative decoding uses a small draft model to propose tokens that the main model verifies in parallel, increasing throughput by 2-3x on capable hardware:

```go
draft, err := inference.Load("gemma-3-1b-q4")
if err != nil {
	log.Fatal("load draft model:", err)
}

target, err := inference.Load("gemma-3-8b-q4") // or any larger model
if err != nil {
	log.Fatal("load target model:", err)
}

srv := serve.NewServer(target,
	serve.WithDraftModel(draft),
)
```

## Step 5: Enable Request Batching

For higher throughput under concurrent load, attach a `BatchScheduler`:

```go
package main

import (
	"log"
	"net/http"

	"github.com/zerfoo/zerfoo/inference"
	"github.com/zerfoo/zerfoo/serve"
)

func main() {
	mdl, err := inference.Load("gemma-3-1b-q4")
	if err != nil {
		log.Fatal(err)
	}

	// Batch up to 8 requests with a 20ms collection window.
	batcher := serve.NewBatchScheduler(serve.BatchSchedulerConfig{
		MaxBatchSize: 8,
		MaxWaitMs:    20,
	})

	srv := serve.NewServer(mdl,
		serve.WithBatchScheduler(batcher),
	)

	log.Fatal(http.ListenAndServe(":8080", srv.Handler()))
}
```

## Step 6: List Available Models

```bash
curl http://localhost:8080/v1/models | jq .
```

## Available Endpoints

| Endpoint | Description |
|----------|-------------|
| `POST /v1/chat/completions` | Chat completion (streaming or non-streaming) |
| `POST /v1/completions` | Raw text completion |
| `POST /v1/embeddings` | Token embeddings |
| `GET /v1/models` | List loaded models |
| `GET /v1/models/{id}` | Model info |
| `GET /metrics` | Prometheus metrics |
| `GET /openapi.yaml` | OpenAPI spec |

## Using the OpenAI Go Client

Because the API is OpenAI-compatible, you can point any OpenAI SDK at your local server:

```go
package main

import (
	"context"
	"fmt"
	"log"

	openai "github.com/sashabaranov/go-openai"
)

func main() {
	cfg := openai.DefaultConfig("ignored-key")
	cfg.BaseURL = "http://localhost:8080/v1"
	client := openai.NewClientWithConfig(cfg)

	resp, err := client.CreateChatCompletion(context.Background(),
		openai.ChatCompletionRequest{
			Model: "gemma-3-1b-q4",
			Messages: []openai.ChatCompletionMessage{
				{Role: "user", Content: "Explain goroutines in one paragraph."},
			},
		},
	)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(resp.Choices[0].Message.Content)
}
```

## Next Steps

- [Tutorial 3: Tabular ML](03-tabular-ml.md) — train an MLP on structured data with `tabular.Train`
- [Tutorial 4: Time-Series Forecasting](04-timeseries-forecasting.md) — forecast with TFT, N-BEATS, and PatchTST
