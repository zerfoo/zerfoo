# Add ML Inference to Your Go Service in 10 Lines

*How to go from zero to LLM inference in a Go application without changing your build, your deployment, or your architecture.*

## The Problem

You have a Go service. You want it to generate text, summarize content, or answer questions using an LLM. Your options today:

1. Call the OpenAI API (or Anthropic, or Google). You now depend on an external service for latency, cost, uptime, and data privacy.
2. Run Ollama as a sidecar. You now have two processes, an HTTP boundary between them, and a separate deployment artifact to manage.
3. Wrap a C++ runtime via CGo. You now need a C compiler toolchain, lose cross-compilation, and debug across language boundaries.

All three add operational complexity that has nothing to do with your actual problem.

## The 10-Line Version

With Zerfoo, ML inference is a library call:

```go
package main

import (
	"context"
	"fmt"
	"log"

	"github.com/zerfoo/zerfoo/inference"
)

func main() {
	model, err := inference.Load("gemma-3-1b-q4")
	if err != nil {
		log.Fatal(err)
	}
	defer model.Close()

	result, err := model.Generate(context.Background(), "Explain quicksort in one paragraph.",
		inference.WithMaxTokens(256),
	)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(result)
}
```

`inference.Load` downloads the model from HuggingFace on first use and caches it locally. If CUDA is installed, the GPU is used automatically. If not, inference runs on CPU. The binary is the same either way -- `go build` is the only build step.

The distilled version, if you strip the error handling and imports:

```go
model, _ := inference.Load("gemma-3-1b-q4")
defer model.Close()
result, _ := model.Generate(context.Background(), "Your prompt here")
fmt.Println(result)
```

Four lines. Load, generate, print.

## Embedding in a Web Server

The real power is embedding inference into an existing service. Load the model once at startup, then serve requests through your own handlers:

```go
func main() {
	model, err := inference.LoadFile("/path/to/model.gguf",
		inference.WithDevice("cuda"),
	)
	if err != nil {
		log.Fatal(err)
	}
	defer model.Close()

	mux := http.NewServeMux()
	mux.HandleFunc("POST /generate", handleGenerate(model))
	mux.HandleFunc("GET /health", func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusOK)
	})

	log.Fatal(http.ListenAndServe(":8080", mux))
}

func handleGenerate(model *inference.Model) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		var req struct {
			Prompt string `json:"prompt"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, "invalid request", http.StatusBadRequest)
			return
		}

		result, err := model.Generate(context.Background(), req.Prompt,
			inference.WithMaxTokens(256),
		)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]string{"text": result})
	}
}
```

This is a complete Go HTTP server with ML inference. It compiles with `go build`, produces a single binary, and runs anywhere. The model is loaded once and shared across all requests.

The full version of this pattern is in [examples/embedding/](../../examples/embedding/).

## Comparison: Three Approaches

Here is what each approach looks like in practice.

### Calling an External API

```go
resp, err := openaiClient.CreateChatCompletion(ctx, openai.ChatCompletionRequest{
	Model: "gpt-4",
	Messages: []openai.ChatCompletionMessage{
		{Role: "user", Content: prompt},
	},
})
```

- Network round-trip on every call (50-500ms latency added)
- Per-token cost ($0.01-0.06 per 1K tokens)
- Your data leaves your infrastructure
- Service goes down, your feature goes down

### Running Ollama as a Sidecar

```go
resp, err := http.Post("http://localhost:11434/api/generate",
	"application/json",
	strings.NewReader(`{"model":"gemma:2b","prompt":"`+prompt+`"}`))
```

- Separate process to deploy and monitor
- HTTP serialization overhead on every request
- Two containers, two health checks, two sets of resource limits
- Ollama's own abstraction layer between you and the model

### Embedding Zerfoo

```go
result, err := model.Generate(ctx, prompt, inference.WithMaxTokens(256))
```

- In-process function call. No network, no serialization.
- Single binary deployment. `go build` and copy.
- Your data never leaves the process.
- You control the model, the sampling parameters, and the lifecycle.

The trade-off is memory: you are loading the model into your process. A 1B parameter model in Q4 quantization uses roughly 700MB of RAM (or VRAM if using GPU). This is the cost of eliminating the external dependency.

## The API Server Mode

Sometimes you do want the OpenAI-compatible API -- for example, when serving multiple clients or when you want to use existing OpenAI client libraries. Zerfoo has a built-in server for that:

```go
model, err := inference.LoadFile(modelPath, inference.WithDevice("cuda"))
if err != nil {
	log.Fatal(err)
}
defer model.Close()

srv := serve.NewServer(model)
http.ListenAndServe(":8080", srv.Handler())
```

This gives you `/v1/chat/completions`, `/v1/completions`, `/v1/models`, Prometheus metrics at `/metrics`, and SSE streaming -- all compatible with OpenAI client libraries in any language.

The full example with graceful shutdown is in [examples/api-server/](../../examples/api-server/).

```bash
# Test it with curl
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"gemma-3-1b-q4","messages":[{"role":"user","content":"Hello!"}]}'

# Or with the OpenAI Python client
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8080/v1", api_key="not-needed")
```

## Production Considerations

**Model loading time.** The first call to `inference.LoadFile` reads the GGUF file into memory. Zerfoo uses `mmap` on Linux, so the OS pages in data on demand rather than reading the entire file upfront. Cold start for a 1B Q4 model is under 2 seconds on NVMe storage.

**Memory usage.** Quantization is your primary lever. Q4_K_M reduces a 1B parameter model from ~4GB (FP32) to ~700MB with minimal quality loss. The KV cache grows with context length -- budget roughly 2MB per 1K tokens for a 1B model.

**GPU acceleration.** If CUDA is installed, `inference.LoadFile` with `inference.WithDevice("cuda")` loads weights directly to GPU memory. No build flags, no CGo, no CUDA toolkit at build time. The GPU path is loaded dynamically via `dlopen` at runtime. See [Zero CGo: Why We Chose Pure Go for ML Inference](zero-cgo-pure-go-ml-inference.md) for the technical details.

**Graceful shutdown.** When your service receives SIGTERM, close the model before exiting to free GPU memory:

```go
ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGTERM)
defer stop()

<-ctx.Done()
model.Close() // frees GPU memory, closes mmap'd files
```

**Concurrency.** `model.Generate` is safe to call from multiple goroutines. The model holds a mutex over the KV cache, so requests are serialized at the generation level. For higher throughput, run multiple model instances across GPUs with `inference.WithDevice("cuda:0")`, `inference.WithDevice("cuda:1")`.

## Conclusion

ML inference in Go should be as simple as importing a package. No Python sidecar, no CGo build complexity, no external API dependency. Load a model, call `Generate`, get text back.

```bash
go get github.com/zerfoo/zerfoo
```

That is the entire build change.

For a full walkthrough covering CLI usage, the library API, and the OpenAI-compatible server, see [Getting Started](../getting-started.md). For runnable code, see the [examples](../../examples/).
