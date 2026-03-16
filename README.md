# Zerfoo

A Go ML framework built for inference performance. Pure Go with no CGo --
GPU acceleration via runtime-loaded CUDA, ROCm, and OpenCL. Ships an
OpenAI-compatible API server, quantized model support (GGUF Q4_K), and
CUDA graph capture for near-zero kernel launch overhead.

## Installation

Prerequisites:

- Go 1.25 or later
- CUDA toolkit (for GPU acceleration; optional for CPU-only usage)

```bash
git clone https://github.com/zerfoo/zerfoo.git
cd zerfoo
go build ./...
```

For GPU support, build the CUDA kernels first:

```bash
cd internal/cuda/kernels && make shared && cd ../../../
```

## Quickstart

Pull a model and run inference:

```bash
go run ./cmd/zerfoo pull gemma3:1b
go run ./cmd/zerfoo run gemma3:1b "The quick brown fox"
```

Or start the API server:

```bash
go run ./cmd/zerfoo serve --model gemma3:1b --port 8080
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

See [docs/benchmarks.md](docs/benchmarks.md) for current throughput numbers.

## API Usage

Start the server, then send requests to the chat completions endpoint:

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"gemma3:1b","messages":[{"role":"user","content":"Hello"}]}'
```

The server implements the OpenAI chat completions API, so any OpenAI-compatible
client library works out of the box -- just point it at `localhost:8080`.

## Key Design Decisions

- **Engine[T] interface** -- unified compute abstraction across CPU and GPU
  backends. All layers delegate arithmetic to Engine, enabling transparent
  hardware acceleration.
- **purego GPU bindings** -- CUDA, ROCm, and OpenCL loaded via dlopen at
  runtime. No CGo, no build tags. `go build ./...` works everywhere.
- **Graph compiler with CUDA graph capture** -- builds a static computation
  DAG, captures it as a CUDA graph for near-zero launch overhead on decode.
- **Arena memory allocator** -- pre-allocated bump-pointer arena serves all
  inference allocations with O(1) reset per token.
- **OpenAI-compatible HTTP server** -- chat completions, completions,
  embeddings, model management, and SSE streaming.

See [docs/design.md](docs/design.md) for the full architecture.

## Contributing

Standard Go workflow: fork, branch, test, PR.

The pre-commit hook enforces single-directory commits. Run tests before
submitting:

```bash
go test ./... -race
```

## License

Apache 2.0
