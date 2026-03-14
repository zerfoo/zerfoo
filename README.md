# Zerfoo

Go 1.25 ML inference framework with GPU acceleration. Supports GGUF (quantized)
and ZMF (full-precision) model formats. Ships an OpenAI-compatible API server.
Beats Ollama by 18% on throughput for Gemma 3 1B Q4K.

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

| Model | Format | Parameters | tok/s (DGX Spark) | CUDA Graph |
|-------|--------|-----------|------------------:|:----------:|
| Gemma 3 1B | GGUF Q4_K_M | 1B | 232.86 | Yes (99.5%) |
| Llama 3.2 1B | ZMF F32 | 1B | 17.56 | Limited |
| Qwen 2.5 | ZMF F32 | - | 7.87 | Limited |

Benchmarks measured on NVIDIA DGX Spark GB10 (sm_121, 128 GB unified memory,
LPDDR5x 273 GB/s).

## API Usage

Start the server, then send requests to the chat completions endpoint:

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"gemma3:1b","messages":[{"role":"user","content":"Hello"}]}'
```

The server implements the OpenAI chat completions API, so any OpenAI-compatible
client library works out of the box -- just point it at `localhost:8080`.

## Performance

Gemma 3 1B Q4K: 232.86 tok/s with CUDA graph capture (+26% vs no-graph).
Beats Ollama 197.21 tok/s by 18.1%.

Key optimizations:

- CUDA graph capture (99.5% of decode ops captured)
- GPU-resident tensor pipeline (zero D2H copies in the hot path)
- Fused attention kernels
- Arena memory pool

## Architecture Overview

- **Engine[T] interface** -- unified compute abstraction across CPU and GPU
  backends. All layers delegate arithmetic to Engine, enabling transparent
  hardware acceleration.
- **purego GPU bindings** -- CUDA runtime and cuBLAS loaded via dlopen at
  startup. No CGo required.
- **Graph compiler with CUDA graph capture** -- builds a static computation
  DAG, captures it as a CUDA graph for near-zero launch overhead on decode.
- **OpenAI-compatible HTTP server** -- chat completions, completions, model
  listing, and SSE streaming.

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
