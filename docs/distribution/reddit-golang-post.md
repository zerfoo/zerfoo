# Reddit r/golang Post

## Title

Zerfoo: Production-grade ML inference in pure Go -- zero CGo, 241 tok/s

## Body

I've been building an ML inference framework in Go and wanted to share it
with the community. Zerfoo lets you run transformer models (Llama 3, Gemma 3,
Mistral, Qwen 2, Phi 3/4, DeepSeek V3) directly from Go code -- no CGo, no
Python, no sidecar processes.

### Why pure Go matters

The entire framework compiles with `go build`. GPU acceleration (CUDA, ROCm,
OpenCL) is loaded dynamically at runtime via purego/dlopen, so there is no C
compiler in the build chain. If the GPU libraries are not present, it falls
back to CPU with ARM NEON SIMD assembly for the hot paths. This means
cross-compilation, static binaries, and all the deployment simplicity you
expect from Go just work.

### Library-first design

Zerfoo is a Go library first, CLI second. Embedding inference in your
application is three lines:

```go
mdl, _ := inference.Load("gemma-3-1b-q4")
defer mdl.Close()
result, _ := mdl.Generate(ctx, "Explain quicksort.", inference.WithMaxTokens(256))
```

It also ships with an OpenAI-compatible API server (`zerfoo serve`) with
streaming, batching, and speculative decoding.

### Performance

On Gemma 3 1B Q4_K_M, Zerfoo decodes at **241 tok/s** on a DGX Spark --
28% faster than Ollama on the same hardware with the same model file.
CUDA graph capture covers 99.5% of the decode path. The benchmarking
methodology, including reproduction steps, is documented here:
https://github.com/zerfoo/zerfoo/blob/main/docs/benchmarking-methodology.md

### Type-safe generics

The tensor library uses Go generics (`tensor.Numeric` constraint) for
compile-time type safety across float32, float64, float16, bfloat16, float8,
and quantized types. No `interface{}` in hot paths.

### Getting started

```bash
go install github.com/zerfoo/zerfoo/cmd/zerfoo@latest
zerfoo pull gemma-3-1b-q4
zerfoo run gemma-3-1b-q4 "The quick brown fox"
```

Repo: https://github.com/zerfoo/zerfoo

Feedback welcome -- especially on the API surface and what models you would
want supported next.
