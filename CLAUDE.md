# Zerfoo Core Framework

See `../CLAUDE.md` for the full ecosystem vision and project map (covers 7 repos).

## Mission

Be the best-in-class ML inference framework in Go — competitive with C++ runtimes (llama.cpp, vLLM) in throughput, superior in developer experience and embeddability. GGUF is the sole model format.

## Hardware

DGX Spark GPU available at `ssh ndungu@192.168.86.29`

## Build & Test

```bash
go test ./...                           # CPU tests (no GPU required)
go test -tags cuda ./...                # With CUDA (on DGX Spark)
go test -run TestParity -count=1 ./tests/parity/...  # Model parity tests (require model files)
```

## Key Conventions

- **Engine[T] is law**: All tensor arithmetic flows through `compute.Engine[T]`. Never operate on raw slices outside the engine — this enables transparent CPU/GPU switching and CUDA graph capture.
- **No CGo by default**: GPU bindings use purego/dlopen. Build tags (`cuda`, `rocm`, `opencl`) are optional and only used for CGo-based alternative paths.
- **GGUF is the sole model format**: GGUF is the only supported model format. ZMF has been removed. This package imports `github.com/zerfoo/ztensor` for tensor/compute/graph and `github.com/zerfoo/ztoken` for tokenizer.
- **Generics throughout**: Use `[T tensor.Numeric]` constraints. Don't write float32-specific code where generics work.
- **Fuse, don't fragment**: Prefer fused operations (FusedAddRMSNorm, FusedSiluGate, etc.) over sequences of primitive ops. Every eliminated kernel launch matters for tok/s.
- **Rebase and merge**: Not squash, not merge commits.

## Architecture Quick Reference

```
cmd/           → CLI (run, serve, pull, predict, tokenize, worker)
serve/         → OpenAI-compatible API server
inference/     → Model loading (GGUF), architecture graph builders
generate/      → Autoregressive decoding, KV cache, sampling, streaming
layers/        → 56+ ops: attention, normalization, activations, core math
model/         → Model abstraction, GGUF parser
training/      → Trainer, optimizers, loss functions, gradient strategies
distributed/   → gRPC/NCCL distributed training
internal/cuda/ → CUDA bindings (purego), memory arena, 25+ custom kernels
internal/xblas/→ ARM NEON + AVX2 SIMD assembly
internal/codegen/ → Megakernel code generator
internal/gpuapi/  → GPU runtime abstraction layer (GRAL)

# External packages (separate repos):
# Tensor, compute, graph → github.com/zerfoo/ztensor
# Tokenizer → github.com/zerfoo/ztoken
```
