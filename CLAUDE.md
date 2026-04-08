# Zerfoo Core Framework

See `../CLAUDE.md` for the full ecosystem vision and project map (covers 7 repos).

## Mission

Be the best-in-class ML inference framework in Go — competitive with C++ runtimes (llama.cpp, vLLM) in throughput, superior in developer experience and embeddability. GGUF is the sole model format.

## Hardware

DGX Spark GPU host is at `192.168.86.250` (NVIDIA GB10, aarch64, unified memory).

**Benchmarks MUST run via Spark, not interactive SSH.** Interactive `ssh dgx 'bench_train ...'` calls from the Claude Code bash tool leak SSH channels and have taken the host down (2026-04-07 outage, see `docs/plans/spark-bench-runner.md`). The host runs Spark v1.6.0+ as a single-node Podman-backed pod orchestrator; submit benches as Kubernetes Pod manifests via the Spark HTTP API on `:8080`.

- Submit a bench: `scripts/bench-spark.sh -samples N -channels C -epochs E` (wraps POST to `http://192.168.86.250:8080/api/v1/pods` with `docs/bench/manifests/patchtst-train.yaml`).
- Inspect: `curl http://192.168.86.250:8080/api/v1/pods/<name>`
- Stream logs: `curl http://192.168.86.250:8080/api/v1/pods/<name>/logs`
- Kill a runaway bench: `curl -X DELETE http://192.168.86.250:8080/api/v1/pods/<name>`

The manifest enforces cgroup limits (`memory: 32Gi`, `cpu: "8"`, `nvidia.com/gpu: 1`) via Podman, so a runaway bench OOM-kills inside the container instead of taking down the host. Podman cannot cgroup-cap VRAM on GB10 (no MIG); VRAM contention is serialized via `SPARK_GPU_MAX=1` (one GPU pod at a time on the host).

**Interactive shell on DGX** is still available for debugging (`ssh ndungu@192.168.86.250`), but **do not run benchmarks through it** — that means no `go test -bench`, no `bench_train`, no `go run ./cmd/bench_*`, no `go test -tags cuda` that touches GPU kernels. Anything that loops for more than ~10 seconds goes through Spark.

ADR: `docs/adr/083-spark-bench-runner.md` (pending) for rationale and alternatives evaluated.

## Build & Test

```bash
go test ./...                           # CPU tests (no GPU required, runs anywhere)
go test -run TestParity -count=1 ./tests/parity/...  # Model parity tests (require model files)
```

GPU tests (`-tags cuda`, `bench_train`, and any benchmark that actually touches CUDA kernels) must run on DGX via Spark — see the Hardware section above. Do not run them via interactive SSH.

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
