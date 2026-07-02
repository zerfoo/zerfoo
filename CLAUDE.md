# Zerfoo Core Framework

See `../CLAUDE.md` for the ecosystem repo map and cross-repo rules. The parent-level `../docs/` folder is archived Aug-2025 material — current documentation lives in this repo's `docs/`.

## Product direction (H2 2026)

Strategy: **Trust, then Traction** — `docs/product-strategy-2026-H2.md` (ADR-093). `docs/plan.md` is scoped to ONE phase at a time and ends with a task to plan the next phase. New session? Read in this order: product-strategy-2026-H2.md → plan.md → devlog.md (newest entry) → ADR-091/092/093.

## Mission

Be the best-in-class ML inference framework in Go — competitive with C++ runtimes (llama.cpp, vLLM) in throughput, superior in developer experience and embeddability. GGUF is the sole model format.

## Zerfoo stays general-purpose

Zerfoo is a general-purpose ML framework. Wolf (feza-ai/wolf) is its most demanding consumer and its hardest stress test — but Wolf is a **workload, not the spec**. Rules:

- **Fix at the contract level, not the consumer level.** When a consumer workload exposes a bug, the fix is a general framework guarantee (e.g. the SaveForBackward lifetime contract, the dst-form "ops write into dst's storage" contract, persistent gradient accumulators), never a special case for that consumer's call pattern.
- **No consumer-specific logic in this repo.** Model-, strategy-, or pipeline-specific code (e.g. Wolf's QK-norm nodes, divergence guards, feature pipelines) lives in the consumer's repo. Upstream only mechanisms that any ML workload could use.
- **Acceptance must not be single-consumer.** "Wolf passes" is necessary but not sufficient for framework-level changes; validation should include at least one non-Wolf path (e.g. a timeseries/patchtst training smoke). Known single-consumer deferrals must be tracked as issues (e.g. the capture-mode hole in ztensor's dst-copy path, the #847 follow-up layer migrations), not silently dropped because Wolf doesn't exercise them.
- **Quality gates are universal.** Every op — used by Wolf or not — must pass gradcheck/OpInfo, the GPU/CPU parity harness, and (for CUDA kernels) the PyTorch-oracle gate before merge. Prefer neutral naming for tests and fixtures (a hazard pattern is "accumulate-across-resets", not a consumer's name).

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

**GPU validation (standing gate):** `scripts/dgx-validate.sh` is the one command for GPU-dependent acceptance — it builds zerfoo natively on the GB10 (purego cannot cross-compile darwin→linux/arm64) inside a Spark pod, runs cuda-tagged tests + the parity subset, and exits green/red on a JSON report. Use `-dry-run` to inspect without submitting. One GPU pod at a time; record run results in docs/devlog.md.

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
