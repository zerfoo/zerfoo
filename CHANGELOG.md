# Changelog — dndungu/zerfoo Fork

Summary of 215 commits implemented on the dndungu/zerfoo fork, organized by phase.
Original branch preserved at tag/branch `backup-main-before-rebase` (commit `19b5820`).
Common ancestor with zerfoo/zerfoo: commit `2093b01` (UINT8 tensor support).

---

## Phase 1 — Architecture & Cleanup

Refactored the codebase from a Numerai/audacity-specific tool into a generic ML framework.

- **Architecture analysis**: cross-package dependency analysis, API design, migration plan
- **Numerai removal**: removed era-specific training code, Numerai references from training/model/data
- **Generic CLI framework**: plugin-based CLI with predict/tokenize commands
- **Generic data types**: replaced Numerai-specific types with generic `data.Sample`/`data.Batch`
- **Model adapter stubs**: round-trip model adapter tests
- **Audacity removal**: removed all audacity references from CLI, training, integration tests
- **Lint & formatting**: resolved critical linting issues, normalized formatting across codebase

## Phase 2 — Core ML Fixes & Model Serialization

Fixed fundamental correctness issues in the ML pipeline.

- **Binary tensor serialization**: replaced lossy `serializeTensorData` with binary `EncodeTensor`
- **RoPE embeddings**: fixed backward pass shape derivation from `dOut` tensor
- **RMSNorm**: reshaped gain gradient to match parameter shape
- **GQA attention**: rewrote backward with correct head-replication reversal
- **Transformer Block**: implemented `Block.Backward` with gradient tests
- **HRM module integration**: implemented `graph.Node` interface on `HModule`/`LModule`
- **ZMF parameter restore**: fixed Linear, SimpleRNN, and S4 parameter restoration from ZMF

## Phase 3 — Test Coverage (0% → 95%+ across packages)

Massive test coverage improvement across the entire codebase:

| Package | Before | After |
|---------|--------|-------|
| internal/xblas | 0% | 100% |
| layers/registry | 0% | 100% |
| pkg/tokenizer | 0% | 100% |
| layers/tokenizers | 66.7% | 100% |
| tensor | 58.5% | 98.9% |
| numeric | — | 98.5% |
| graph | 66.3% | 97.0% |
| layers/transpose | 66.7% | 97.2% |
| layers/activations | 59.0% | 97.1% |
| layers/recurrent | 62.3% | 96.7% |
| cmd/cli | 27.9% | 96.5% |
| layers/core | 56.1% | 96.0% |
| layers/reducesum | 0% | 95.9% |
| training | 27.5% | 95.7% |
| layers/attention | 52.3% | 95.1% |
| model | 47.5% | 95.4% |
| layers/gather | 0% | 91.7% |
| layers/normalization | 42.3% | 69.7% |

Additional error-path tests for: optimizer, distributed, embeddings, transformer, loss, components, features, coordinator, HRM, compute, data, gather.

## Phase 4 — GPU Engine (CUDA)

Full CUDA GPU compute engine implementation:

- **CUDA runtime bindings**: CGO bindings for CUDA runtime (alloc, memcpy, stream, device)
- **Device management**: CUDA allocator, device registration
- **GPU tensor storage**: `GPUStorage[T]` with `ToGPU`/`ToCPU` transfer, `Storage[T]` interface
- **cuBLAS integration**: CGO bindings with `Sgemm`, `GPUEngine[T]` with cuBLAS MatMul
- **Native CUDA kernels**: elementwise ops, Softmax with shared-memory reduction, SumAxis reduction
- **Stream-based execution**: CUDA stream support, async memcpy, size-bucketed memory pool
- **Device-resident pipeline**: `NewWithStorage`, `NewGPUStorageFromPtr` for zero-copy GPU ops
- **GPU integration tests**: linear layer forward/backward, chained ops, mixed storage

## Phase 5 — Production Infrastructure

Enterprise-grade production readiness features:

- **Structured logging**: leveled logger with package-wide integration
- **Configuration**: generic JSON config loader with env overrides and validation; Engine/Training/Distributed configs
- **Health checks**: HTTP health server with liveness/readiness probes, pprof endpoints
- **Graceful shutdown**: `Closer` interface and orderly shutdown coordinator
- **Signal handling**: SIGINT/SIGTERM handling wired into CLI
- **Metrics instrumentation**: counters and duration histograms for CPUEngine and AllReduceStrategy
- **TLS/mTLS**: TLS config with server/client credentials for distributed communication
- **CI pipeline**: coverage gate (93% threshold), benchmark regression detection (10% threshold), race detector
- **Distributed training**: gRPC-based AllReduce/Barrier/Broadcast with input validation, worker node lifecycle
- **Deployment docs**: runbook with config reference, troubleshooting guide

## Phase 6 — Open Weights Model Import (ONNX/ZMF)

Support for importing and running open-weight models:

- **Extended tensor decoding**: BFLOAT16, INT32, INT64, FLOAT64, UINT8 in `DecodeTensor`
- **Constant node handling**: ZMF constant nodes and `Attribute_Tensor` in model builder
- **New layers**: Softmax, Erf, Sigmoid, LayerNormalization, Slice, Pad, TopK
- **Vision layers**: Conv2d, GlobalAveragePool, Resize, BatchNormalization
- **MoE layers**: MoEGate, MixtureOfExperts with error-path tests
- **Layer registry**: all new layers registered in central registry
- **Parity tests**: Gemma 3 and SigLIP/Kimi-VL forward pass parity tests

## Phase 7 — Architecture Cleanup

Final cleanup and consolidation:

- **Doc consolidation**: merged gpu.md, runbook.md, troubleshooting.md into design.md
- **Dead code removal**: deleted empty `pkg/prelude`, dead test files (parity, numerics, wire stubs)
- **Exported builders**: exported `BuildFFN`, removed `init()` registration
- **Thread safety**: added `sync.Mutex` to protect memo map in graph `Forward`/`Backward`
- **FFN registration**: added FFN to `RegisterAll`

## New Layers Added

- S4 (diagonal state space model)
- Dropout (inverted)
- Softmax, Erf, Sigmoid
- LayerNormalization
- Slice, Pad, TopK
- Conv2d, GlobalAveragePool, Resize
- BatchNormalization
- MoEGate, MixtureOfExperts
- FFN (feed-forward network)

## New Packages/Commands

- `log/` — structured leveled logger
- `config/` — generic config loader
- `health/` — HTTP health server
- `shutdown/` — graceful shutdown coordinator
- `device/` — CUDA device management
- `cmd/bench-compare/` — benchmark regression detection
- `cmd/coverage-gate/` — per-package coverage enforcement
- `distributed/` — gRPC AllReduce strategy, worker node, TLS
