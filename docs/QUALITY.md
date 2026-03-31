# Quality Gates

## Test Coverage (2026-03-05)

Minimum threshold: 75% (enforced by coverage-gate CI)

### 100% Coverage
- config, data, device, internal/xblas, layers/components, layers/registry,
  metrics, shutdown

### 98-99%
- compute (98.0%), distributed/coordinator (98.3%), features (99.0%),
  model/hrm (98.1%), numeric (98.5%), tensor (97.9%),
  testing/testutils (99.3%), tests/internal/testutil (98.5%)

### 96-97%
- graph (97.3%), inference (96.3%), layers/activations (97.4%),
  layers/attention (96.5%), layers/recurrent (97.0%),
  layers/regularization (97.6%), layers/transformer (96.4%),
  layers/transpose (97.6%), log (97.7%), metrics/runtime (96.5%),
  pkg/tokenizer (96.2%), serve (96.4%), training/loss (97.4%),
  training/optimizer (96.6%)

### 95-96%
- distributed (95.8%), generate (95.0%), layers/core (95.9%),
  layers/hrm (95.5%), layers/normalization (95.7%),
  layers/reducesum (95.9%), model (95.1%), training (95.9%)

### 90-94%
- cmd/cli (93.6%), cmd/bench-compare (89.7%), health (90.0%),
  layers/embeddings (92.9%),
  layers/gather (93.5%), registry (93.2%)

### Below 90%
- cmd/coverage-gate (84.9%), cmd/zerfoo-predict (76.6%), cmd/zerfoo-tokenize (74.1%)

### Known Untestable Gaps
- health: EngineCheck takes concrete *CPUEngine type, preventing mock testing
- Most remaining gaps are tensor.New unreachable error paths or engine error
  paths that require mock infrastructure
- layers/attention: dupl linter blocks MLA Forward engine error test
- cmd/zerfoo-predict: main(), runNewCLI (requires cli framework), os.Exit paths
- cmd/zerfoo-tokenize: main() and os.Exit paths

## Performance Benchmarks (2026-03-20, DGX Spark GB10)

### Inference Throughput -- Gemma 3 1B Q4_K_M, 256 tokens, greedy decoding

| Configuration | tok/s | Notes |
|---------------|-------|-------|
| Pre-optimization | 8.61 | Initial GPU Q4 |
| Arena allocator | 80.35 | Eliminated cudaMalloc |
| All fused kernels | 188.92 | Best without CUDA graph |
| CUDA graph capture | 244.45 | Current baseline (E103 fix) |
| Roofline (GB10, 200 GB/s) | ~257 | 95% utilization achieved |

See docs/benchmarks.md for current baselines and historical progression.

### Output Quality

| Criterion | Status |
|-----------|--------|
| Coherence | Coherent English output |
| Relevance | Topically relevant responses |
| Format | Raw completion (chat template applied via serve layer) |

### Correctness

| Check | Status |
|-------|--------|
| Output coherent English | PASS |
| No gibberish/random tokens | PASS (fixed from earlier sessions) |
| Greedy decoding deterministic | PASS (same output across runs) |
| go build ./... (no build tags) | PASS |
| go vet ./... | PASS (pre-existing purego warnings only) |
| Test suite (DGX Spark) | PARTIAL (pre-existing race in TestBatchGenerate) |

### CUDA Graph Benchmark (2026-03-20)

| Mode | Run 1 | Run 2 | Run 3 | Average |
|------|-------|-------|-------|---------|
| CUDA graph (longest-region) | 244.45 | 244.18 | 244.62 | 244.42 |
| Ollama (gemma3:1b) | 203.60 | -- | -- | 203.60 |

CUDA graph captures the longest contiguous capturable region (all transformer
layers). EmbeddingLookup and other non-capturable ops run pre/post capture.
Zerfoo is up to 18% faster than Ollama on the same hardware.

## Static Analysis (2026-03-16)

### go vet

| Repo | Result | Notes |
|------|--------|-------|
| ztoken | PASS | Zero warnings |
| ztensor | PASS* | 16 pre-existing purego unsafe.Pointer warnings only |
| zerfoo | PASS* | Same 16 purego warnings (shared internal/ packages) |

*Pre-existing purego `unsafe.Pointer` warnings are expected and intentional.
These arise from dlopen-style GPU bindings via purego and cannot be avoided
without CGo. They are safe because the pointer conversions follow the purego
calling convention for passing device pointers to GPU runtime APIs.

#### Pre-existing purego unsafe.Pointer Warnings (ztensor/zerfoo)

| File | Line(s) | Package |
|------|---------|---------|
| internal/cuda/purego_darwin.go | 91 | cuda |
| internal/cuda/runtime_purego.go | 60, 79, 94, 204 | cuda |
| internal/hip/runtime_purego.go | 50, 69, 177 | hip |
| internal/cudnn/cudnn_purego.go | 402 | cudnn |
| internal/opencl/runtime_purego.go | 199, 298, 303, 308, 313 | opencl |
| internal/tensorrt/tensorrt_purego.go | 271, 391 | tensorrt |

### golangci-lint

Not available on this machine. When available, use v2 with project .golangci.yml.

### Other Linters

- gosec: G704/G705 excluded (taint analysis false positives)

## Research Inference Quality Gates (2026-03-27, Waves 4-8)

### New Feature Quality (39 tasks, PRs #262-#265)

| Epic | Feature | Build | Vet | Tests | Status |
|------|---------|-------|-----|-------|--------|
| E35 | QuaRot weight fusion | PASS | PASS | 6 tests | Shipped |
| E35 | Quantized KV cache (Q4/Q3) | PASS | PASS | 7 tests, Q4: 7.5x reduction | Shipped |
| E36 | EAGLE speculative decoding | PASS | PASS | 19 tests | Shipped |
| E37 | Native Sparse Attention | PASS | PASS | 3+11 tests | Shipped |
| E38 | Hybrid CPU/GPU MoE | PASS | PASS | 13+9 tests | Shipped |
| E39 | BitNet ternary inference | PASS | PASS | 10 tests | Shipped |
| E40 | TransMLA conversion + inference | PASS | PASS | 12+3 tests | Shipped |
| E41 | I-Quant dequantization | PASS | PASS | 8 tests | Shipped |
| E42 | RadixAttention KV cache | PASS | PASS | 18+10 tests | Shipped |
| E43 | Flash decoding | PASS | PASS | Wired, DGX benchmark pending | Shipped |
| E44 | Multi-LoRA serving | PASS | PASS | 7+3 tests | Shipped |

### Bug Fixes

| Bug | Root Cause | Fix | PR |
|-----|-----------|-----|-----|
| BitNet GEMV never dispatched from GGUF | `decodeTernaryTensor` called `tensor.New` instead of `tensor.NewWithStorage` | Preserve TernaryStorage | #264 |
| Phi3/Llama3.1 GGUF load failure | Q2_K/Q3_K constants defined but no decode paths | Added decoders | #265 |

### Pre-existing Failures (not introduced by Waves 4-8)

| Test | Package | Status |
|------|---------|--------|
| TestGQABackward | layers/attention | FAIL (gradient mismatch, pre-existing) |
| TestTSPulseClassify | inference/timeseries | Flaky (intermittent) |
| CI arm64-build | GitHub Actions | FAIL (ztensor replace dir not available) |

## Build

- `go build ./...`: clean
- `-race` flag enabled for all tests
