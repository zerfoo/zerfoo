# Performance Baselines

Current and historical performance measurements. Updated after each
verification run on DGX.

## Current Baselines (2026-03-19, main @ b81b616)

| Model | Format | Tok/s | CUDA Graph | Tokens | Date | Commit | Notes |
|-------|--------|-------|------------|--------|------|--------|-------|
| Gemma 3 1B | Q4_K_M | 136.30 | Yes | 256 | 2026-03-19 | b81b616 | Regression from 245 — bisecting |
| Gemma 3 1B | Q4_K_M | 229.45 | Yes | 256 | 2026-03-19 | 4e85b12 (old binary) | Old code, same kernels |

**Regression alert:** Current HEAD achieves ~136 tok/s vs ~229 tok/s at commit 4e85b12.
The regression is in Go code (generate/ or inference/), not in CUDA kernels.
Roofline analysis: GB10 LPDDR5x ~200 GB/s → max ~257 tok/s for 778 MB model.
500 tok/s target requires hardware with higher memory bandwidth.

## Previous Baselines (2026-03-17, ztensor @ 4e85b12)

| Model | Format | Tok/s | CUDA Graph | Tokens | Date | Commit |
|-------|--------|-------|------------|--------|------|--------|
| Gemma 3 1B | Q4_K_M | 219.17 | Yes | 50 | 2026-03-17 | 4e85b12 |
| Gemma 3 1B | Q4_K_M | 245.15 | Yes | 256 | 2026-03-17 | 4e85b12 |
| Gemma 3 1B | Q4_K_M | 248.47 | Yes | 512 | 2026-03-17 | 4e85b12 |
| Gemma 3 1B | Q4_K_M | 174.44 | No | 256 | 2026-03-17 | 8717a12 |
| Ollama gemma3:1b | - | 203.60 | - | 989 | 2026-03-17 | - |

Zerfoo vs Ollama: +20% at 256 tokens with CUDA graphs.

Note: dp4a INT8 GEMV (T4.1) and arena free-list reuse (T4.2) merged into ztensor.
At batch=1 decode, throughput is memory-bandwidth-bound so dp4a shows parity as expected.
dp4a benefits will appear at larger batch sizes where compute becomes the bottleneck.

### Previous Baselines (2026-03-16, main @ a5c54c3)

| Model | Format | Tok/s | CUDA Graph % | Output Quality | Tokens | Date |
|-------|--------|-------|-------------|----------------|--------|------|
| Gemma 3 1B | GGUF Q4_0 | 103.22 (CUDA) / 9.20 (CPU) | 99.5% | Coherent | 50 | 2026-03-16 |
| Gemma 3 1B FP16 | GGUF Q4_0 + FP16 | 9.20 (CPU) | - | Valid | 30 | 2026-03-16 |
| Gemma 3 1B FP8 | GGUF Q4_0 + FP8 | ~0.5 (CPU) | - | Valid | 30 | 2026-03-16 |
| TinyLlama 1.1B | GGUF Q4_K_M | 7.18 (CPU) | - | Low (small model) | 50 | 2026-03-16 |
| Qwen 2.5 0.5B | GGUF Q4_K_M | ~13 (CPU) | - | Garbled (tokenizer bug) | 50 | 2026-03-16 |
| Mistral 7B | GGUF Q4_K_M | ~0.55 (CPU) | - | Low (loads as llama) | 50 | 2026-03-16 |
| Phi-3.5 mini | GGUF Q4_K_M | - | - | FAIL (merged QKV) | - | 2026-03-16 |

### Previous Baselines (2026-03-15, main @ 4724c47)

| Model | Format | Tok/s | CUDA Graph % | Output Quality | Tokens | Date |
|-------|--------|-------|-------------|----------------|--------|------|
| Gemma 3 1B | GGUF Q4_K | 232.21 | 99.5% | Baseline | 256 | 2026-03-15 |
| Llama 3 1B | GGUF | 12.93 | 2.0% | Semi-coherent | 20 | 2026-03-15 |
| Qwen 2.5 0.5B | GGUF | 15.79 | 1.8% | Working (rep. penalty helps) | 20 | 2026-03-15 |
| Mistral 7B | GGUF | 3.94 | 1.2% | Working (spaces fixed) | 20 | 2026-03-15 |
| Phi-3 mini | GGUF | 4.14 | 0.5% | Semi-coherent | 20 | 2026-03-15 |

Hardware: DGX Spark GB10, sm_121, 128GB LPDDR5x, Go 1.25.0, CUDA 13.0

## Vision Models

Vision model benchmarks use synthetic weights with small dimensions for CI,
and full GGUF models for hardware throughput validation.

| Model | Test | Status | Env Var |
|-------|------|--------|---------|
| LLaVA | BenchmarkLLaVA_Throughput | Synthetic (CI) | - |
| LLaVA | TestLLaVA_VisionPipeline | Full model | LLAVA_GGUF_PATH |
| Qwen-VL | BenchmarkQwenVL_Throughput | Synthetic (CI) | - |
| Qwen-VL | TestQwenVL_VisionPipeline | Full model | QWENVL_GGUF_PATH |

Run synthetic benchmarks:
```bash
go test -bench BenchmarkLLaVA -count=1 ./tests/parity/
go test -bench BenchmarkQwenVL -count=1 ./tests/parity/
```

Run full-model vision pipeline tests (requires GGUF files):
```bash
LLAVA_GGUF_PATH=/path/to/llava.gguf go test -run TestLLaVA_VisionPipeline -count=1 -v ./tests/parity/
QWENVL_GGUF_PATH=/path/to/qwenvl.gguf go test -run TestQwenVL_VisionPipeline -count=1 -v ./tests/parity/
```

## Performance Milestones

| Date | Milestone | Tok/s | Notes |
|------|-----------|-------|-------|
| 2026-03-17 | dp4a + arena reuse (T4.1+T4.2) | 245.15 | Parity at batch=1 (memory-bound); dp4a benefits at larger batches |
| 2026-03-17 | Q4_0 re-quant restored | 244.99 | +32% vs regression, +20% vs Ollama |
| 2026-03-14 | CUDA graph capture | 234.30 | +26% vs non-graph baseline |
| 2026-03-13 | GPU-first pipeline | 6.84 | Phase 32, +33.6% from D2H elimination |
| 2026-03-13 | Graph compilation | 6.86 | Phase 30, +5% from worker pool |
| 2026-03-12 | NEON SIMD | 8.15 | Phase 34, +18.8% CPU acceleration |
| 2026-03-12 | CPU baseline | 6.5 | Phase 29, parallelFor + xblas |
| 2026-03-11 | Initial GPU | 5.12 | Phase 31, 43% cgocall overhead |
| 2026-03-10 | Initial CPU | 3.60 | Phase 26, Gemma 3 2B Q4 |

---

## Historical Measurements

Older baselines moved here during /trim.
