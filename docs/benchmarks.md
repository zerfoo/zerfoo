# Performance Baselines

Current and historical performance measurements. Updated after each
verification run on DGX.

## Current Baselines (2026-03-15, main @ 4724c47)

| Model | Format | Tok/s | CUDA Graph % | Output Quality | Tokens | Date |
|-------|--------|-------|-------------|----------------|--------|------|
| Gemma 3 1B | GGUF Q4_K | 232.21 | 99.5% | Baseline | 256 | 2026-03-15 |
| Llama 3 1B | ZMF/ONNX | 12.93 | 2.0% | Semi-coherent | 20 | 2026-03-15 |
| Qwen 2.5 0.5B | ZMF/ONNX | 15.79 | 1.8% | Working (rep. penalty helps) | 20 | 2026-03-15 |
| Mistral 7B | ZMF/ONNX | 3.94 | 1.2% | Working (spaces fixed) | 20 | 2026-03-15 |
| Phi-3 mini | ZMF/ONNX | 4.14 | 0.5% | Semi-coherent | 20 | 2026-03-15 |

Hardware: DGX Spark GB10, sm_121, 128GB LPDDR5x, Go 1.25.0, CUDA 13.0

## Performance Milestones

| Date | Milestone | Tok/s | Notes |
|------|-----------|-------|-------|
| 2026-03-14 | Ollama parity surpassed | 234.30 | CUDA graph capture, +18.7% vs Ollama 197.21 |
| 2026-03-13 | GPU-first pipeline | 6.84 | Phase 32, +33.6% from D2H elimination |
| 2026-03-13 | Graph compilation | 6.86 | Phase 30, +5% from worker pool |
| 2026-03-12 | NEON SIMD | 8.15 | Phase 34, +18.8% CPU acceleration |
| 2026-03-12 | CPU baseline | 6.5 | Phase 29, parallelFor + xblas |
| 2026-03-11 | Initial GPU | 5.12 | Phase 31, 43% cgocall overhead |
| 2026-03-10 | Initial CPU | 3.60 | Phase 26, Gemma 3 2B Q4 |

---

## Historical Measurements

Older baselines moved here during /trim.
