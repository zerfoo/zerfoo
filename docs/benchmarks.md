# Zerfoo Benchmarks

All benchmarks run on DGX Spark GB10 unless noted. Greedy sampling, 128 output tokens, 3-run median.

## Throughput vs Ollama (CPU, 2026-03-27)

| Model | Quant | Zerfoo tok/s | Ollama tok/s | Ratio | Date |
|-------|-------|-------------|--------------|-------|------|
| Gemma 3 1B | Q4_K_M | 235 | 204 | **1.25x** | 2026-03-27 |
| DeepSeek-R1 1.5B | Q4_K_M | 186 | 168 | **1.11x** | 2026-03-27 |
| Llama 3.2 3B | Q4_K_M | 92 | 93 | 0.99x | 2026-03-27 |
| Mistral 7B | Q4_K_M | 44 | 44 | 1.00x | 2026-03-27 |

Full results: `results/benchmark-2026-03-27.json`

## Over-RAM Inference (2026-03-29)

| Model | Params | Quant | File Size | Shards | RAM | Load Time | Tokens | Zerfoo | Ollama |
|-------|--------|-------|-----------|--------|-----|-----------|--------|--------|--------|
| MiniMax-M2 | 229B (MoE) | Q4_K_M | 128.8 GB | 3 | 128 GB | 6.3s | ✅ 0.06 tok/s | ✅ | ❌ 500 error |

Notes:
- CPU-only (no GPU acceleration for over-RAM path yet)
- Throughput is NVMe-bound; faster storage = faster tokens
- Ollama: "unable to load model" 500 error on same hardware

## Granite Time Series vs Python granite-tsfm

Results in devlog, 2026-03-27.

## Flash Decode (GPU, pending)

Target: ≥ 1.5x speedup at seqLen_KV > 1024 (E43). To be benchmarked on DGX Spark after GPU streaming GEMM lands.
