# Zerfoo Benchmarks

All benchmarks run on DGX Spark GB10 unless noted. Greedy sampling, 128 output tokens, 3-run median.

## Throughput vs Ollama (GPU, 2026-03-30, v1.38.4)

| Model | Quant | Zerfoo tok/s | Ollama tok/s | Ratio | Date |
|-------|-------|-------------|--------------|-------|------|
| Gemma 3 1B | Q4_K_M | 241 | 188 | **1.28x** | 2026-03-31 |
| DeepSeek-R1 1.5B | Q4_K_M | 186 | 168 | **1.11x** | 2026-03-30 |
| Llama 3.2 3B | Q4_K_M | 92 | 93 | 0.99x | 2026-03-30 |
| Mistral 7B | Q4_K_M | 44 | 44 | 1.00x | 2026-03-30 |

CUDA graph capture: 184/185 instructions (99.5%). Fused kernels: softmax+V multiply, repeat-interleave for GQA, fused AddRMSNorm, fused SwiGLU, fused QKNormRoPE, merged QKV, merged gate+up.

Full results: `results/benchmark-2026-03-31.json` (3-run median)

## Training: PatchTST (GPU, 2026-04-09, v1.42+)

| Workload | v1.37 | v1.38.4 | v1.42+ | Speedup (v1.37→v1.42+) |
|----------|-------|---------|--------|------------------------|
| 28K×20×10 | 596s | 128.5s | **40.3s** | **14.8x** |
| 20K×20×5 | N/A | N/A | **15.0s** | — |
| 5K×10×3 | N/A | N/A | **3.0s** | — |

DGX GB10, commit `2ecf473a`. Pre-allocated batch workspace + GPU dst-memory
reuse (ztensor#84/#85). Loss converges 99.9% (0.0178 → 0.000022) on 28K×20×10.

Previous v1.38.4 baseline (128.5s at 28K×20×10) regressed to OOM after E85
buffer pre-allocation (commit `09a318c6`) introduced per-op GPU memory leaks.
Fixed in ztensor#85 by reusing dst device pointers instead of allocating per call.

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
