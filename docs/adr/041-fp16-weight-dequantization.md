# ADR 041: FP16 Weight Dequantization Instead of Native GEMV Kernels

## Status
Accepted

## Date
2026-03-17

## Context

Phase 24 wrote fused dequant-GEMV CUDA kernels for Q6_K, Q5_K, and Q5_0 quantized
weight types. The kernels correctly dequantize in registers and compute dot products,
but benchmarked at 50-89 tok/s -- 2-3x slower than cuBLAS SGEMM on dequantized
float32 (170 tok/s). cuBLAS exploits Blackwell's 4th-gen tensor cores and heavily
optimized memory access patterns that custom kernels cannot easily match.

The goal is 231+ tok/s to substantiate the claimed +18% advantage over Ollama (196
tok/s). The current float32 cuBLAS path (188 tok/s at 512 tokens) is bandwidth-bound
at 2.4 GB of weight reads per token on 273 GB/s LPDDR5x.

## Decision

Dequantize GGUF quantized weights (Q5_0, Q4_K, Q5_K, Q6_K) to FP16 instead of
float32 in the GGUF loader. Use `tensor.NewFloat16StorageFromF32` after the existing
dequantization-to-float32 step. The GPU engine already has an FP16 MatMul path
(`fp16MatMulNative`) that uses cuBLAS FP16 GEMM with tensor cores.

FP16 halves memory bandwidth (1.2 GB per token vs 2.4 GB), enabling a theoretical
maximum of 228 tok/s on GB10. Combined with L2 cache effects, this should approach
or exceed the 231 tok/s target.

The Phase 24 native GEMV kernel code remains in ztensor for potential future use on
hardware where custom kernels outperform cuBLAS (older GPUs without tensor cores).

## Consequences

**Positive:**
- 2x bandwidth reduction for weight reads (FP16 vs F32).
- cuBLAS FP16 GEMM uses tensor cores (higher compute throughput).
- No new CUDA kernels required (uses existing cuBLAS path).
- FP16 has 10-bit mantissa, more than enough for 4-6 bit quantized source data.
- Minimal code changes: 4 loader functions + 2 ztensor integration points.

**Negative:**
- FP16 has limited dynamic range (6e-8 to 65504). Weight values outside this range
  would clip. In practice, quantized weights are small (max ~16 for Q5_0).
- The theoretical max (228 tok/s) is 1% short of the 231 target. GPU caching effects
  may or may not bridge the gap.
- The Phase 24 native GEMV kernels become unused code. They should be kept but
  clearly documented as experimental/future.
