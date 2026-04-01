# ADR 040: Native GGUF GEMV Kernels Instead of Re-Quantization

## Status
Accepted

## Date
2026-03-17

## Context

GGUF Q4_K_M models contain mixed quantization types: Q4_K, Q5_0, Q5_K, Q6_K, Q8_0,
F32, and F16. Zerfoo only had optimized GEMV kernels for Q4_0 and Q4_K. To handle
Q5_0 and Q4_K tensors, the GGUF loader re-quantized them to Q4_0 at load time. Q5_K
and Q6_K were dequantized to float32.

This approach had two problems:

1. **Output quality regression**: Q4_K-to-Q4_0 re-quantization drops per-sub-block
   6-bit scales. Q5_0-to-Q4_0 drops 1 bit per weight. Both cause logit divergence
   that compounds through 26 transformer layers, producing garbled output text.

2. **Throughput regression**: Q5_K and Q6_K dequantized to float32 use 32 bits per
   weight instead of 5.5-6.6 bits, consuming 5-7x more memory bandwidth. This makes
   cuBLAS SGEMM memory-bandwidth-bound instead of compute-bound.

The prior "241 tok/s beats Ollama" benchmark (ADR-033) was measured with Q4_0 ZMF
format where all weights used the fast Q4_0 GEMV. That measurement produced garbled
output and was not a valid quality comparison.

## Decision

Write native fused dequant-GEMV CUDA kernels for Q6_K, Q5_K, and Q5_0. Each kernel
reads quantized super-blocks directly from GPU memory, dequantizes in registers
(no global memory intermediary), and accumulates the dot product in FP32.

Remove all lossy re-quantization from the GGUF loader. Each quant type uses its
native storage (Q4KStorage, Q5KStorage, Q6KStorage, Q5_0Storage) and is dispatched
to its corresponding GEMV kernel by the GPU engine.

The existing pattern (gemv_q4k.cu: shared-mem x, warp-per-row, shuffle reduce) is
used as the template for all new kernels.

## Consequences

**Positive:**
- Correct output: no precision loss from re-quantization.
- Throughput: 5-7x bandwidth reduction for Q5_K/Q6_K (float32 -> native quant).
- Maintainability: each quant type has a clear, independent code path.
- Extensibility: adding new GGUF quant types follows the same pattern.

**Negative:**
- Three new CUDA kernels to maintain (gemv_q6k, gemv_q5k, gemv_q5_0).
- Three new purego bindings (purego + CGo variants).
- GPU engine complexity increases (more storage type checks in MatMul dispatch).
- Q5_0 uses 32-element blocks (not 256 super-blocks), requiring a different
  blocking strategy with potentially lower GPU occupancy.
