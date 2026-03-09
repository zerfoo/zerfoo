# ADR 020: Q4 Quantized Dot Product

## Status
Accepted

## Date
2026-03-06

## Context
Phase 28 DGX Spark benchmarks show 3.80 tok/s for Gemma 3 2B Q4_0 on ARM64
CPU. The dominant bottleneck is the dequantize-then-MatMul pattern: every Q4
tensor is fully expanded to float32 before GEMM. This wastes memory bandwidth
(reading 4-bit data, writing 32-bit intermediates, then reading those
intermediates again for GEMM) and creates allocation pressure (79,537
allocs/token, 39.4 GB/op).

A fused Q4*float32 dot product kernel would read packed Q4 nibbles directly
during the dot product, eliminating the intermediate float32 buffer entirely.
This halves memory bandwidth for weight-matrix reads and removes the largest
source of allocations.

The existing `internal/xblas` package provides NEON-optimized SGEMM
infrastructure (`sgemmAccRowNeon`). The Q4 dot product builds on this pattern.

## Decision
Add a `Q4DotF32` function to `internal/xblas` that computes the dot product
of a Q4_0 packed weight vector and a float32 activation vector without
intermediate dequantization. The kernel:

1. Reads Q4_0 super-blocks (block_size=32, 18 bytes per block: 2-byte
   float16 scale + 16 bytes of packed nibbles).
2. For each block, extracts 32 nibbles, subtracts the zero-point (8),
   multiplies by the block scale, and accumulates the dot product with the
   corresponding float32 activation values.
3. Uses NEON SIMD intrinsics (ARM64 assembly) for the inner loop: load 16
   bytes of packed nibbles, split into low/high nibbles with AND/SHIFT,
   convert to float32x4 via int8->int16->int32->float32 widening, FMA with
   activation vector and scale.
4. Falls back to a pure Go scalar implementation on non-ARM64 platforms.

Integration into the GEMM path:
- `CPUEngine.MatMul` detects when the left operand uses `Q4Storage` and
  dispatches to a row-wise `Q4DotF32` loop instead of dequantizing first.
- K-quant variants (Q4_K, Q5_K, Q6_K) get similar fused kernels in a
  follow-up phase, reusing the same pattern with different block layouts.

## Consequences
**Positive:**
- Eliminates the largest source of allocations (dequantize buffers).
- Halves memory bandwidth for weight reads (4-bit vs 32-bit).
- Expected 2-3x throughput improvement for Q4 models on ARM64.
- Pattern extends to Q4_K, Q5_K, Q6_K with minimal additional work.

**Negative:**
- Adds platform-specific assembly (ARM64 NEON), increasing maintenance.
- Scalar fallback on x86 will be slower than NEON path; x86 AVX2/AVX-512
  variants would be needed for parity on Intel/AMD.
- Q4DotF32 is specific to Q4_0 block format; each quantization format
  needs its own kernel variant.
