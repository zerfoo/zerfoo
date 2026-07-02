# ADR-015: CUTLASS Quantized GEMM Kernels

**Status:** Accepted
**Date:** 2026-03-03
**Phase:** 18 (CUTLASS INT4/INT8 GEMM)

## Context

Quantized models use 4-bit or 8-bit integer weights to reduce memory and
increase throughput. The existing MatMulNBits layer dequantized INT4 weights
to float32 on CPU before using cuBLAS for matrix multiplication. This created
a CPU-GPU round trip: CPU dequant -> H2D copy -> cuBLAS. Fused kernels that
dequantize and multiply on GPU eliminate this overhead.

## Decision

### 1. CUDA Kernel Architecture

Three kernel files compile into the existing `libkernels.a`:

| File | Operation | Layout |
|------|-----------|--------|
| `gemm_int8.cu` | C = int8(A) * float32(B) | A[M,K] int8, B[K,N] f32, C[M,N] f32 |
| `gemm_int4.cu` | C = dequant(A) * B (left-mul) | A[M,K/2] packed INT4, B[K,N] f32 |
| `gemm_int4.cu` | C = B * dequant(W) (right-mul) | W[in,out/2] packed INT4, B[batch,in] f32 |

All kernels use 32x32 thread block tiling with shared memory staging.

### 2. INT4 Quantization Format

Weights are packed as two 4-bit signed values per byte (low nibble first).
Block quantization uses per-group scale factors (float32) and zero points
(uint8). The dequantization formula is:

    float_val = (int4_val - zero_point) * scale

The right-multiply variant (`gemm_int4_f32_rmul`) matches the standard neural
network forward pass layout: `output = input @ dequant(W)`, quantizing along
the output features (column) dimension.

### 3. Build Tag Strategy

All quantized GEMM code uses `//go:build cuda && cutlass`, matching the flash
attention pattern. The CGo bindings link against the same `libkernels.a`.

### 4. MatMulNBits GPU Dispatch

Build-tag-gated dispatch in `layers/core/`:

- `matmul_nbits_cuda.go` (`cuda && cutlass`): `tryQuantizedGemm` uploads
  quantized weights/scale/zeros to GPU, calls the fused INT4 GEMM kernel.
- `matmul_nbits_nocuda.go` (`!(cuda && cutlass)`): Returns (nil, nil) to
  signal fallback to CPU dequant + MatMul.

The Forward method tries the GPU path first, falling back to CPU on:
- Non-CUDA device type
- Non-4-bit quantization
- Any kernel error

### 5. INT8 GEMM Kernel

The INT8 kernel casts each int8 weight to float32 during the tiled multiply.
It uses shared memory to stage both the int8->float32 converted weights and
float32 activations, accumulating in float32. This avoids a separate
dequantization buffer.

## Consequences

**Positive:**
- Fused dequant+multiply eliminates CPU-GPU dequantization round trip
- MatMulNBits transparently accelerates on CUDA with zero API changes
- Build-tag-gated: no overhead when CUDA/CUTLASS not available
- Both left-multiply (C=dequant(A)*B) and right-multiply (C=B*dequant(W)) supported

**Negative:**
- Cannot be hardware-tested without CUDA GPU
- Per-forward-pass weight upload to GPU (future optimization: cache on device)
- 32x32 tile size may not be optimal for all GPU architectures

**Risks:**
- INT4 sign extension edge cases may produce different results than CPU
- Large weight matrices may exceed GPU memory during upload
