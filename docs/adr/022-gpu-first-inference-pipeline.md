# ADR 022: GPU-First Inference Pipeline

## Status

Accepted

## Date

2026-03-06

## Context

Phase 31 profiled GPU inference on DGX Spark GB10 with Gemma 3 2B Q4.
GPU inference (5.12 tok/s) was slower than CPU (5.94 tok/s). The profile
revealed:

- 43% of wall time in `runtime.cgocall` (CUDA kernel launches + host-device
  data transfers). Tensors shuttle between CPU and GPU for every operation.
- Only MatMul runs on GPU. All other ops (Transpose, element-wise binary ops,
  Gather, RMSNorm decomposition, rotary embedding) fall back to CPUEngine.
- 9.4% in Q4 dequantize on CPU, 8.1% in Transpose on CPU, 4.4% in binary
  ops on CPU.

The GPUEngine already has CUDA kernels for element-wise ops (Add, Sub, Mul,
Div, Exp, Log, Sqrt, Rsqrt, Tanh, Softmax, ReduceSum). The CPU fallbacks
happen because:

1. **Transpose** has no GPU implementation (hardcoded CPU fallback).
2. **Binary ops** fall back when shapes differ (broadcasting not supported
   on GPU path, which requires `sameShape()`).
3. **Gather** (embedding lookup) has no GPU implementation.
4. **Intermediate tensors** are created on CPU by default. Each GPU op must
   copy input H2D and output D2H, causing the 43% cgocall overhead.

## Decision

Phase 32 adopts a GPU-first inference pipeline strategy:

1. **GPU tensor residency:** Intermediate tensors remain on GPU between
   operations. Model weights are uploaded once at load time. Only the final
   logits tensor is copied back to CPU. This eliminates most H2D/D2H transfers.

2. **GPU Transpose kernel:** Write a CUDA transpose kernel for 2D, 3D, and 4D
   tensors. Wire into GPUEngine.Transpose to eliminate the CPU fallback.

3. **GPU broadcasting:** Extend GPU element-wise kernels to support broadcasting
   for common patterns (scalar, row, column). Eliminate the sameShape() guard.

4. **GPU Gather kernel:** Write a CUDA gather kernel for embedding table lookups
   and KV cache operations. Keep embedding output on GPU.

5. **Fused GPU RMSNorm kernel:** Single-kernel RMSNorm (reduce + normalize +
   scale) instead of decomposing into 5+ element-wise ops. Reduces kernel launch
   count and memory traffic.

6. **Order of implementation:** Tensor residency first (biggest impact at 43%),
   then Transpose, broadcasting, Gather, and fused kernels.

## Consequences

### Positive

- Eliminates the dominant 43% cgocall overhead by keeping tensors GPU-resident.
- Removes all major CPU fallbacks during transformer inference.
- Expected throughput improvement: 2x or more (5.12 -> >10 tok/s).
- Enables future GPU-only inference path where CPU is not involved in the
  hot loop at all.

### Negative

- GPU tensor residency requires changes to how tensors flow through the graph
  execution pipeline. Engine methods must return GPU-resident tensors.
- Debugging is harder when tensors are on GPU (cannot inspect values without
  D2H copy).
- Memory usage increases slightly (all intermediates on GPU instead of CPU).
  DGX Spark has 128GB unified memory so this is not a concern.
- New CUDA kernels need parity tests (GPU vs CPU output comparison).
- Build-tag complexity: all new kernels gated behind `//go:build cuda`.
