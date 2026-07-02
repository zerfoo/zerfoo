# ADR 023: GPU Scalar Ops and D2H Elimination Strategy

## Status
Accepted

## Date
2026-03-06

## Context

Phase 32 brought GPU inference from 5.12 to 6.84 tok/s (+33.6%) on DGX Spark
GB10 by adding GPU kernels for Transpose, Gather, Broadcasting, and RMSNorm,
plus Q4 weight pre-upload. However, the 10 tok/s target was not met.

Profiling identified three remaining bottlenecks:
1. Pow CPU fallback (8.9%): `engine.Pow(base, exponent)` where exponent is a
   scalar tensor [1] with value 2.0 (x^2 in normalization). `gpuPow` requires
   `sameShape(base, exponent)` which fails, falling back to CPU. The CPU path
   then calls `GPUStorage.Slice()` triggering a full D2H copy of the base tensor.
2. Binary op CPU fallback (10.4%): the 2D broadcast kernel in `gpuBroadcastOp`
   only supports row, column, and same-shape patterns. Scalar-vs-tensor (e.g.
   [1] op [M,D]) is not handled, falling back to CPU with D2H copies.
3. GPUStorage.Slice D2H (24%): all CPU fallback ops that read GPU-resident
   tensor data trigger `GPUStorage.Slice()` which copies the entire buffer D2H.
   This is the root cause multiplier for both (1) and (2).

## Decision

Implement three targeted changes to eliminate CPU fallbacks in the inference
hot loop:

1. **PowScalar kernel**: Add a CUDA kernel `pow_scalar(x, p, out, n)` that
   computes `out[i] = pow(x[i], p)` where p is a host scalar. Add
   `PowScalar` to the KernelRunner interface. In `gpuPow`, detect when the
   exponent tensor has 1 element (scalar broadcast) and dispatch to the scalar
   kernel instead of falling back to CPU.

2. **Scalar-broadcast for all binary ops**: Extend `gpuBroadcastOp` to detect
   when either operand has exactly 1 element total and treat it as a scalar
   broadcast. Use the existing `*Scalar` kernel variants (AddScalar, MulScalar,
   DivScalar) for matching ops, and add SubScalar + PowScalar for the gaps.

3. **GPU Slice kernel**: Add a CUDA kernel for strided sub-tensor extraction
   so that `tensor.Slice()` on GPU-resident tensors stays on GPU. This
   eliminates the D2H copies triggered by CPU fallback ops reading GPU data.
   Also add GPU Split and Concat to avoid D2H in those paths.

## Consequences

Positive:
- Eliminates the three largest remaining CPU fallback sources (~43% of wall time).
- All float32 ops in the Gemma 3 inference graph should execute on GPU.
- Expected throughput: >10 tok/s on DGX Spark GB10.
- cgocall % should drop from 58% to <15%.

Negative:
- 3 new CUDA kernels to maintain (pow_scalar, sub_scalar, gpu_slice).
- KernelRunner interface grows by 3-4 methods.
- ROCm and OpenCL backends need stub implementations (CPU fallback acceptable).
