# ADR 030: Ollama Performance Parity Strategy

## Status
Accepted

## Date
2026-03-11

## Context
Zerfoo achieves 0.44-12.84 tok/s on DGX Spark GB10 depending on execution path.
Ollama running open weights on the same hardware achieves ~100 tok/s. The gap is
8-230x. Additionally, both CPU and CUDA inference paths produce degenerate output,
indicating a pre-existing correctness bug that must be fixed before meaningful
performance comparison.

The DGX Spark GB10 has 273 GB/s LPDDR5x bandwidth. For a 1.5GB Q4 model, the
theoretical max is ~182 tok/s. Ollama at ~100 tok/s is 55% of theoretical.
Zerfoo at 12.84 tok/s (best case) is 7% of theoretical.

Previous non-goals (cuBLAS/cuDNN/TensorRT/CUTLASS/ROCm/OpenCL purego conversion
and performance tuning) are now in scope per user direction.

## Decision
A phased approach prioritizing correctness, then performance, then portability:

1. **Correctness first**: Fix the degenerate output bug. No performance work
   matters if the model produces garbage.

2. **Eliminate CPU fallbacks**: GPU Transpose, GPU Gather, GPU broadcasting.
   These cause D2H/H2D round-trips that dominate inference time (43% cgocall
   overhead per ADR-022 analysis).

3. **Kernel fusion**: SwiGLU, Scale+Softmax, dequant+GEMV fused kernels per
   ADR-024. Reduce memory bandwidth by eliminating intermediate writes.

4. **CUDA graph capture**: Record decode forward pass, replay per token.
   Eliminate all per-op launch overhead per ADR-024.

5. **Megakernel fix or replace**: Investigate the 30x performance gap. Either
   fix the code generator or pivot to CUDA graph + fused kernels approach.

6. **Kernel optimization**: NVCC flags, register pressure, shared memory,
   occupancy tuning on sm_121 (Blackwell).

7. **CGo to purego conversion**: cuBLAS, cuDNN, TensorRT, CUTLASS. Enables
   single-binary deployment without CGo toolchain.

8. **Backend expansion**: ROCm and OpenCL purego conversion for portability.

## Consequences
Positive:
- Clear priority ordering prevents wasted effort on performance before correctness.
- Phased approach allows early wins (GPU residency) before complex work (megakernel).
- Single binary deployment (no CGo) simplifies distribution.
- Multi-backend purego enables running on AMD and Intel GPUs.

Negative:
- Very large scope. Estimated 100+ hours of implementation work.
- purego conversion of cuBLAS/cuDNN may have performance implications vs CGo.
- ROCm/OpenCL backends are incomplete and may need significant new kernel work.
- Risk of diminishing returns: last 2x from 50 to 100 tok/s may be hardest.
