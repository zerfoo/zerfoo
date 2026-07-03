/* Tiny-matrix strided-batched GEMM interface (ADR 075 lever L3).
 *
 * Computes batch independent small f32 GEMMs C_b = A_b * B_b (alpha=1, beta=0,
 * row-major) in a single launch, one CUDA thread-block per batch element, with
 * the A/B tiles staged in shared memory. Replaces cuBLAS SgemmStridedBatched on
 * the tiny attention shapes (12x12, 12x64 over batch=1024) where cuBLAS falls
 * back to a GEMV + split-K reduction fan-out.
 *
 * All arguments are integers / pointers -- no floating-point scalars -- so the
 * purego integer-register dispatch ABI (see fused_adamw.h) is satisfied.
 */
#ifndef TINY_BATCHED_GEMM_H
#define TINY_BATCHED_GEMM_H

#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

/* tiny_batched_gemm_f32 computes, for each b in [0, batch):
 *     C_b[i,j] = sum_l A_b[i,l] * B_b[l,j]
 * where A_b = A + b*strideA, B_b = B + b*strideB, C_b = C + b*strideC
 * (strides in ELEMENTS), A_b is row-major [m,k], B_b row-major [k,n],
 * C_b row-major [m,n]. Accumulation is in f32.
 *
 * Returns cudaErrorInvalidValue if any of m,n,k exceeds the supported tiny
 * bound (64) or any dimension/batch is non-positive, so the caller can fall
 * back to cuBLAS.
 */
cudaError_t tiny_batched_gemm_f32(const float* A, const float* B, float* C,
                                  int m, int n, int k,
                                  long long strideA, long long strideB,
                                  long long strideC,
                                  int batch, cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif /* TINY_BATCHED_GEMM_H */
