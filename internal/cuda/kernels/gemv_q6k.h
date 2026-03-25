/* Q6_K fused dequant-GEMV kernel interface.
 *
 * Q6_K super-block format (210 bytes per 256 values):
 *   - 128 bytes: ql (low 4 bits of each 6-bit value)
 *   - 64 bytes:  qh (high 2 bits of each 6-bit value)
 *   - 16 bytes:  sc (int8 scales for 16 sub-blocks of 16 values)
 *   - 2 bytes:   fp16 d (super-block scale)
 *
 * Computes: y[m] = sum_k( dequant(W_q6k[m,k]) * x[k] )
 * W_q6k is raw Q6_K super-blocks laid out row-major.
 * x is [K] FP32 input vector. y is [M] FP32 output vector.
 * Batch=1 only (GEMV, not GEMM).
 */
#ifndef GEMV_Q6K_H
#define GEMV_Q6K_H

#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

/* gemv_q6k_f32 performs Q6_K fused dequant-GEMV:
 *   y[m] = sum_k( dequant(W_q6k[m,k]) * x[k] )
 *
 * W_q6k: device pointer to raw Q6_K super-blocks for matrix W [M, K].
 *        M * ceil(K/256) super-blocks, each 210 bytes. Row-major layout.
 * x:     device pointer to [K] float input vector.
 * y:     device pointer to [M] float output vector.
 * M, K:  matrix dimensions. K must be a multiple of 256.
 * stream: CUDA stream.
 */
cudaError_t gemv_q6k_f32(
    const void* W_q6k, const float* x, float* y,
    int M, int K,
    cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif /* GEMV_Q6K_H */
