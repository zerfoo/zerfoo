/* Q4_K fused dequant-GEMV kernel interface.
 *
 * Q4_K super-block format (144 bytes per 256 values):
 *   - 2 bytes: fp16 d (super-block scale)
 *   - 2 bytes: fp16 dmin (super-block min)
 *   - 12 bytes: packed 6-bit scales and mins for 8 sub-blocks
 *   - 128 bytes: 256 x 4-bit quantized values packed (low nibble first)
 *
 * Computes: y[m] = sum_k( dequant(W_q4k[m,k]) * x[k] )
 * W_q4k is raw Q4_K super-blocks laid out row-major.
 * x is [K] FP32 input vector. y is [M] FP32 output vector.
 * Batch=1 only (GEMV, not GEMM).
 */
#ifndef GEMV_Q4K_H
#define GEMV_Q4K_H

#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

/* gemv_q4k_f32 performs Q4_K fused dequant-GEMV:
 *   y[m] = sum_k( dequant(W_q4k[m,k]) * x[k] )
 *
 * W_q4k: device pointer to raw Q4_K super-blocks for matrix W [M, K].
 *        M * ceil(K/256) super-blocks, each 144 bytes. Row-major layout.
 * x:     device pointer to [K] float input vector.
 * y:     device pointer to [M] float output vector.
 * M, K:  matrix dimensions. K must be a multiple of 256.
 * stream: CUDA stream.
 */
cudaError_t gemv_q4k_f32(
    const void* W_q4k, const float* x, float* y,
    int M, int K,
    cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif /* GEMV_Q4K_H */
