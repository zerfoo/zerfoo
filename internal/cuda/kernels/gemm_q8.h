/* Q8_0 dequant-GEMM kernel interface for Zerfoo's Q8Storage format.
 *
 * Block format (36 bytes per 32 values):
 *   - 4 bytes: float32 scale (little-endian IEEE 754)
 *   - 32 bytes: 32 x int8 quantized values
 *   Dequant: float_val = int8_val * scale
 *
 * Computes: C[m,n] = sum_k( dequant(A[m,k]) * B[k,n] )
 * A is [M * num_blocks_per_row * 36] packed Q8_0 blocks.
 * B is [K, N] row-major FP32. C is [M, N] row-major FP32.
 */
#ifndef GEMM_Q8_H
#define GEMM_Q8_H

#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

/* gemm_q8_f32 performs Q8_0 dequant-GEMM:
 *   C[m,n] = sum_k( dequant(A_q8[m,k]) * B[k,n] )
 *
 * A_q8:   device pointer to packed Q8_0 blocks for matrix A [M, K].
 *         M * ceil(K/32) blocks, each 36 bytes. Row-major block layout.
 * B:      device pointer to [K * N] float array (row-major).
 * C:      device pointer to [M * N] float array (row-major output).
 * M, K, N: matrix dimensions. K must be a multiple of 32.
 * stream: CUDA stream.
 */
cudaError_t gemm_q8_f32(
    const void* A_q8, const float* B, float* C,
    int M, int K, int N,
    cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif /* GEMM_Q8_H */
