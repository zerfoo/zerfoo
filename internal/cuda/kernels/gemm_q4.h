/* Q4_0 dequant-GEMM kernel interface for Zerfoo's GPU-optimized Q4 layout.
 *
 * GPU layout (global separated):
 *   [all_scales: N_blocks * 2 bytes] [pad to 16B] [all_data: N_blocks * 16 bytes]
 *
 * Computes: C[m,n] = sum_k( dequant(A[m,k]) * B[k,n] )
 * A is packed Q4_0 blocks in separated layout.
 * B is [K, N] row-major FP32. C is [M, N] row-major FP32.
 */
#ifndef GEMM_Q4_H
#define GEMM_Q4_H

#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

/* gemm_q4_f32 performs Q4_0 dequant-GEMM:
 *   C[m,n] = sum_k( dequant(A_q4[m,k]) * B[k,n] )
 *
 * A_q4:        device pointer to separated Q4 layout (scales then data).
 * B:           device pointer to [K * N] float array (row-major).
 * C:           device pointer to [M * N] float array (row-major output).
 * M, K, N:     matrix dimensions. K must be a multiple of 32.
 * data_offset: byte offset from A_q4 to the packed data region.
 * stream:      CUDA stream.
 */
cudaError_t gemm_q4_f32(
    const void* A_q4, const float* B, float* C,
    int M, int K, int N,
    int data_offset,
    cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif /* GEMM_Q4_H */
