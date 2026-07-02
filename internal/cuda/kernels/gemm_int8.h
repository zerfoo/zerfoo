/* INT8 mixed-precision GEMM kernel interface.
 * Computes: C = dequant(A) * B where A is INT8 weights, B is FP32 activations.
 *
 * Layout: A is [M, K] row-major INT8, B is [K, N] row-major FP32,
 * C is [M, N] row-major FP32. Each INT8 value is cast to float before multiply.
 */
#ifndef GEMM_INT8_H
#define GEMM_INT8_H

#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

/* gemm_int8_f32 performs mixed-precision matrix multiplication:
 *   C[m,n] = sum_k( float(A[m,k]) * B[k,n] )
 *
 * A:      device pointer to [M * K] int8_t array (row-major weights).
 * B:      device pointer to [K * N] float array (row-major activations).
 * C:      device pointer to [M * N] float array (row-major output).
 * M, K, N: matrix dimensions.
 * stream: CUDA stream for async execution.
 */
cudaError_t gemm_int8_f32(
    const void* A, const float* B, float* C,
    int M, int K, int N,
    cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif /* GEMM_INT8_H */
