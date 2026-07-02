/* INT4 mixed-precision GEMM kernel interface.
 * Computes: C = dequant(A) * B where A is packed INT4 weights with
 * block-quantized scales and zero points, B is FP32 activations.
 *
 * Layout: A is [M, K/2] row-major packed INT4 (two 4-bit values per byte,
 * low nibble first). B is [K, N] row-major FP32. C is [M, N] row-major FP32.
 *
 * Block quantization: weights are quantized in groups of group_size along
 * the K dimension. Each group has a float32 scale and uint8 zero point.
 * scales is [M, K/group_size] row-major FP32.
 * zeros is [M, K/group_size] row-major uint8.
 * Dequantized value: (int4_value - zero_point) * scale.
 */
#ifndef GEMM_INT4_H
#define GEMM_INT4_H

#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

/* gemm_int4_f32 performs block-quantized INT4 mixed-precision GEMM:
 *   C[m,n] = sum_k( dequant(A[m,k]) * B[k,n] )
 * where dequant(v) = (v - zeros[m, k/group_size]) * scales[m, k/group_size]
 *
 * A:          device pointer to [M * K/2] uint8 array (packed INT4, row-major).
 * B:          device pointer to [K * N] float array (row-major activations).
 * C:          device pointer to [M * N] float array (row-major output).
 * scales:     device pointer to [M * num_groups] float array (per-group scales).
 * zeros:      device pointer to [M * num_groups] uint8 array (per-group zero points).
 * M, K, N:    matrix dimensions (K must be even).
 * group_size: quantization group size along K (typically 32 or 128).
 * stream:     CUDA stream for async execution.
 */
cudaError_t gemm_int4_f32(
    const void* A, const float* B, float* C,
    const float* scales, const void* zeros,
    int M, int K, int N, int group_size,
    cudaStream_t stream);

/* gemm_int4_f32_rmul performs right-multiply: C = B * dequant(W)
 *   C[i,j] = sum_k( B[i,k] * dequant(W[k,j]) )
 * This is the standard neural network forward pass pattern where W is the
 * weight matrix in [in_features, out_features] layout.
 *
 * W:          device pointer to [in_features * out_features/2] uint8 (packed INT4).
 * B:          device pointer to [batch * in_features] float (row-major input).
 * C:          device pointer to [batch * out_features] float (row-major output).
 * scales:     device pointer to [in_features * num_groups] float (per-group).
 * zeros:      device pointer to [in_features * num_groups] uint8 (per-group).
 * batch:      number of input rows.
 * in_features:  weight matrix rows (= K, must be even for packed INT4 along cols).
 * out_features: weight matrix cols after unpacking (= K * 2 from packed).
 * group_size: quantization group size along out_features dimension.
 * stream:     CUDA stream.
 */
cudaError_t gemm_int4_f32_rmul(
    const void* W, const float* B, float* C,
    const float* scales, const void* zeros,
    int batch, int in_features, int out_features, int group_size,
    cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif /* GEMM_INT4_H */
