/* Q4_K dequantization kernel interface.
 *
 * Dequantizes Q4_K super-blocks to FP32 in global memory so the result
 * can be fed to cuBLAS Sgemm for non-GEMV (batch>1) MatMul.
 *
 * Q4_K super-block format (144 bytes per 256 values):
 *   - 2 bytes: fp16 d (super-block scale)
 *   - 2 bytes: fp16 dmin (super-block min)
 *   - 12 bytes: packed 6-bit scales and mins for 8 sub-blocks
 *   - 128 bytes: 256 x 4-bit quantized values packed (low nibble first)
 */
#ifndef DEQUANT_Q4K_H
#define DEQUANT_Q4K_H

#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

/* dequant_q4k_f32 dequantizes Q4_K super-blocks to FP32.
 *
 * src:    device pointer to raw Q4_K super-blocks for matrix [rows, K].
 *         rows * (K/256) super-blocks, each 144 bytes. Row-major layout.
 * dst:    device pointer to [rows, K] float output.
 * rows:   number of rows.
 * K:      number of columns. Must be a multiple of 256.
 * stream: CUDA stream.
 */
cudaError_t dequant_q4k_f32(
    const void* src, float* dst,
    int rows, int K,
    cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif /* DEQUANT_Q4K_H */
