/* Fused Add + RMSNorm kernel interface.
 *
 * Combines residual addition and RMS normalization into one kernel launch:
 *   sum_out    = input + residual
 *   normed_out = rmsnorm(sum_out, weight, eps)
 */
#ifndef FUSED_ADD_RMSNORM_H
#define FUSED_ADD_RMSNORM_H

#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

/* fused_add_rmsnorm_f32 performs fused add + RMSNorm in one kernel launch.
 *
 * input:      device pointer to [rows, D] float array.
 * residual:   device pointer to [rows, D] float array (read-only).
 * weight:     device pointer to [D] float array (RMSNorm weight).
 * normed_out: device pointer to [rows, D] float array (normalized output).
 * sum_out:    device pointer to [rows, D] float array (input + residual).
 * eps_bits:   epsilon as uint32 bit pattern (use float-to-bits conversion).
 * rows:       number of rows.
 * D:          row dimension (number of columns).
 * stream:     CUDA stream (pass NULL for default stream).
 */
cudaError_t fused_add_rmsnorm_f32(
    const float* input, const float* residual, const float* weight,
    float* normed_out, float* sum_out,
    unsigned int eps_bits, int rows, int D, cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif /* FUSED_ADD_RMSNORM_H */
