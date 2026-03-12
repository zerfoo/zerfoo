/* Fused Add + RMSNorm kernel interface.
 *
 * Combines residual addition and RMS normalization into one kernel launch:
 *   residual = input + residual          (in-place)
 *   output   = rmsnorm(residual, weight, eps)
 */
#ifndef FUSED_ADD_RMSNORM_H
#define FUSED_ADD_RMSNORM_H

#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

/* fused_add_rmsnorm_f32 performs fused add + RMSNorm in one kernel launch.
 *
 * input:    device pointer to [rows, D] float array.
 * residual: device pointer to [rows, D] float array (updated in-place with input+residual).
 * weight:   device pointer to [D] float array (RMSNorm weight).
 * output:   device pointer to [rows, D] float array (normalized output).
 * eps_bits: epsilon as uint32 bit pattern (use float-to-bits conversion).
 * rows:     number of rows.
 * D:        row dimension (number of columns).
 * stream:   CUDA stream (pass NULL for default stream).
 */
cudaError_t fused_add_rmsnorm_f32(
    const float* input, float* residual, const float* weight, float* output,
    unsigned int eps_bits, int rows, int D, cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif /* FUSED_ADD_RMSNORM_H */
