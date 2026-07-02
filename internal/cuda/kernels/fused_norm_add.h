/* Fused RMSNorm + Add kernel interface.
 *
 * Combines RMS normalization and residual addition into one kernel launch:
 *   output = rmsnorm(input, weight, eps) + residual
 */
#ifndef FUSED_NORM_ADD_H
#define FUSED_NORM_ADD_H

#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

/* fused_norm_add_f32 applies RMSNorm then adds residual in one kernel launch.
 *
 * input:    device pointer to [rows, D] float array (data to normalize).
 * weight:   device pointer to [D] float array (RMSNorm weight).
 * residual: device pointer to [rows, D] float array (added after norm).
 * output:   device pointer to [rows, D] float array.
 * eps_bits: epsilon as uint32 bit pattern.
 * rows:     number of rows.
 * D:        row dimension.
 * stream:   CUDA stream (pass NULL for default stream).
 */
cudaError_t fused_norm_add_f32(
    const float* input, const float* weight, const float* residual,
    float* output,
    unsigned int eps_bits, int rows, int D,
    cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif /* FUSED_NORM_ADD_H */
