/* Fused SwiGLU activation kernel interface.
 *
 * Computes output[i] = w1[i] * sigmoid(w1[i]) * w3[i] in one kernel launch,
 * replacing the Concat + Split + sigmoid + Mul + Mul chain.
 */
#ifndef FUSED_SWIGLU_H
#define FUSED_SWIGLU_H

#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

/* fused_swiglu_f32 applies SwiGLU in one kernel launch.
 *
 * w1:     device pointer to [n] float array (gate projection output).
 * w3:     device pointer to [n] float array (up projection output).
 * output: device pointer to [n] float array.
 * n:      total number of elements.
 * stream: CUDA stream (pass NULL for default stream).
 */
cudaError_t fused_swiglu_f32(
    const float* w1, const float* w3, float* output,
    int n, cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif /* FUSED_SWIGLU_H */
