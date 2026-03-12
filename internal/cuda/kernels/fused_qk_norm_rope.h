/* Fused per-head QK RMSNorm + RoPE kernel interface.
 *
 * Replaces 4 kernel launches (Q_norm + K_norm + Q_RoPE + K_RoPE) with 1.
 * Input: contiguous [totalHeads, headDim] buffer containing Q heads followed
 * by K heads. Heads 0..numQHeads-1 use weightQ, the rest use weightK.
 */
#ifndef FUSED_QK_NORM_ROPE_H
#define FUSED_QK_NORM_ROPE_H

#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

/* fused_qk_norm_rope_f32 applies per-head RMSNorm + RoPE in one kernel launch.
 *
 * input:      device pointer to [totalHeads, headDim] float array (Q then K heads).
 * weightQ:    device pointer to [headDim] float array (RMSNorm weight for Q).
 * weightK:    device pointer to [headDim] float array (RMSNorm weight for K).
 * cosAngles:  device pointer to [halfRotary] float array.
 * sinAngles:  device pointer to [halfRotary] float array.
 * output:     device pointer to [totalHeads, headDim] float array.
 * eps_bits:   RMSNorm epsilon as uint32 bit pattern.
 * totalHeads: numQHeads + numKVHeads.
 * headDim:    dimension per head.
 * numQHeads:  boundary index (heads 0..numQHeads-1 use weightQ).
 * halfRotary: half of the rotary dimension.
 * stream:     CUDA stream (pass NULL for default stream).
 */
cudaError_t fused_qk_norm_rope_f32(
    const float* input, const float* weightQ, const float* weightK,
    const float* cosAngles, const float* sinAngles, float* output,
    unsigned int eps_bits,
    int totalHeads, int headDim, int numQHeads, int halfRotary,
    cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif /* FUSED_QK_NORM_ROPE_H */
