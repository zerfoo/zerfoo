/* Fused Rotary Position Embedding (RoPE) kernel interface.
 *
 * Applies rotary position embeddings in a single pass over
 * [batch, seq_len, head_dim] input, using precomputed cos/sin tables.
 *
 * For each position (b, s, d):
 *   d < halfRotary:                  rotated (first half)
 *   halfRotary <= d < 2*halfRotary:  rotated (second half)
 *   d >= rotaryDim:                  pass-through
 */
#ifndef FUSED_ROPE_H
#define FUSED_ROPE_H

#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

/* fused_rope_f32 applies rotary position embeddings in one kernel launch.
 *
 * input:       device pointer to [batch * seq_len * head_dim] float array.
 * cos_angles:  device pointer to [seq_len * cos_stride] float array.
 * sin_angles:  device pointer to [seq_len * cos_stride] float array.
 * output:      device pointer to [batch * seq_len * head_dim] float array.
 * batch:       batch size.
 * seq_len:     sequence length.
 * head_dim:    total dimension per head.
 * half_rotary: half of rotaryDim (number of cos/sin columns used).
 * cos_stride:  row stride of cos/sin tables (elements per row).
 * stream:      CUDA stream (pass NULL for default stream).
 */
cudaError_t fused_rope_f32(
    const float* input, const float* cos_angles, const float* sin_angles,
    float* output,
    int batch, int seq_len, int head_dim, int half_rotary, int cos_stride,
    cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif /* FUSED_ROPE_H */
