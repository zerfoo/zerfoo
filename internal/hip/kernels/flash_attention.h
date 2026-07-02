/* Flash attention forward kernel interface (HIP).
 * Computes: O = softmax(Q * K^T / sqrt(head_dim)) * V
 * with optional causal masking.
 *
 * Layout: All tensors are [batch, heads, seq_len, head_dim] in row-major order.
 */
#ifndef FLASH_ATTENTION_HIP_H
#define FLASH_ATTENTION_HIP_H

#include <hip/hip_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

hipError_t flash_attention_forward_f32(
    const float* Q, const float* K, const float* V, float* O,
    int batch, int heads, int seq_len, int head_dim,
    int causal, hipStream_t stream);

#ifdef __cplusplus
}
#endif

#endif /* FLASH_ATTENTION_HIP_H */
