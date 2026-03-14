/* Flash attention forward kernel interface.
 * Computes: O = softmax(Q * K^T / sqrt(head_dim)) * V
 * with optional causal masking.
 *
 * Layout: All tensors are [batch, heads, seq_len, head_dim] in row-major order.
 */
#ifndef FLASH_ATTENTION_H
#define FLASH_ATTENTION_H

#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

/* flash_attention_forward_f32 computes scaled dot-product attention in a single
 * fused pass using tiled computation with shared memory staging.
 *
 * Q, K, V: device pointers to [batch * heads * seq_len * head_dim] float32 arrays.
 * O:       device pointer to output [batch * heads * seq_len * head_dim].
 * batch:   number of sequences in the batch.
 * heads:   number of attention heads.
 * seq_len: sequence length (same for Q, K, V).
 * head_dim: dimension per head.
 * causal:  if nonzero, apply causal (upper-triangular) mask.
 * stream:  CUDA stream for async execution.
 */
cudaError_t flash_attention_forward_f32(
    const float* Q, const float* K, const float* V, float* O,
    int batch, int heads, int seq_len, int head_dim,
    int causal, cudaStream_t stream);

/* flash_attention_decode_f32 computes attention for single-query decode with
 * separate Q and KV sequence lengths. Supports GQA where num_q_heads may
 * differ from num_kv_heads (num_q_heads must be a multiple of num_kv_heads).
 *
 * Q: [batch * num_q_heads, 1, head_dim]            -- single query per head.
 * K: [batch * num_kv_heads, max_kv_len, head_dim]  -- pre-allocated KV buffer.
 * V: [batch * num_kv_heads, max_kv_len, head_dim]
 * O: [batch * num_q_heads, 1, head_dim]            -- output, same shape as Q.
 * num_bh:        batch * num_q_heads (grid size).
 * max_kv_len:    stride of K/V buffer (allocated capacity).
 * head_dim:      dimension per head.
 * kv_len:        actual KV sequence length (used if kv_len_ptr is null).
 * kv_len_ptr:    GPU-resident int32. If non-null, *kv_len_ptr is read at
 *                runtime for the actual KV length, making the kernel
 *                compatible with CUDA graph replay (the value is not frozen).
 * num_q_heads:   number of query heads per batch element.
 * num_kv_heads:  number of KV heads per batch element.
 */
cudaError_t flash_attention_decode_f32(
    const float* Q, const float* K, const float* V, float* O,
    int num_bh, int max_kv_len, int head_dim,
    int kv_len, const int* kv_len_ptr,
    int num_q_heads, int num_kv_heads,
    cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif /* FLASH_ATTENTION_H */
