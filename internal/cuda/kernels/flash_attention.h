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
 * separate Q and KV sequence lengths.
 *
 * Q: [num_bh, 1, head_dim]           -- single query per batch-head.
 * K: [num_bh, max_kv_len, head_dim]  -- pre-allocated KV buffer.
 * V: [num_bh, max_kv_len, head_dim]
 * O: [num_bh, 1, head_dim]           -- output, same shape as Q.
 * num_bh:      batch * heads.
 * max_kv_len:  stride of K/V buffer (allocated capacity).
 * head_dim:    dimension per head.
 * kv_len:      actual KV sequence length (used if kv_len_ptr is null).
 * kv_len_ptr:  GPU-resident int32. If non-null, *kv_len_ptr is read at
 *              runtime for the actual KV length, making the kernel
 *              compatible with CUDA graph replay (the value is not frozen).
 */
cudaError_t flash_attention_decode_f32(
    const float* Q, const float* K, const float* V, float* O,
    int num_bh, int max_kv_len, int head_dim,
    int kv_len, const int* kv_len_ptr,
    cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif /* FLASH_ATTENTION_H */
