/* Split-KV flash decode kernel interface (float32).
 *
 * Optimized for autoregressive decode (seqLen_Q = 1): splits the KV cache
 * across S thread blocks per head, each computing partial attention with
 * online softmax. A second reduction kernel merges partial results using
 * log-sum-exp correction.
 *
 * Grid: [numHeads, S] where S = ceil(seqLen_KV / chunk_size).
 */
#ifndef FLASH_DECODE_H
#define FLASH_DECODE_H

#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

/* flash_decode_splitkv_f32 computes single-query attention with split-KV
 * parallelism for long KV caches.
 *
 * Q:             [batch * num_q_heads, head_dim]
 * K:             [batch, max_kv_len, num_kv_heads * head_dim]
 * V:             [batch, max_kv_len, num_kv_heads * head_dim]
 * O:             [batch * num_q_heads, head_dim]
 * partial_O:     [batch * num_q_heads * S, head_dim] scratch buffer
 * partial_lse:   [2 * batch * num_q_heads * S] scratch buffer (max + sum)
 * max_kv_len:    allocated KV dimension
 * head_dim:      dimension per head
 * kv_len:        actual KV sequence length (used if kv_len_ptr is null)
 * kv_len_ptr:    GPU-resident int32; if non-null, read at runtime for CUDA
 *                graph compatibility
 * num_q_heads:   query heads per batch element
 * num_kv_heads:  KV heads per batch element
 * chunk_size:    number of KV positions per thread block
 */
cudaError_t flash_decode_splitkv_f32(
    const float* Q, const float* K, const float* V, float* O,
    float* partial_O, float* partial_lse,
    int num_bh, int max_kv_len, int head_dim,
    int kv_len, const int* kv_len_ptr,
    int num_q_heads, int num_kv_heads,
    int chunk_size,
    cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif /* FLASH_DECODE_H */
