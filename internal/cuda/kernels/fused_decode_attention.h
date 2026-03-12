/* Fused single-token decode attention kernel interface.
 *
 * Replaces QK^T + ScaledSoftmax + attn*V with a single kernel launch.
 * For decode (seqLen_q=1) with GQA broadcasting.
 */
#ifndef FUSED_DECODE_ATTENTION_H
#define FUSED_DECODE_ATTENTION_H

#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

/* fused_decode_attention_f32 runs fused single-token attention.
 *
 * Q:          [numQHeads, headDim] device pointer.
 * K:          [kvSeqLen, headDim] device pointer.
 * V:          [kvSeqLen, headDim] device pointer.
 * O:          [numQHeads, headDim] device pointer (output).
 * numQHeads:  number of query heads.
 * kvSeqLen:   cached key/value sequence length.
 * headDim:    dimension per head.
 * scale_bits: 1/sqrt(headDim) as IEEE 754 uint32 bits.
 * stream:     CUDA stream (pass NULL for default stream).
 */
cudaError_t fused_decode_attention_f32(
    const float* Q, const float* K, const float* V, float* O,
    int numQHeads, int kvSeqLen, int headDim,
    unsigned int scale_bits,
    cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif /* FUSED_DECODE_ATTENTION_H */
