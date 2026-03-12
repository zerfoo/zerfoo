// fused_decode_attention.cu -- Fused single-token attention kernel.
//
// For decode step (seqLen_q=1) with GQA broadcasting:
//   scores = Q * K^T * scale    (dot products)
//   weights = softmax(scores)
//   output = weights * V
//
// Replaces 3 kernel launches (MatMulTransposeB + ScaledSoftmax + MatMul)
// with a single kernel launch, saving 2 launches per transformer layer.
//
// Grid: one block per query head.
// Block: BLOCK_SIZE threads (256 default).

#include <cuda_runtime.h>
#include <math.h>
#include <float.h>
#include <string.h>

#define FUSED_ATTN_BLOCK 256

// Each block processes one query head's attention against all KV positions.
// Shared memory layout: [q_cache: headDim] [scores: kvSeqLen]
__global__ void kernel_fused_decode_attention(
    const float* __restrict__ Q,       // [numQHeads, headDim]
    const float* __restrict__ K,       // [kvSeqLen, headDim]
    const float* __restrict__ V,       // [kvSeqLen, headDim]
    float*       __restrict__ O,       // [numQHeads, headDim]
    int kvSeqLen, int headDim, float scale)
{
    int head = blockIdx.x;
    const float* q = Q + head * headDim;
    float* o = O + head * headDim;

    extern __shared__ float smem[];
    float* q_cache = smem;                    // [headDim]
    float* scores  = smem + headDim;          // [kvSeqLen]

    // Step 1: Load Q vector into shared memory.
    for (int d = threadIdx.x; d < headDim; d += blockDim.x) {
        q_cache[d] = q[d];
    }
    __syncthreads();

    // Step 2: Compute scaled dot products Q * K^T.
    // Each thread handles a strided subset of KV positions.
    float local_max = -FLT_MAX;
    for (int j = threadIdx.x; j < kvSeqLen; j += blockDim.x) {
        const float* kj = K + j * headDim;
        float dot = 0.0f;
        for (int d = 0; d < headDim; d++) {
            dot = __fmaf_rn(q_cache[d], kj[d], dot);
        }
        float s = dot * scale;
        scores[j] = s;
        local_max = fmaxf(local_max, s);
    }

    // Reduce max across threads via shared memory.
    // Reuse a small region at the end of smem for the reduction buffer.
    __syncthreads();
    float* reduce_buf = scores + kvSeqLen; // [blockDim.x]
    reduce_buf[threadIdx.x] = local_max;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            reduce_buf[threadIdx.x] = fmaxf(reduce_buf[threadIdx.x],
                                              reduce_buf[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    float max_val = reduce_buf[0];
    __syncthreads();

    // Step 3: Compute exp(score - max) and accumulate sum.
    float local_sum = 0.0f;
    for (int j = threadIdx.x; j < kvSeqLen; j += blockDim.x) {
        float e = expf(scores[j] - max_val);
        scores[j] = e;
        local_sum += e;
    }
    reduce_buf[threadIdx.x] = local_sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            reduce_buf[threadIdx.x] += reduce_buf[threadIdx.x + stride];
        }
        __syncthreads();
    }
    float sum_val = reduce_buf[0];
    __syncthreads();

    // Step 4: Normalize scores (in-place).
    float inv_sum = 1.0f / sum_val;
    for (int j = threadIdx.x; j < kvSeqLen; j += blockDim.x) {
        scores[j] *= inv_sum;
    }
    __syncthreads();

    // Step 5: Compute output = weights * V.
    // Each thread computes one or more dimensions of the output vector.
    for (int d = threadIdx.x; d < headDim; d += blockDim.x) {
        float acc = 0.0f;
        for (int j = 0; j < kvSeqLen; j++) {
            acc = __fmaf_rn(scores[j], V[j * headDim + d], acc);
        }
        o[d] = acc;
    }
}

static inline float bits_to_float(unsigned int bits) {
    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

extern "C" {

// fused_decode_attention_f32 runs fused single-token attention.
//
// Q: [numQHeads, headDim], K: [kvSeqLen, headDim], V: [kvSeqLen, headDim]
// O: [numQHeads, headDim]
// scale_bits: 1/sqrt(headDim) as IEEE 754 uint32 bits.
cudaError_t fused_decode_attention_f32(
    const float* Q, const float* K, const float* V, float* O,
    int numQHeads, int kvSeqLen, int headDim,
    unsigned int scale_bits,
    cudaStream_t stream)
{
    float scale = bits_to_float(scale_bits);
    int block = FUSED_ATTN_BLOCK;
    // Shared memory: q_cache[headDim] + scores[kvSeqLen] + reduce_buf[block]
    size_t smem = (headDim + kvSeqLen + block) * sizeof(float);
    kernel_fused_decode_attention<<<numQHeads, block, smem, stream>>>(
        Q, K, V, O, kvSeqLen, headDim, scale);
    return cudaGetLastError();
}

} // extern "C"
