/* Split-KV flash decode CUDA kernel (float32).
 *
 * For autoregressive decode (seqLen_Q = 1), splits the KV cache across
 * S thread blocks per (batch, head). Each block computes partial attention
 * over its chunk of the KV sequence using online softmax. A second kernel
 * reduces the partial results across blocks using log-sum-exp correction.
 *
 * Two kernels:
 * 1. flash_decode_partial_v2_kernel — each block processes one chunk of KV.
 *    Stores partial output (unnormalized), local max, and local sum per block.
 * 2. flash_decode_reduce_v2_kernel — merges partial outputs across S blocks
 *    for each (batch, head) using rescaling by exp(local_max - global_max).
 *
 * Grid for partial: [numHeads, S] where S = ceil(seqLen_KV / chunk_size).
 * Grid for reduce:  [numHeads].
 */

#include "flash_decode.h"
#include <float.h>
#include <math.h>

/* Maximum head dimension (shared-memory based). */
#define FD_MAX_HEAD_DIM 256

/* Threads per block for partial kernel. */
#ifndef FD_PARTIAL_WARPS
#define FD_PARTIAL_WARPS 4
#endif

#define FD_WARP_SIZE 32
#define FD_PARTIAL_THREADS (FD_PARTIAL_WARPS * FD_WARP_SIZE)

/* Maximum number of splits for the reduce kernel. */
#define FD_MAX_SPLITS 128

/* Threads per block for reduce kernel. */
#define FD_REDUCE_THREADS 128

/* ================================================================
 * Partial kernel — one block per (head, chunk).
 *
 * Each thread block processes KV positions [chunk_id * chunk_size,
 * min((chunk_id+1) * chunk_size, kv_len)). Multiple warps tile the
 * chunk with interleaved assignment. Each warp maintains its own
 * partial softmax state; a cross-warp reduction within the block
 * produces the block-level partial output and log-sum-exp.
 *
 * Shared memory layout:
 *   sQ[head_dim]                              — scaled query
 *   warp_max[FD_PARTIAL_WARPS]                — per-warp softmax max
 *   warp_sum[FD_PARTIAL_WARPS]                — per-warp softmax sum
 *   warp_acc[FD_PARTIAL_WARPS * head_dim]     — per-warp V accumulator
 * ================================================================ */
__global__ void flash_decode_partial_v2_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ partial_O,
    float* __restrict__ partial_max,
    float* __restrict__ partial_sum,
    int max_kv_len, int head_dim, int kv_dim,
    int kv_len_param,
    const int* __restrict__ kv_len_ptr,
    int num_q_heads, int num_kv_heads,
    int chunk_size, int num_splits)
{
    int bh = blockIdx.x;
    int chunk_id = blockIdx.y;
    int tid = threadIdx.x;
    int warp_id = tid / FD_WARP_SIZE;
    int lane_id = tid % FD_WARP_SIZE;

    int kv_len = kv_len_ptr ? *kv_len_ptr : kv_len_param;
    int partial_idx = bh * num_splits + chunk_id;

    /* Chunk boundaries. */
    int chunk_start = chunk_id * chunk_size;
    if (chunk_start >= kv_len) {
        if (tid == 0) {
            partial_max[partial_idx] = -FLT_MAX;
            partial_sum[partial_idx] = 0.0f;
        }
        for (int d = tid; d < head_dim; d += FD_PARTIAL_THREADS) {
            partial_O[partial_idx * head_dim + d] = 0.0f;
        }
        return;
    }
    int chunk_end = min(chunk_start + chunk_size, kv_len);

    /* GQA head mapping. */
    int head_ratio = num_q_heads / num_kv_heads;
    int batch_idx = bh / num_q_heads;
    int q_head = bh % num_q_heads;
    int kv_head = q_head / head_ratio;

    const float* Q_bh = Q + bh * head_dim;
    const float* K_base = K + batch_idx * max_kv_len * kv_dim + kv_head * head_dim;
    const float* V_base = V + batch_idx * max_kv_len * kv_dim + kv_head * head_dim;

    float scale = rsqrtf((float)head_dim);

    extern __shared__ float smem[];
    float* sQ = smem;
    float* warp_max_arr = smem + head_dim;
    float* warp_sum_arr = warp_max_arr + FD_PARTIAL_WARPS;
    float* warp_acc = warp_sum_arr + FD_PARTIAL_WARPS;

    /* Load and scale Q. */
    for (int d = tid; d < head_dim; d += FD_PARTIAL_THREADS) {
        sQ[d] = Q_bh[d] * scale;
    }
    for (int d = tid; d < FD_PARTIAL_WARPS * head_dim; d += FD_PARTIAL_THREADS) {
        warp_acc[d] = 0.0f;
    }
    __syncthreads();

    float local_max = -FLT_MAX;
    float local_sum = 0.0f;
    float* my_acc = warp_acc + warp_id * head_dim;

    for (int j = chunk_start + warp_id; j < chunk_end; j += FD_PARTIAL_WARPS) {
        float partial = 0.0f;
        for (int d = lane_id; d < head_dim; d += FD_WARP_SIZE) {
            partial += sQ[d] * K_base[j * kv_dim + d];
        }
        for (int offset = FD_WARP_SIZE / 2; offset > 0; offset >>= 1) {
            partial += __shfl_down_sync(0xFFFFFFFF, partial, offset);
        }
        float s = __shfl_sync(0xFFFFFFFF, partial, 0);

        float prev_max = local_max;
        if (s > local_max) local_max = s;
        float exp_diff = expf(prev_max - local_max);
        float exp_s = expf(s - local_max);
        local_sum = local_sum * exp_diff + exp_s;

        for (int d = lane_id; d < head_dim; d += FD_WARP_SIZE) {
            my_acc[d] = my_acc[d] * exp_diff + exp_s * V_base[j * kv_dim + d];
        }
    }

    if (lane_id == 0) {
        warp_max_arr[warp_id] = local_max;
        warp_sum_arr[warp_id] = local_sum;
    }
    __syncthreads();

    /* Cross-warp reduction. */
    float global_max_block = warp_max_arr[0];
    for (int w = 1; w < FD_PARTIAL_WARPS; w++) {
        if (warp_max_arr[w] > global_max_block) global_max_block = warp_max_arr[w];
    }

    float my_warp_max = warp_max_arr[warp_id];
    float my_warp_sum = warp_sum_arr[warp_id];
    float rescale = expf(my_warp_max - global_max_block);
    float scaled_sum = my_warp_sum * rescale;

    for (int d = lane_id; d < head_dim; d += FD_WARP_SIZE) {
        my_acc[d] *= rescale;
    }
    if (lane_id == 0) {
        warp_sum_arr[warp_id] = scaled_sum;
    }
    __syncthreads();

    float block_sum = 0.0f;
    for (int w = 0; w < FD_PARTIAL_WARPS; w++) {
        block_sum += warp_sum_arr[w];
    }

    /* Store unnormalized partial output, local max, and local sum. */
    for (int d = tid; d < head_dim; d += FD_PARTIAL_THREADS) {
        float total = 0.0f;
        for (int w = 0; w < FD_PARTIAL_WARPS; w++) {
            total += warp_acc[w * head_dim + d];
        }
        partial_O[partial_idx * head_dim + d] = total;
    }
    if (tid == 0) {
        partial_max[partial_idx] = global_max_block;
        partial_sum[partial_idx] = block_sum;
    }
}

/* ================================================================
 * Reduce kernel — merges partial outputs across S splits.
 *
 * One block per (batch, head). Each thread handles a stripe of head_dim.
 * Reads S partial outputs and their max/sum values, then combines
 * them using the standard rescaling formula:
 *   O = sum_s(exp(max_s - global_max) * partial_O_s)
 *     / sum_s(exp(max_s - global_max) * sum_s)
 * ================================================================ */
__global__ void flash_decode_reduce_v2_kernel(
    const float* __restrict__ partial_O,
    const float* __restrict__ partial_max,
    const float* __restrict__ partial_sum,
    float* __restrict__ O,
    int head_dim, int num_splits)
{
    int bh = blockIdx.x;
    int tid = threadIdx.x;

    /* Find global max across all splits. */
    float global_max = -FLT_MAX;
    for (int s = 0; s < num_splits; s++) {
        float m = partial_max[bh * num_splits + s];
        if (m > global_max) global_max = m;
    }

    /* Compute total denominator: sum_s(exp(max_s - global_max) * sum_s). */
    __shared__ float s_inv_denom;
    if (tid == 0) {
        float denom = 0.0f;
        for (int s = 0; s < num_splits; s++) {
            int idx = bh * num_splits + s;
            denom += expf(partial_max[idx] - global_max) * partial_sum[idx];
        }
        s_inv_denom = (denom > 0.0f) ? (1.0f / denom) : 0.0f;
    }
    __syncthreads();

    float inv_denom = s_inv_denom;

    /* Accumulate rescaled partial outputs and normalize. */
    for (int d = tid; d < head_dim; d += FD_REDUCE_THREADS) {
        float acc = 0.0f;
        for (int s = 0; s < num_splits; s++) {
            int idx = bh * num_splits + s;
            float w = expf(partial_max[idx] - global_max);
            acc += w * partial_O[idx * head_dim + d];
        }
        O[bh * head_dim + d] = acc * inv_denom;
    }
}

/* ---- C entry point ---- */

extern "C" cudaError_t flash_decode_splitkv_f32(
    const float* Q, const float* K, const float* V, float* O,
    float* partial_O, float* partial_lse,
    int num_bh, int max_kv_len, int head_dim,
    int kv_len, const int* kv_len_ptr,
    int num_q_heads, int num_kv_heads,
    int chunk_size,
    cudaStream_t stream)
{
    if (head_dim > FD_MAX_HEAD_DIM) {
        return cudaErrorInvalidValue;
    }

    int num_splits = (kv_len + chunk_size - 1) / chunk_size;
    if (num_splits < 1) num_splits = 1;
    if (num_splits > FD_MAX_SPLITS) {
        return cudaErrorInvalidValue;
    }

    int kv_dim = num_kv_heads * head_dim;

    /* partial_lse is repurposed as two contiguous arrays:
     * partial_max: partial_lse[0 .. num_bh*num_splits)
     * partial_sum: partial_lse[num_bh*num_splits .. 2*num_bh*num_splits) */
    float* p_max = partial_lse;
    float* p_sum = partial_lse + num_bh * num_splits;

    /* --- Launch partial kernel --- */
    {
        dim3 grid(num_bh, num_splits);
        dim3 block(FD_PARTIAL_THREADS);

        size_t smem = (head_dim + 2 * FD_PARTIAL_WARPS +
                       FD_PARTIAL_WARPS * head_dim) * sizeof(float);

        if (smem > 48 * 1024) {
            cudaFuncSetAttribute(flash_decode_partial_v2_kernel,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 (int)smem);
        }

        flash_decode_partial_v2_kernel<<<grid, block, smem, stream>>>(
            Q, K, V, partial_O, p_max, p_sum,
            max_kv_len, head_dim, kv_dim,
            kv_len, kv_len_ptr,
            num_q_heads, num_kv_heads,
            chunk_size, num_splits);
    }

    /* --- Launch reduce kernel --- */
    {
        dim3 grid(num_bh);
        dim3 block(FD_REDUCE_THREADS);

        flash_decode_reduce_v2_kernel<<<grid, block, sizeof(float), stream>>>(
            partial_O, p_max, p_sum, O, head_dim, num_splits);
    }

    return cudaGetLastError();
}
