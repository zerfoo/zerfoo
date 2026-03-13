/* Flash attention forward kernel (float32).
 *
 * Implements the FlashAttention-2 algorithm: tiled computation of
 * softmax(Q*K^T / sqrt(d)) * V in O(n) memory with shared memory staging.
 *
 * Each thread block handles one (batch, head, query_tile) slice. It iterates
 * over KV tiles, accumulating the softmax numerator and denominator online
 * (log-sum-exp trick) so the full S = Q*K^T matrix is never materialized.
 *
 * Tile size BLOCK_SIZE controls shared memory usage. On sm_121 (Blackwell
 * GB10, 228KB shared memory per SM) we use BLOCK_SIZE=64 with dynamic shared
 * memory (64KB). On older architectures with the 48KB static limit we use
 * BLOCK_SIZE=32 (32KB). Pass -DFLASH_BLOCK_SIZE=64 when compiling for sm_121.
 */

#include "flash_attention.h"
#include <float.h>
#include <math.h>

/* Tile size for sequence dimension. Each block processes BLOCK_SIZE query rows.
 * Default 32 (32 * 128 * 4 * 2 = 32KB). Set FLASH_BLOCK_SIZE=64 for sm_121
 * (64 * 128 * 4 * 2 = 64KB, within 228KB shared memory capacity). */
#ifndef FLASH_BLOCK_SIZE
#define FLASH_BLOCK_SIZE 32
#endif
#define BLOCK_SIZE FLASH_BLOCK_SIZE

/* Maximum head dimension supported. Adjust if models exceed this. */
#define MAX_HEAD_DIM 128

/* Kernel: one block per (batch, head, query_tile).
 * Shared memory is allocated dynamically to support tile sizes > 48KB. */
__global__ void flash_attention_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    int seq_len, int head_dim, int causal)
{
    /* Identify which (batch, head, query_tile) this block handles. */
    int bh = blockIdx.x;           /* batch * heads index */
    int q_tile = blockIdx.y;       /* which tile of query rows */
    int tid = threadIdx.x;         /* thread within block [0, BLOCK_SIZE) */

    int q_start = q_tile * BLOCK_SIZE;
    if (q_start >= seq_len) return;

    int q_end = min(q_start + BLOCK_SIZE, seq_len);
    int q_count = q_end - q_start;

    /* Base pointers for this (batch, head). */
    int bh_offset = bh * seq_len * head_dim;
    const float* Q_bh = Q + bh_offset;
    const float* K_bh = K + bh_offset;
    const float* V_bh = V + bh_offset;
    float* O_bh = O + bh_offset;

    float scale = rsqrtf((float)head_dim);

    /* Shared memory for K and V tiles.
     * Layout: sK[BLOCK_SIZE * head_dim] followed by sV[BLOCK_SIZE * head_dim].
     * Using dynamic shared memory allows tile sizes > 48KB static limit. */
    extern __shared__ float smem[];
    float* sK = smem;
    float* sV = smem + BLOCK_SIZE * head_dim;

    /* Per-thread accumulators for one query row (if tid < q_count). */
    float row_max = -FLT_MAX;
    float row_sum = 0.0f;
    float acc[MAX_HEAD_DIM];
    for (int d = 0; d < head_dim; d++) {
        acc[d] = 0.0f;
    }

    /* Load this thread's query row into registers. */
    float q_row[MAX_HEAD_DIM];
    if (tid < q_count) {
        int q_idx = q_start + tid;
        for (int d = 0; d < head_dim; d++) {
            q_row[d] = Q_bh[q_idx * head_dim + d] * scale;
        }
    }

    /* Iterate over KV tiles. */
    int num_kv_tiles = (seq_len + BLOCK_SIZE - 1) / BLOCK_SIZE;

    /* For causal masking, we only need tiles where k_start <= q_end - 1. */
    int max_kv_tile = causal ? min(num_kv_tiles, (q_start + BLOCK_SIZE - 1) / BLOCK_SIZE + 1) : num_kv_tiles;

    for (int kv_tile = 0; kv_tile < max_kv_tile; kv_tile++) {
        int k_start = kv_tile * BLOCK_SIZE;
        int k_end = min(k_start + BLOCK_SIZE, seq_len);
        int k_count = k_end - k_start;

        /* Cooperatively load K and V tile into shared memory.
         * With BLOCK_SIZE > warp size, multiple threads load in parallel. */
        for (int row = tid; row < k_count; row += BLOCK_SIZE) {
            int k_idx = k_start + row;
            for (int d = 0; d < head_dim; d++) {
                sK[row * head_dim + d] = K_bh[k_idx * head_dim + d];
                sV[row * head_dim + d] = V_bh[k_idx * head_dim + d];
            }
        }
        __syncthreads();

        if (tid < q_count) {
            int q_idx = q_start + tid;

            /* Compute attention scores for this tile: s[j] = dot(q_row, sK[j]). */
            for (int j = 0; j < k_count; j++) {
                int k_idx = k_start + j;

                /* Causal mask: skip if key position > query position. */
                if (causal && k_idx > q_idx) continue;

                float s = 0.0f;
                for (int d = 0; d < head_dim; d++) {
                    s += q_row[d] * sK[j * head_dim + d];
                }

                /* Online softmax update (log-sum-exp trick). */
                float prev_max = row_max;
                if (s > row_max) {
                    row_max = s;
                }

                /* Rescale previous accumulator. */
                float exp_diff = expf(prev_max - row_max);
                row_sum = row_sum * exp_diff + expf(s - row_max);

                /* Rescale existing output accumulator. */
                for (int d = 0; d < head_dim; d++) {
                    acc[d] = acc[d] * exp_diff + expf(s - row_max) * sV[j * head_dim + d];
                }
            }
        }

        __syncthreads();
    }

    /* Write final output: O[q] = acc / row_sum. */
    if (tid < q_count) {
        int q_idx = q_start + tid;
        float inv_sum = (row_sum > 0.0f) ? (1.0f / row_sum) : 0.0f;
        for (int d = 0; d < head_dim; d++) {
            O_bh[q_idx * head_dim + d] = acc[d] * inv_sum;
        }
    }
}

extern "C" cudaError_t flash_attention_forward_f32(
    const float* Q, const float* K, const float* V, float* O,
    int batch, int heads, int seq_len, int head_dim,
    int causal, cudaStream_t stream)
{
    if (head_dim > MAX_HEAD_DIM) {
        return cudaErrorInvalidValue;
    }

    int num_bh = batch * heads;
    int num_q_tiles = (seq_len + BLOCK_SIZE - 1) / BLOCK_SIZE;

    dim3 grid(num_bh, num_q_tiles);
    dim3 block(BLOCK_SIZE);

    /* Dynamic shared memory: sK[BLOCK_SIZE * head_dim] + sV[BLOCK_SIZE * head_dim]. */
    size_t smem = 2 * BLOCK_SIZE * head_dim * sizeof(float);

    /* For tile sizes requiring >48KB, opt in to extended shared memory. */
    if (smem > 48 * 1024) {
        cudaFuncSetAttribute(flash_attention_kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             (int)smem);
    }

    flash_attention_kernel<<<grid, block, smem, stream>>>(
        Q, K, V, O, seq_len, head_dim, causal);

    return cudaGetLastError();
}
