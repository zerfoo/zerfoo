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

/* Maximum head dimension for the prefill kernel (register-limited). */
#define MAX_HEAD_DIM 128

/* Maximum head dimension for the decode kernel (shared-memory based, no
 * large register arrays). 256 covers Gemma 3 (key_length=256). */
#define MAX_HEAD_DIM_DECODE 256

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

/* -----------------------------------------------------------------------
 * Decode-specific attention kernel with GPU-resident KV sequence length
 * and GQA support.
 *
 * NOTE: This kernel is currently not called from Go. It was disabled because
 * it caused a 51% regression (234 -> 114 tok/s on Gemma 3 1B) compared to
 * cuBLAS SDPA at kv_len >= 256. The Go-side fast path was removed in the
 * T1001.2 revert. The kernel is retained here for future optimization
 * (e.g., warp-level parallelism, shared-memory tiling over KV positions).
 *
 * During autoregressive decode, Q has 1 row per (batch, query_head) while
 * K/V are stored in the KV cache with heads packed in the dim dimension.
 *
 * Layouts:
 *   Q:  [batch * num_q_heads, 1, head_dim]        (separated by head)
 *   K:  [batch, max_kv_len, num_kv_heads * head_dim]  (heads packed in dim)
 *   V:  [batch, max_kv_len, num_kv_heads * head_dim]
 *   O:  [batch * num_q_heads, 1, head_dim]
 *
 * kv_dim:     num_kv_heads * head_dim (stride between KV positions).
 * kv_len_ptr: GPU-resident int32 pointer. If non-null, the kernel reads
 *             the actual KV length from *kv_len_ptr at runtime (not frozen
 *             by CUDA graph capture). If null, kv_len_param is used.
 *
 * GQA: each query head maps to a KV head via kv_head = q_head / head_ratio.
 * The kernel indexes into the packed KV dim: K_base + kv_head * head_dim.
 * Between positions the stride is kv_dim (not head_dim).
 *
 * Design: One thread block per (batch, query_head). Q is loaded into shared
 * memory. KV is iterated position by position. For each position, all
 * threads cooperatively compute the dot product via parallel reduction.
 * Thread 0 does the online softmax update and broadcasts rescaling factors.
 * All threads update the shared-memory V accumulator. Finally thread 0
 * writes the normalized output.
 *
 * This avoids large per-thread register arrays and works with head_dim up
 * to 256 without register spilling.
 * ----------------------------------------------------------------------- */
__global__ void flash_attention_decode_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    int max_kv_len, int head_dim, int kv_dim,
    int kv_len_param,
    const int* __restrict__ kv_len_ptr,
    int num_q_heads, int num_kv_heads)
{
    int bh = blockIdx.x;    /* (batch, query_head) index */
    int tid = threadIdx.x;  /* thread within block [0, BLOCK_SIZE) */

    /* Read KV length from GPU memory if pointer is provided. */
    int kv_len = kv_len_ptr ? *kv_len_ptr : kv_len_param;

    /* Compute KV head index for GQA. When num_q_heads == num_kv_heads,
     * head_ratio=1 and kv_head == q_head (backward compatible). */
    int head_ratio = num_q_heads / num_kv_heads;
    int batch_idx = bh / num_q_heads;
    int q_head = bh % num_q_heads;
    int kv_head = q_head / head_ratio;

    /* Q/O base: single row at [bh, 0, :] (Q has num_q_heads per batch) */
    const float* Q_bh = Q + bh * head_dim;
    float* O_bh = O + bh * head_dim;

    /* K/V base: heads are packed in dim. For batch_idx b, kv_head h:
     * base = K + b * max_kv_len * kv_dim + h * head_dim
     * Position stride between K[pos] and K[pos+1] is kv_dim. */
    const float* K_base = K + batch_idx * max_kv_len * kv_dim + kv_head * head_dim;
    const float* V_base = V + batch_idx * max_kv_len * kv_dim + kv_head * head_dim;

    float scale = rsqrtf((float)head_dim);

    /* Shared memory layout:
     *   sQ[head_dim]                          -- scaled query
     *   sAcc[head_dim]                        -- softmax-weighted V accumulator
     *   sPartial[BLOCK_SIZE]                  -- dot-product partial sums
     */
    extern __shared__ float smem[];
    float* sQ = smem;
    float* sAcc = smem + head_dim;
    float* sPartial = smem + 2 * head_dim;

    /* Load and scale Q into shared memory cooperatively. */
    for (int d = tid; d < head_dim; d += BLOCK_SIZE) {
        sQ[d] = Q_bh[d] * scale;
    }
    /* Zero the accumulator. */
    for (int d = tid; d < head_dim; d += BLOCK_SIZE) {
        sAcc[d] = 0.0f;
    }
    __syncthreads();

    /* Thread 0 tracks the online softmax state. */
    float row_max = -FLT_MAX;
    float row_sum = 0.0f;

    /* Iterate over KV positions. Stride between positions is kv_dim. */
    for (int j = 0; j < kv_len; j++) {
        /* Step 1: parallel dot product Q * K[j]. Each thread handles a
         * stripe of the head dimension. */
        float partial = 0.0f;
        for (int d = tid; d < head_dim; d += BLOCK_SIZE) {
            partial += sQ[d] * K_base[j * kv_dim + d];
        }
        sPartial[tid] = partial;
        __syncthreads();

        /* Tree reduction to get the full dot product in sPartial[0]. */
        for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                sPartial[tid] += sPartial[tid + stride];
            }
            __syncthreads();
        }

        /* Step 2: thread 0 does online softmax update. */
        if (tid == 0) {
            float s = sPartial[0];
            float prev_max = row_max;
            if (s > row_max) row_max = s;

            float exp_diff = expf(prev_max - row_max);
            float exp_s = expf(s - row_max);
            row_sum = row_sum * exp_diff + exp_s;

            /* Rescale existing accumulator and add weighted V[j].
             * Done cooperatively below after broadcasting exp_diff, exp_s. */
            sPartial[0] = exp_diff;
            sPartial[1] = exp_s;
        }
        __syncthreads();

        /* Step 3: all threads update accumulator cooperatively. */
        {
            float exp_diff = sPartial[0];
            float exp_s = sPartial[1];
            for (int d = tid; d < head_dim; d += BLOCK_SIZE) {
                sAcc[d] = sAcc[d] * exp_diff + exp_s * V_base[j * kv_dim + d];
            }
        }
        __syncthreads();
    }

    /* Write output: O = sAcc / row_sum. */
    float inv_sum;
    if (tid == 0) {
        inv_sum = (row_sum > 0.0f) ? (1.0f / row_sum) : 0.0f;
        /* Reuse sPartial[0] to broadcast inv_sum. */
        sPartial[0] = inv_sum;
    }
    __syncthreads();
    inv_sum = sPartial[0];
    for (int d = tid; d < head_dim; d += BLOCK_SIZE) {
        O_bh[d] = sAcc[d] * inv_sum;
    }
}

extern "C" cudaError_t flash_attention_decode_f32(
    const float* Q, const float* K, const float* V, float* O,
    int num_bh, int max_kv_len, int head_dim,
    int kv_len, const int* kv_len_ptr,
    int num_q_heads, int num_kv_heads,
    cudaStream_t stream)
{
    if (head_dim > MAX_HEAD_DIM_DECODE) {
        return cudaErrorInvalidValue;
    }

    int kv_dim = num_kv_heads * head_dim;

    /* One block per (batch, query_head). num_bh = batch * num_q_heads. */
    dim3 grid(num_bh);
    dim3 block(BLOCK_SIZE);

    /* Shared memory: sQ[head_dim] + sAcc[head_dim] + sPartial[BLOCK_SIZE]. */
    size_t smem = (2 * head_dim + BLOCK_SIZE) * sizeof(float);

    if (smem > 48 * 1024) {
        cudaFuncSetAttribute(flash_attention_decode_kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             (int)smem);
    }

    flash_attention_decode_kernel<<<grid, block, smem, stream>>>(
        Q, K, V, O, max_kv_len, head_dim, kv_dim, kv_len, kv_len_ptr,
        num_q_heads, num_kv_heads);

    return cudaGetLastError();
}
