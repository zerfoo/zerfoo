/* Flash attention forward kernel (float32) -- HIP port.
 *
 * Direct port of the CUDA flash_attention.cu kernel to HIP.
 * Uses hipLaunchKernelGGL instead of <<<>>> syntax.
 */

#include "flash_attention.h"
#include <float.h>
#include <math.h>

#define BLOCK_SIZE 64
#define MAX_HEAD_DIM 128

__global__ void flash_attention_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    int seq_len, int head_dim, int causal)
{
    int bh = blockIdx.x;
    int q_tile = blockIdx.y;
    int tid = threadIdx.x;

    int q_start = q_tile * BLOCK_SIZE;
    if (q_start >= seq_len) return;

    int q_end = min(q_start + BLOCK_SIZE, seq_len);
    int q_count = q_end - q_start;

    int bh_offset = bh * seq_len * head_dim;
    const float* Q_bh = Q + bh_offset;
    const float* K_bh = K + bh_offset;
    const float* V_bh = V + bh_offset;
    float* O_bh = O + bh_offset;

    float scale = rsqrtf((float)head_dim);

    __shared__ float sK[BLOCK_SIZE][MAX_HEAD_DIM];
    __shared__ float sV[BLOCK_SIZE][MAX_HEAD_DIM];

    float row_max = -FLT_MAX;
    float row_sum = 0.0f;
    float acc[MAX_HEAD_DIM];
    for (int d = 0; d < head_dim; d++) {
        acc[d] = 0.0f;
    }

    float q_row[MAX_HEAD_DIM];
    if (tid < q_count) {
        int q_idx = q_start + tid;
        for (int d = 0; d < head_dim; d++) {
            q_row[d] = Q_bh[q_idx * head_dim + d] * scale;
        }
    }

    int num_kv_tiles = (seq_len + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int max_kv_tile = causal ? min(num_kv_tiles, (q_start + BLOCK_SIZE - 1) / BLOCK_SIZE + 1) : num_kv_tiles;

    for (int kv_tile = 0; kv_tile < max_kv_tile; kv_tile++) {
        int k_start = kv_tile * BLOCK_SIZE;
        int k_end = min(k_start + BLOCK_SIZE, seq_len);
        int k_count = k_end - k_start;

        if (tid < k_count) {
            int k_idx = k_start + tid;
            for (int d = 0; d < head_dim; d++) {
                sK[tid][d] = K_bh[k_idx * head_dim + d];
                sV[tid][d] = V_bh[k_idx * head_dim + d];
            }
        }
        __syncthreads();

        if (tid < q_count) {
            int q_idx = q_start + tid;

            for (int j = 0; j < k_count; j++) {
                int k_idx = k_start + j;
                if (causal && k_idx > q_idx) continue;

                float s = 0.0f;
                for (int d = 0; d < head_dim; d++) {
                    s += q_row[d] * sK[j][d];
                }

                float prev_max = row_max;
                if (s > row_max) {
                    row_max = s;
                }

                float exp_diff = expf(prev_max - row_max);
                row_sum = row_sum * exp_diff + expf(s - row_max);

                for (int d = 0; d < head_dim; d++) {
                    acc[d] = acc[d] * exp_diff + expf(s - row_max) * sV[j][d];
                }
            }
        }

        __syncthreads();
    }

    if (tid < q_count) {
        int q_idx = q_start + tid;
        float inv_sum = (row_sum > 0.0f) ? (1.0f / row_sum) : 0.0f;
        for (int d = 0; d < head_dim; d++) {
            O_bh[q_idx * head_dim + d] = acc[d] * inv_sum;
        }
    }
}

extern "C" hipError_t flash_attention_forward_f32(
    const float* Q, const float* K, const float* V, float* O,
    int batch, int heads, int seq_len, int head_dim,
    int causal, hipStream_t stream)
{
    if (head_dim > MAX_HEAD_DIM) {
        return hipErrorInvalidValue;
    }

    int num_bh = batch * heads;
    int num_q_tiles = (seq_len + BLOCK_SIZE - 1) / BLOCK_SIZE;

    dim3 grid(num_bh, num_q_tiles);
    dim3 block(BLOCK_SIZE);

    hipLaunchKernelGGL(flash_attention_kernel, grid, block, 0, stream,
                       Q, K, V, O, seq_len, head_dim, causal);

    return hipGetLastError();
}
