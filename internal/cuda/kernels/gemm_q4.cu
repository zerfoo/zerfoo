/* Q4_0 dequant-GEMM kernel for Zerfoo's GPU-optimized Q4 layout.
 *
 * Global separated layout (repacked from GGUF at upload time):
 *   [all_scales: N_blocks * 2 bytes] [pad to 16B] [all_data: N_blocks * 16 bytes]
 *
 * The kernel receives a data_offset parameter that points to the start of
 * the packed data region. Block i's scale is at scales[i] (2 bytes) and
 * block i's packed nibbles are at data_base[i*16] (16 bytes, 16-byte aligned).
 *
 * This layout enables 128-bit (uint4) vectorized loads for the packed data
 * and coalesced 16-bit loads for scales.
 *
 * Two kernels:
 *   gemv_q4_kernel -- optimized for N=1 (single-token generation).
 *   gemm_q4_kernel -- general GEMM fallback for N>1 (prompt processing).
 */

#include "gemm_q4.h"
#include <cuda_fp16.h>
#include <stdint.h>

#define Q4_BLOCK_SIZE 32
#define Q4_BLOCK_BYTES 18

/* ---------- Optimized GEMV kernel (N=1) ----------
 *
 * y[row] = sum_k dequant(W_q4[row, k]) * x[k]
 *
 * Strategy:
 *   - Load input vector x into shared memory (all threads cooperate).
 *   - Each warp computes two output rows for better occupancy.
 *   - Lanes in the warp split Q4 blocks and accumulate partial sums.
 *   - Dual accumulators break FMA dependency chains for ILP.
 *   - Packed data is loaded via 128-bit (uint4) loads for maximum bandwidth.
 *   - Warp shuffle reduction produces the final dot product.
 */
#define WARPS_PER_BLOCK 8
#define ROWS_PER_WARP 2
#define WARP_SIZE 32

__global__ void gemv_q4_kernel(
    const __half*   __restrict__ all_scales,
    const uint8_t*  __restrict__ all_data,
    const float*    __restrict__ x,
    float*          __restrict__ y,
    int M, int K)
{
    extern __shared__ float sx[];

    /* Cooperatively load x[0..K-1] into shared memory. */
    int threads_per_block = blockDim.x;
    for (int i = threadIdx.x; i < K; i += threads_per_block) {
        sx[i] = x[i];
    }
    __syncthreads();

    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    int row_base = blockIdx.x * (WARPS_PER_BLOCK * ROWS_PER_WARP) + warp_id * ROWS_PER_WARP;

    int blocks_per_row = K / Q4_BLOCK_SIZE;

    /* Each warp processes ROWS_PER_WARP rows, reusing shared memory x. */
    #pragma unroll
    for (int r = 0; r < ROWS_PER_WARP; r++) {
        int row = row_base + r;
        if (row >= M) return;

        int row_block_base = row * blocks_per_row;

        /* Dual accumulators for ILP: low nibbles and high nibbles. */
        float acc0 = 0.0f;
        float acc1 = 0.0f;

        /* Each lane handles a strided subset of Q4 blocks. */
        for (int bi = lane_id; bi < blocks_per_row; bi += WARP_SIZE) {
            int block_idx = row_block_base + bi;

            /* Coalesced 16-bit scale load. */
            float scale = __half2float(__ldg(&all_scales[block_idx]));

            int k_base = bi * Q4_BLOCK_SIZE;

            /* 128-bit vectorized load of 16 packed bytes (always 16-byte aligned). */
            uint4 pv = __ldg((const uint4*)(all_data + block_idx * 16));
            const uint8_t* p = (const uint8_t*)&pv;

            #pragma unroll
            for (int j = 0; j < 16; j++) {
                uint8_t bv = p[j];
                float dq0 = (float)((int)(bv & 0x0F) - 8) * scale;
                float dq1 = (float)((int)(bv >> 4) - 8) * scale;

                acc0 = __fmaf_rn(dq0, sx[k_base + j], acc0);
                acc1 = __fmaf_rn(dq1, sx[k_base + j + 16], acc1);
            }
        }

        float acc = acc0 + acc1;

        /* Warp shuffle reduction. */
        #pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
        }

        if (lane_id == 0) {
            y[row] = acc;
        }
    }
}

/* ---------- General GEMM kernel (N>1) ---------- */
#define TILE_M 16
#define TILE_N 16

__global__ void gemm_q4_kernel(
    const __half*   __restrict__ all_scales,
    const uint8_t*  __restrict__ all_data,
    const float*    __restrict__ B,
    float*          __restrict__ C,
    int M, int K, int N)
{
    int row = blockIdx.y * TILE_M + threadIdx.y;
    int col = blockIdx.x * TILE_N + threadIdx.x;

    if (row >= M || col >= N) return;

    int blocks_per_row = K / Q4_BLOCK_SIZE;
    int row_block_base = row * blocks_per_row;
    float acc = 0.0f;

    for (int bi = 0; bi < blocks_per_row; bi++) {
        int block_idx = row_block_base + bi;

        float scale = __half2float(__ldg(&all_scales[block_idx]));

        int k_base = bi * Q4_BLOCK_SIZE;

        /* 128-bit vectorized load. */
        uint4 pv = __ldg((const uint4*)(all_data + block_idx * 16));
        const uint8_t* p = (const uint8_t*)&pv;

        #pragma unroll
        for (int j = 0; j < 16; j++) {
            uint8_t byte_val = p[j];
            int q0 = (int)(byte_val & 0x0F) - 8;
            int q1 = (int)(byte_val >> 4) - 8;

            float v0 = (float)q0 * scale;
            float v1 = (float)q1 * scale;

            acc += v0 * B[(k_base + j) * N + col];
            acc += v1 * B[(k_base + j + 16) * N + col];
        }
    }

    C[row * N + col] = acc;
}

/* ---------- Dispatcher ---------- */

extern "C" cudaError_t gemm_q4_f32(
    const void* A_q4, const float* B, float* C,
    int M, int K, int N,
    int data_offset,
    cudaStream_t stream)
{
    if (K % Q4_BLOCK_SIZE != 0) {
        return cudaErrorInvalidValue;
    }

    const __half*  all_scales = (const __half*)A_q4;
    const uint8_t* all_data   = (const uint8_t*)A_q4 + data_offset;

    if (N == 1) {
        /* GEMV fast path: y = W * x. */
        int threads = WARPS_PER_BLOCK * WARP_SIZE;  /* 256 */
        int rows_per_block = WARPS_PER_BLOCK * ROWS_PER_WARP;
        int grid = (M + rows_per_block - 1) / rows_per_block;
        int smem = K * sizeof(float);

        gemv_q4_kernel<<<grid, threads, smem, stream>>>(
            all_scales, all_data, B, C, M, K);
    } else {
        dim3 block(TILE_N, TILE_M);
        dim3 grid_dim((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);

        gemm_q4_kernel<<<grid_dim, block, 0, stream>>>(
            all_scales, all_data, B, C, M, K, N);
    }

    return cudaGetLastError();
}
