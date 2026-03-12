/* Q4_0 dequant-GEMM kernel for Zerfoo's Q4Storage format.
 *
 * Each Q4_0 block (18 bytes):
 *   bytes[0:2] = float16 scale (little-endian)
 *   bytes[2:18] = 32 x 4-bit unsigned values packed (2 per byte, low nibble first)
 *   Dequant: val = (nibble - 8) * scale
 *
 * Two kernels:
 *   gemv_q4_kernel -- optimized for N=1 (single-token generation).
 *     Loads input vector into shared memory, uses warp-per-row with
 *     warp shuffle reduction for the dot product.
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
 *   - Each warp computes one output row.
 *   - Lanes in the warp split Q4 blocks and accumulate partial sums.
 *   - Warp shuffle reduction produces the final dot product.
 *
 * WARPS_PER_BLOCK warps per block, grid has ceil(M / WARPS_PER_BLOCK) blocks.
 */
#define WARPS_PER_BLOCK 8
#define WARP_SIZE 32

__global__ void gemv_q4_kernel(
    const uint8_t* __restrict__ W_q4,
    const float*   __restrict__ x,
    float*         __restrict__ y,
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
    int row = blockIdx.x * WARPS_PER_BLOCK + warp_id;

    if (row >= M) return;

    int blocks_per_row = K / Q4_BLOCK_SIZE;
    const uint8_t* row_data = W_q4 + (size_t)row * blocks_per_row * Q4_BLOCK_BYTES;

    float acc = 0.0f;

    /* Each lane handles a strided subset of Q4 blocks. */
    for (int bi = lane_id; bi < blocks_per_row; bi += WARP_SIZE) {
        const uint8_t* blk = row_data + bi * Q4_BLOCK_BYTES;

        uint16_t scale_bits = (uint16_t)blk[0] | ((uint16_t)blk[1] << 8);
        float scale = __half2float(*reinterpret_cast<const __half*>(&scale_bits));

        int k_base = bi * Q4_BLOCK_SIZE;
        const uint8_t* packed = blk + 2;

        #pragma unroll
        for (int j = 0; j < 16; j++) {
            uint8_t bv = packed[j];
            int q0 = (int)(bv & 0x0F) - 8;
            int q1 = (int)(bv >> 4) - 8;

            acc += (float)q0 * scale * sx[k_base + j];
            acc += (float)q1 * scale * sx[k_base + j + 16];
        }
    }

    /* Warp shuffle reduction. */
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
    }

    if (lane_id == 0) {
        y[row] = acc;
    }
}

/* ---------- General GEMM kernel (N>1) ---------- */
#define TILE_M 16
#define TILE_N 16

__global__ void gemm_q4_kernel(
    const uint8_t* __restrict__ A_q4,
    const float*   __restrict__ B,
    float*         __restrict__ C,
    int M, int K, int N)
{
    int row = blockIdx.y * TILE_M + threadIdx.y;
    int col = blockIdx.x * TILE_N + threadIdx.x;

    if (row >= M || col >= N) return;

    int blocks_per_row = K / Q4_BLOCK_SIZE;
    float acc = 0.0f;

    const uint8_t* row_blocks = A_q4 + (size_t)row * blocks_per_row * Q4_BLOCK_BYTES;

    for (int bi = 0; bi < blocks_per_row; bi++) {
        const uint8_t* blk = row_blocks + bi * Q4_BLOCK_BYTES;

        uint16_t scale_bits = (uint16_t)blk[0] | ((uint16_t)blk[1] << 8);
        float scale = __half2float(*reinterpret_cast<const __half*>(&scale_bits));

        int k_base = bi * Q4_BLOCK_SIZE;
        const uint8_t* packed = blk + 2;

        #pragma unroll
        for (int j = 0; j < 16; j++) {
            uint8_t byte_val = packed[j];
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
    cudaStream_t stream)
{
    if (K % Q4_BLOCK_SIZE != 0) {
        return cudaErrorInvalidValue;
    }

    if (N == 1) {
        /* GEMV fast path: y = W * x. */
        int threads = WARPS_PER_BLOCK * WARP_SIZE;  /* 256 */
        int grid = (M + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
        int smem = K * sizeof(float);

        gemv_q4_kernel<<<grid, threads, smem, stream>>>(
            (const uint8_t*)A_q4, B, C, M, K);
    } else {
        dim3 block(TILE_N, TILE_M);
        dim3 grid_dim((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);

        gemm_q4_kernel<<<grid_dim, block, 0, stream>>>(
            (const uint8_t*)A_q4, B, C, M, K, N);
    }

    return cudaGetLastError();
}
