/* Q4_0 dequant-GEMM kernel for Zerfoo's Q4Storage format.
 *
 * Each Q4_0 block (18 bytes):
 *   bytes[0:2] = float16 scale (little-endian)
 *   bytes[2:18] = 32 x 4-bit unsigned values packed (2 per byte, low nibble first)
 *   Dequant: val = (nibble - 8) * scale
 *
 * Kernel strategy: each thread computes one C[row, col].
 * Iterate over K in steps of 32 (one Q4 block per step).
 * Dequantize the block into registers and accumulate dot product.
 */

#include "gemm_q4.h"
#include <cuda_fp16.h>
#include <stdint.h>

#define Q4_BLOCK_SIZE 32
#define Q4_BLOCK_BYTES 18
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

    /* Each row of A has blocks_per_row Q4 blocks (18 bytes each). */
    const uint8_t* row_blocks = A_q4 + (size_t)row * blocks_per_row * Q4_BLOCK_BYTES;

    for (int bi = 0; bi < blocks_per_row; bi++) {
        const uint8_t* blk = row_blocks + bi * Q4_BLOCK_BYTES;

        /* Read float16 scale (2 bytes, little-endian). */
        uint16_t scale_bits = (uint16_t)blk[0] | ((uint16_t)blk[1] << 8);
        float scale = __half2float(*reinterpret_cast<const __half*>(&scale_bits));

        /* Dequantize 32 values and dot-product with B column. */
        int k_base = bi * Q4_BLOCK_SIZE;
        const uint8_t* packed = blk + 2;

        /* GGML Q4_0 split format: low nibbles -> positions 0-15,
         * high nibbles -> positions 16-31. */
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

extern "C" cudaError_t gemm_q4_f32(
    const void* A_q4, const float* B, float* C,
    int M, int K, int N,
    cudaStream_t stream)
{
    if (K % Q4_BLOCK_SIZE != 0) {
        return cudaErrorInvalidValue;
    }

    dim3 block(TILE_N, TILE_M);
    dim3 grid((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);

    gemm_q4_kernel<<<grid, block, 0, stream>>>(
        (const uint8_t*)A_q4, B, C, M, K, N);

    return cudaGetLastError();
}
