/* INT8 mixed-precision GEMM kernel (INT8 weights * FP32 activations -> FP32 output).
 *
 * Uses tiled computation with shared memory to reduce global memory bandwidth.
 * Each thread block computes a TILE_M x TILE_N tile of the output matrix C.
 * The inner loop iterates over K in TILE_K-sized chunks, loading A (int8) and
 * B (float32) tiles into shared memory, dequantizing A on the fly.
 */

#include "gemm_int8.h"
#include <stdint.h>

#define TILE_M 32
#define TILE_N 32
#define TILE_K 32

__global__ void gemm_int8_kernel(
    const int8_t* __restrict__ A,
    const float*  __restrict__ B,
    float*        __restrict__ C,
    int M, int K, int N)
{
    int row = blockIdx.y * TILE_M + threadIdx.y;
    int col = blockIdx.x * TILE_N + threadIdx.x;

    __shared__ float sA[TILE_M][TILE_K];
    __shared__ float sB[TILE_K][TILE_N];

    float acc = 0.0f;

    int numTiles = (K + TILE_K - 1) / TILE_K;

    for (int t = 0; t < numTiles; t++) {
        int aCol = t * TILE_K + threadIdx.x;
        if (row < M && aCol < K) {
            sA[threadIdx.y][threadIdx.x] = (float)A[row * K + aCol];
        } else {
            sA[threadIdx.y][threadIdx.x] = 0.0f;
        }

        int bRow = t * TILE_K + threadIdx.y;
        if (bRow < K && col < N) {
            sB[threadIdx.y][threadIdx.x] = B[bRow * N + col];
        } else {
            sB[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        for (int k = 0; k < TILE_K; k++) {
            acc += sA[threadIdx.y][k] * sB[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = acc;
    }
}

extern "C" cudaError_t gemm_int8_f32(
    const void* A, const float* B, float* C,
    int M, int K, int N,
    cudaStream_t stream)
{
    dim3 block(TILE_N, TILE_M);
    dim3 grid((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);

    gemm_int8_kernel<<<grid, block, 0, stream>>>(
        (const int8_t*)A, B, C, M, K, N);

    return cudaGetLastError();
}
