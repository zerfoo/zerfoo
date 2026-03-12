/* Q8_0 dequant-GEMM kernel for Zerfoo's Q8Storage format.
 *
 * Each Q8_0 block (36 bytes per 32 values):
 *   bytes[0:4]  = float32 scale (little-endian IEEE 754)
 *   bytes[4:36] = 32 x int8 quantized values
 *   Dequant: val = int8_val * scale
 *
 * Two kernels:
 *   gemv_q8_kernel -- optimized for N=1 (single-token generation).
 *     Loads input vector into shared memory, uses warp-per-row with
 *     warp shuffle reduction for the dot product.
 *   gemm_q8_kernel -- general GEMM fallback for N>1 (prompt processing).
 */

#include <cuda_runtime.h>
#include <stdint.h>

#define Q8_BLOCK_SIZE 32
#define Q8_BLOCK_BYTES 36

/* ---------- Optimized GEMV kernel (N=1) ---------- */
#define Q8_WARPS_PER_BLOCK 8
#define Q8_WARP_SIZE 32

__global__ void gemv_q8_kernel(
    const uint8_t* __restrict__ W_q8,
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

    int warp_id = threadIdx.x / Q8_WARP_SIZE;
    int lane_id = threadIdx.x % Q8_WARP_SIZE;
    int row = blockIdx.x * Q8_WARPS_PER_BLOCK + warp_id;

    if (row >= M) return;

    int blocks_per_row = K / Q8_BLOCK_SIZE;
    const uint8_t* row_data = W_q8 + (size_t)row * blocks_per_row * Q8_BLOCK_BYTES;

    float acc = 0.0f;

    /* Each lane handles a strided subset of Q8 blocks. */
    for (int bi = lane_id; bi < blocks_per_row; bi += Q8_WARP_SIZE) {
        const uint8_t* blk = row_data + bi * Q8_BLOCK_BYTES;

        /* Read float32 scale (4 bytes, little-endian). */
        float scale;
        memcpy(&scale, blk, 4);

        int k_base = bi * Q8_BLOCK_SIZE;
        const int8_t* qvals = (const int8_t*)(blk + 4);

        #pragma unroll
        for (int j = 0; j < Q8_BLOCK_SIZE; j++) {
            acc += (float)qvals[j] * scale * sx[k_base + j];
        }
    }

    /* Warp shuffle reduction. */
    #pragma unroll
    for (int offset = Q8_WARP_SIZE / 2; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
    }

    if (lane_id == 0) {
        y[row] = acc;
    }
}

/* ---------- General GEMM kernel (N>1) ---------- */
#define Q8_TILE_M 16
#define Q8_TILE_N 16

__global__ void gemm_q8_kernel(
    const uint8_t* __restrict__ A_q8,
    const float*   __restrict__ B,
    float*         __restrict__ C,
    int M, int K, int N)
{
    int row = blockIdx.y * Q8_TILE_M + threadIdx.y;
    int col = blockIdx.x * Q8_TILE_N + threadIdx.x;

    if (row >= M || col >= N) return;

    int blocks_per_row = K / Q8_BLOCK_SIZE;
    float acc = 0.0f;

    const uint8_t* row_blocks = A_q8 + (size_t)row * blocks_per_row * Q8_BLOCK_BYTES;

    for (int bi = 0; bi < blocks_per_row; bi++) {
        const uint8_t* blk = row_blocks + bi * Q8_BLOCK_BYTES;

        float scale;
        memcpy(&scale, blk, 4);

        int k_base = bi * Q8_BLOCK_SIZE;
        const int8_t* qvals = (const int8_t*)(blk + 4);

        #pragma unroll
        for (int j = 0; j < Q8_BLOCK_SIZE; j++) {
            float v = (float)qvals[j] * scale;
            acc += v * B[(k_base + j) * N + col];
        }
    }

    C[row * N + col] = acc;
}

/* ---------- Dispatcher ---------- */

extern "C" cudaError_t gemm_q8_f32(
    const void* A_q8, const float* B, float* C,
    int M, int K, int N,
    cudaStream_t stream)
{
    if (K % Q8_BLOCK_SIZE != 0) {
        return cudaErrorInvalidValue;
    }

    if (N == 1) {
        /* GEMV fast path: y = W * x. */
        int threads = Q8_WARPS_PER_BLOCK * Q8_WARP_SIZE;  /* 256 */
        int grid = (M + Q8_WARPS_PER_BLOCK - 1) / Q8_WARPS_PER_BLOCK;
        int smem = K * sizeof(float);

        gemv_q8_kernel<<<grid, threads, smem, stream>>>(
            (const uint8_t*)A_q8, B, C, M, K);
    } else {
        dim3 block(Q8_TILE_N, Q8_TILE_M);
        dim3 grid_dim((N + Q8_TILE_N - 1) / Q8_TILE_N, (M + Q8_TILE_M - 1) / Q8_TILE_M);

        gemm_q8_kernel<<<grid_dim, block, 0, stream>>>(
            (const uint8_t*)A_q8, B, C, M, K, N);
    }

    return cudaGetLastError();
}
