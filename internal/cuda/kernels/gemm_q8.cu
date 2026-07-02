/* Q8_0 dequant-GEMM kernel for Zerfoo's Q8Storage format.
 *
 * Each Q8_0 block (36 bytes per 32 values):
 *   bytes[0:4]  = float32 scale (little-endian IEEE 754)
 *   bytes[4:36] = 32 x int8 quantized values
 *   Dequant: val = int8_val * scale
 *
 * Two kernels:
 *   gemv_q8_kernel -- optimized for N=1 (single-token generation).
 *     Uses vectorized int4 loads (16 int8 values at once) and float4
 *     loads from shared memory. Warp-per-row with warp shuffle reduction.
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

    /* Cooperatively load x[0..K-1] into shared memory using float4 loads. */
    int threads_per_block = blockDim.x;
    int k4 = K / 4;
    const float4* x4 = (const float4*)x;
    float4* sx4 = (float4*)sx;
    for (int i = threadIdx.x; i < k4; i += threads_per_block) {
        sx4[i] = __ldg(&x4[i]);
    }
    /* Handle remainder if K is not a multiple of 4. */
    for (int i = k4 * 4 + threadIdx.x; i < K; i += threads_per_block) {
        sx[i] = __ldg(&x[i]);
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

        /* Read float32 scale (4 bytes, little-endian) using __ldg. */
        float scale = __ldg((const float*)blk);

        int k_base = bi * Q8_BLOCK_SIZE;
        const int8_t* qvals = (const int8_t*)(blk + 4);

        /* Vectorized load: read 32 int8 values as two int4 (16 bytes each).
         * int4 is a CUDA vector type: {int x, y, z, w} = 16 bytes. */
        const int4* qv4 = (const int4*)qvals;
        int4 q_lo = __ldg(&qv4[0]);  /* qvals[0..15] */
        int4 q_hi = __ldg(&qv4[1]);  /* qvals[16..31] */

        /* Unpack int4 into individual int8 values and dot with shared mem.
         * Each int4 component (int x,y,z,w) holds 4 int8 values. */
        const int8_t* q_lo_bytes = (const int8_t*)&q_lo;
        const int8_t* q_hi_bytes = (const int8_t*)&q_hi;

        /* Process first 16 values using float4 loads from shared memory. */
        float4 sx0 = ((float4*)&sx[k_base])[0];   /* sx[k_base+0..3] */
        float4 sx1 = ((float4*)&sx[k_base])[1];   /* sx[k_base+4..7] */
        float4 sx2 = ((float4*)&sx[k_base])[2];   /* sx[k_base+8..11] */
        float4 sx3 = ((float4*)&sx[k_base])[3];   /* sx[k_base+12..15] */

        acc += scale * (
            (float)q_lo_bytes[0]  * sx0.x + (float)q_lo_bytes[1]  * sx0.y +
            (float)q_lo_bytes[2]  * sx0.z + (float)q_lo_bytes[3]  * sx0.w +
            (float)q_lo_bytes[4]  * sx1.x + (float)q_lo_bytes[5]  * sx1.y +
            (float)q_lo_bytes[6]  * sx1.z + (float)q_lo_bytes[7]  * sx1.w +
            (float)q_lo_bytes[8]  * sx2.x + (float)q_lo_bytes[9]  * sx2.y +
            (float)q_lo_bytes[10] * sx2.z + (float)q_lo_bytes[11] * sx2.w +
            (float)q_lo_bytes[12] * sx3.x + (float)q_lo_bytes[13] * sx3.y +
            (float)q_lo_bytes[14] * sx3.z + (float)q_lo_bytes[15] * sx3.w);

        /* Process second 16 values. */
        float4 sx4 = ((float4*)&sx[k_base + 16])[0];  /* sx[k_base+16..19] */
        float4 sx5 = ((float4*)&sx[k_base + 16])[1];  /* sx[k_base+20..23] */
        float4 sx6 = ((float4*)&sx[k_base + 16])[2];  /* sx[k_base+24..27] */
        float4 sx7 = ((float4*)&sx[k_base + 16])[3];  /* sx[k_base+28..31] */

        acc += scale * (
            (float)q_hi_bytes[0]  * sx4.x + (float)q_hi_bytes[1]  * sx4.y +
            (float)q_hi_bytes[2]  * sx4.z + (float)q_hi_bytes[3]  * sx4.w +
            (float)q_hi_bytes[4]  * sx5.x + (float)q_hi_bytes[5]  * sx5.y +
            (float)q_hi_bytes[6]  * sx5.z + (float)q_hi_bytes[7]  * sx5.w +
            (float)q_hi_bytes[8]  * sx6.x + (float)q_hi_bytes[9]  * sx6.y +
            (float)q_hi_bytes[10] * sx6.z + (float)q_hi_bytes[11] * sx6.w +
            (float)q_hi_bytes[12] * sx7.x + (float)q_hi_bytes[13] * sx7.y +
            (float)q_hi_bytes[14] * sx7.z + (float)q_hi_bytes[15] * sx7.w);
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

        float scale = __ldg((const float*)blk);

        int k_base = bi * Q8_BLOCK_SIZE;
        const int8_t* qvals = (const int8_t*)(blk + 4);

        #pragma unroll
        for (int j = 0; j < Q8_BLOCK_SIZE; j++) {
            acc += (float)qvals[j] * scale * __ldg(&B[(k_base + j) * N + col]);
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
