/* Custom SGEMV kernel for M=1 single-token decode.
 *
 * Computes y[M] = A[M x N] * x[N] where A is row-major FP32.
 * Eliminates cuBLAS per-call overhead for the M=1 decode case.
 *
 * Strategy:
 *   - Load x into shared memory (all threads cooperate, x is reused across rows).
 *   - Each thread block processes ROWS_PER_BLOCK output rows.
 *   - Each warp handles one row: lanes compute strided partial dot products
 *     using vectorized float4 loads where possible, then warp-shuffle reduce.
 */

#include <cuda_runtime.h>

#define BLOCK_SIZE       256
#define WARP_SIZE        32
#define WARPS_PER_BLOCK  (BLOCK_SIZE / WARP_SIZE)  /* 8 */
#define ROWS_PER_BLOCK   WARPS_PER_BLOCK

__global__ void sgemv_m1_kernel(
    float*       __restrict__ y,
    const float* __restrict__ A,
    const float* __restrict__ x,
    int M, int N)
{
    extern __shared__ float sx[];

    /* Cooperatively load x into shared memory. */
    for (int i = threadIdx.x; i < N; i += BLOCK_SIZE) {
        sx[i] = x[i];
    }
    __syncthreads();

    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    int row = blockIdx.x * ROWS_PER_BLOCK + warp_id;

    if (row >= M) return;

    const float* row_ptr = A + (size_t)row * N;
    float acc = 0.0f;

    /* Vectorized float4 path for the bulk of the row. */
    int n4 = N / 4;
    const float4* row4 = (const float4*)row_ptr;
    const float4* sx4  = (const float4*)sx;

    for (int i = lane_id; i < n4; i += WARP_SIZE) {
        float4 a4 = __ldg(&row4[i]);
        float4 x4 = sx4[i];
        acc = __fmaf_rn(a4.x, x4.x, acc);
        acc = __fmaf_rn(a4.y, x4.y, acc);
        acc = __fmaf_rn(a4.z, x4.z, acc);
        acc = __fmaf_rn(a4.w, x4.w, acc);
    }

    /* Handle remainder elements (N not divisible by 4). */
    int rem_start = n4 * 4;
    for (int i = rem_start + lane_id; i < N; i += WARP_SIZE) {
        acc = __fmaf_rn(row_ptr[i], sx[i], acc);
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

extern "C" cudaError_t launch_sgemv_m1(
    float* y, const float* A, const float* x,
    int M, int N, cudaStream_t stream)
{
    int grid = (M + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;
    int smem = N * sizeof(float);

    sgemv_m1_kernel<<<grid, BLOCK_SIZE, smem, stream>>>(y, A, x, M, N);

    return cudaGetLastError();
}
