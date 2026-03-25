/* Q6_K fused dequant-GEMV kernel for single-token decode (batch=1).
 *
 * Reads Q6_K super-blocks directly, dequantizes in registers (no global
 * memory intermediary), multiplies by the activation vector, and accumulates
 * in FP32. This halves memory traffic compared to separate dequant + GEMV.
 *
 * Q6_K super-block (210 bytes, 256 values):
 *   [0:128]   ql     -- low 4 bits of each 6-bit value
 *   [128:192] qh     -- high 2 bits of each 6-bit value
 *   [192:208] sc     -- int8 scales for 16 sub-blocks of 16 values
 *   [208:210] fp16 d -- super-block scale
 *
 * Dequantization (matching tensor/quantized_kquant.go DequantizeQ6K):
 *   Two 128-element halves. For each half (offset qlOff, qhOff, scOff, outOff):
 *     For l in 0..31:
 *       is = l / 16
 *       q1 = int8((ql[qlOff+l] & 0xF) | ((qh[qhOff+l] & 3) << 4)) - 32
 *       q2 = int8((ql[qlOff+32+l] & 0xF) | (((qh[qhOff+l]>>2) & 3) << 4)) - 32
 *       q3 = int8((ql[qlOff+l] >> 4) | (((qh[qhOff+l]>>4) & 3) << 4)) - 32
 *       q4 = int8((ql[qlOff+32+l] >> 4) | (((qh[qhOff+l]>>6) & 3) << 4)) - 32
 *       dst[outOff+l]    = d * sc[scOff+is+0] * q1
 *       dst[outOff+32+l] = d * sc[scOff+is+2] * q2
 *       dst[outOff+64+l] = d * sc[scOff+is+4] * q3
 *       dst[outOff+96+l] = d * sc[scOff+is+6] * q4
 */

#include "gemv_q6k.h"
#include <cuda_fp16.h>
#include <stdint.h>

#define Q6K_SUPER_BLOCK_SIZE 256
#define Q6K_BLOCK_BYTES      210
#define Q6K_WARPS_PER_BLOCK  4
#define Q6K_WARP_SIZE        32

/* ---------- Fused GEMV kernel ----------
 *
 * y[row] = sum_k dequant(W_q6k[row, k]) * x[k]
 *
 * Strategy:
 *   - Load input vector x into shared memory (all threads cooperate).
 *   - One warp per row for simplicity and good occupancy.
 *   - Each lane processes a strided subset of super-blocks.
 *   - Within each super-block, two 128-element halves are processed.
 *   - Warp shuffle reduction produces the final dot product.
 */
__global__ void gemv_q6k_kernel(
    const uint8_t* __restrict__ W_q6k,
    const float*   __restrict__ x,
    float*         __restrict__ y,
    int M, int K)
{
    extern __shared__ float sx[];

    /* Cooperatively load x into shared memory. */
    int threads_per_block = blockDim.x;
    for (int i = threadIdx.x; i < K; i += threads_per_block) {
        sx[i] = x[i];
    }
    __syncthreads();

    int warp_id = threadIdx.x / Q6K_WARP_SIZE;
    int lane_id = threadIdx.x % Q6K_WARP_SIZE;
    int row = blockIdx.x * Q6K_WARPS_PER_BLOCK + warp_id;

    if (row >= M) return;

    int blocks_per_row = K / Q6K_SUPER_BLOCK_SIZE;
    const uint8_t* row_data = W_q6k + (size_t)row * blocks_per_row * Q6K_BLOCK_BYTES;

    float acc = 0.0f;

    /* Each lane handles a strided subset of super-blocks. */
    for (int bi = lane_id; bi < blocks_per_row; bi += Q6K_WARP_SIZE) {
        const uint8_t* blk = row_data + bi * Q6K_BLOCK_BYTES;

        const uint8_t* ql = blk;         /* [0:128] low 4 bits */
        const uint8_t* qh = blk + 128;   /* [128:192] high 2 bits */
        const int8_t*  sc = (const int8_t*)(blk + 192); /* [192:208] int8 scales */
        float d = __half2float(__ldg((const __half*)(blk + 208)));

        int k_base = bi * Q6K_SUPER_BLOCK_SIZE;

        /* Process two 128-element halves. */
        #pragma unroll
        for (int half = 0; half < 2; half++) {
            int qlOff = half * 64;
            int qhOff = half * 32;
            int scOff = half * 8;
            int outOff = k_base + half * 128;

            #pragma unroll
            for (int l = 0; l < 32; l++) {
                int is = l / 16; /* sub-block offset within each group of 32 */

                uint8_t ql_v  = __ldg(&ql[qlOff + l]);
                uint8_t ql_v2 = __ldg(&ql[qlOff + 32 + l]);
                uint8_t qh_v  = __ldg(&qh[qhOff + l]);

                float s0 = d * (float)sc[scOff + is + 0];
                float s2 = d * (float)sc[scOff + is + 2];
                float s4 = d * (float)sc[scOff + is + 4];
                float s6 = d * (float)sc[scOff + is + 6];

                /* q1: low nibble of ql[qlOff+l] + bits 0-1 of qh */
                int8_t q1 = (int8_t)((ql_v & 0xF) | ((qh_v & 3) << 4)) - 32;
                /* q2: low nibble of ql[qlOff+32+l] + bits 2-3 of qh */
                int8_t q2 = (int8_t)((ql_v2 & 0xF) | (((qh_v >> 2) & 3) << 4)) - 32;
                /* q3: high nibble of ql[qlOff+l] + bits 4-5 of qh */
                int8_t q3 = (int8_t)((ql_v >> 4) | (((qh_v >> 4) & 3) << 4)) - 32;
                /* q4: high nibble of ql[qlOff+32+l] + bits 6-7 of qh */
                int8_t q4 = (int8_t)((ql_v2 >> 4) | (((qh_v >> 6) & 3) << 4)) - 32;

                acc = __fmaf_rn(s0 * (float)q1, sx[outOff + l],      acc);
                acc = __fmaf_rn(s2 * (float)q2, sx[outOff + 32 + l], acc);
                acc = __fmaf_rn(s4 * (float)q3, sx[outOff + 64 + l], acc);
                acc = __fmaf_rn(s6 * (float)q4, sx[outOff + 96 + l], acc);
            }
        }
    }

    /* Warp shuffle reduction. */
    #pragma unroll
    for (int offset = Q6K_WARP_SIZE / 2; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
    }

    if (lane_id == 0) {
        y[row] = acc;
    }
}

/* ---------- Dispatcher ---------- */

extern "C" cudaError_t gemv_q6k_f32(
    const void* W_q6k, const float* x, float* y,
    int M, int K,
    cudaStream_t stream)
{
    if (K % Q6K_SUPER_BLOCK_SIZE != 0) {
        return cudaErrorInvalidValue;
    }

    int threads = Q6K_WARPS_PER_BLOCK * Q6K_WARP_SIZE;  /* 128 */
    int grid = (M + Q6K_WARPS_PER_BLOCK - 1) / Q6K_WARPS_PER_BLOCK;
    int smem = K * sizeof(float);

    gemv_q6k_kernel<<<grid, threads, smem, stream>>>(
        (const uint8_t*)W_q6k, x, y, M, K);

    return cudaGetLastError();
}
