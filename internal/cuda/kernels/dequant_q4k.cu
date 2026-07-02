/* Q4_K dequantization kernel: converts Q4_K super-blocks to FP32.
 *
 * Used for non-GEMV MatMul (batch>1) where the fused GEMV kernel cannot
 * be used. Dequantizes to global memory F32 so cuBLAS Sgemm can operate
 * on the result.
 *
 * Each thread block processes one super-block (256 values).
 * 256 threads per block, one value per thread.
 */

#include "dequant_q4k.h"
#include <cuda_fp16.h>
#include <stdint.h>

#define Q4K_SUPER_BLOCK_SIZE 256
#define Q4K_BLOCK_BYTES      144

/* Decode 6-bit scales and mins from the 12-byte packed region.
 * Matches tensor/quantized_kquant.go decodeQ4KScalesMins and gemv_q4k.cu. */
__device__ __forceinline__ void dq4k_decode_scales_mins(
    const uint8_t* sc,
    float d, float dmin,
    float* __restrict__ sub_scales,
    float* __restrict__ sub_mins)
{
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        sub_scales[i] = d * (float)(sc[i] & 63);
        sub_mins[i]   = dmin * (float)(sc[4+i] & 63);
    }
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        sub_scales[4+i] = d * (float)((sc[8+i] & 0xF) | ((sc[i] >> 6) << 4));
        sub_mins[4+i]   = dmin * (float)((sc[8+i] >> 4) | ((sc[4+i] >> 6) << 4));
    }
}

__global__ void dequant_q4k_kernel(
    const uint8_t* __restrict__ src,
    float*         __restrict__ dst,
    int total_blocks)
{
    int block_idx = blockIdx.x;
    if (block_idx >= total_blocks) return;

    int tid = threadIdx.x;  /* 0..255 */

    const uint8_t* blk = src + (size_t)block_idx * Q4K_BLOCK_BYTES;

    /* Read fp16 d and dmin. */
    float d    = __half2float(__ldg((const __half*)(blk)));
    float dmin = __half2float(__ldg((const __half*)(blk + 2)));

    /* Decode sub-block scales and mins (shared across all 256 threads). */
    /* Use shared memory to avoid redundant decoding. */
    __shared__ float s_scales[8];
    __shared__ float s_mins[8];

    if (tid < 8) {
        float sub_scales[8], sub_mins[8];
        dq4k_decode_scales_mins(blk + 4, d, dmin, sub_scales, sub_mins);
        s_scales[tid] = sub_scales[tid];
        s_mins[tid]   = sub_mins[tid];
    }
    __syncthreads();

    /* Each thread dequantizes one value.
     * tid maps to: group = tid / 64, position within group = tid % 64.
     * Within group: positions 0..31 use low nibble (sub-block 2*group),
     *               positions 32..63 use high nibble (sub-block 2*group+1). */
    int group = tid / 64;
    int pos   = tid % 64;

    const uint8_t* qdata = blk + 16;
    float val;

    if (pos < 32) {
        /* Low nibble path: sub-block sb0 = 2*group. */
        int sb = group * 2;
        uint8_t q = __ldg(&qdata[group * 32 + pos]);
        val = s_scales[sb] * (float)(q & 0xF) - s_mins[sb];
    } else {
        /* High nibble path: sub-block sb1 = 2*group+1. */
        int sb = group * 2 + 1;
        uint8_t q = __ldg(&qdata[group * 32 + (pos - 32)]);
        val = s_scales[sb] * (float)(q >> 4) - s_mins[sb];
    }

    int out_idx = block_idx * Q4K_SUPER_BLOCK_SIZE + tid;
    dst[out_idx] = val;
}

extern "C" cudaError_t dequant_q4k_f32(
    const void* src, float* dst,
    int rows, int K,
    cudaStream_t stream)
{
    if (K % Q4K_SUPER_BLOCK_SIZE != 0) {
        return cudaErrorInvalidValue;
    }

    int blocks_per_row = K / Q4K_SUPER_BLOCK_SIZE;
    int total_blocks = rows * blocks_per_row;

    /* One thread block per super-block, 256 threads each. */
    dequant_q4k_kernel<<<total_blocks, Q4K_SUPER_BLOCK_SIZE, 0, stream>>>(
        (const uint8_t*)src, dst, total_blocks);

    return cudaGetLastError();
}
