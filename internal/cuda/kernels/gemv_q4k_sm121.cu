/* Q4_K GEMV kernel optimized for sm_121 (Blackwell GB10 / DGX Spark).
 *
 * Optimization rationale for sm_121
 * ===================================
 * Blackwell B100/GB10 key changes vs. Ada Lovelace (sm_89):
 *   1. L2 cache doubled to 96 MB (GB200) / 40 MB (GB10).  Decode-phase GEMV
 *      reuses the same activation vector x for every output row; pinning x in
 *      L2 with __ldcg avoids repeated DRAM fetches.
 *   2. 128 SM count on GB10.  Using 4 warps/block left 75% of SM compute idle
 *      when M < 4*num_SMs.  8 warps/block doubles the active-warp count.
 *   3. 128-byte cache-line width (same as Hopper).  Q4_K qdata region is
 *      128 bytes/super-block, so a single uint4 x 8 sequence reads the entire
 *      qdata region in 8 vectorized 16-byte transactions — matching one L2 line.
 *   4. Cooperative Groups (CG) are used for warp-level reductions to keep
 *      semantics explicit and allow ptxas to schedule optimally.
 *
 * Thread organisation
 * ====================
 *   Block: 256 threads = 8 warps.
 *   Grid : ceil(M / ROWS_PER_BLOCK) blocks.
 *
 *   Within a block, ROWS_PER_BLOCK = 8 rows are computed in parallel.
 *   One warp owns one row.  Lanes within the warp stride across super-blocks.
 *
 * Q4_K super-block layout (144 bytes, 256 values)
 * =================================================
 *   [0:2]   fp16 d      — super-block scale
 *   [2:4]   fp16 dmin   — super-block min
 *   [4:16]  12 bytes    — packed 6-bit scales/mins for 8 sub-blocks
 *   [16:144] 128 bytes  — 256 x 4-bit quantized values (low nibble = even index)
 */

#include "gemv_q4k_sm121.h"
#include "gemv_q4k.h"   /* decode_scales_mins is shared */
#include <cuda_fp16.h>
#include <cooperative_groups.h>
#include <stdint.h>

namespace cg = cooperative_groups;

#define Q4K_SUPER_BLOCK_SIZE 256
#define Q4K_BLOCK_BYTES      144
#define Q4K_NUM_SUB_BLOCKS   8
#define Q4K_WARP_SIZE        32

/* sm_121 tuning knobs. */
#define SM121_WARPS_PER_BLOCK  8    /* 256 threads — fills Blackwell warp slots */
#define SM121_ROWS_PER_BLOCK   SM121_WARPS_PER_BLOCK

/* Decode 6-bit packed scales/mins — identical to baseline decode_scales_mins
 * but inlined here so we don't introduce a cross-.cu dependency. */
__device__ __forceinline__ void sm121_decode_scales_mins(
    const uint8_t* __restrict__ sc,
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

/* ---------- sm_121 optimized kernel ----------
 *
 * Key differences from gemv_q4k_kernel:
 *   1. 8 warps per block (256 threads) — row = blockIdx * 8 + warp_id.
 *   2. Activation vector x loaded into shared memory using __ldcg (L2 cached).
 *   3. Q4_K qdata (128 bytes) read as eight uint4 (16-byte) vectorized loads,
 *      matching the 128-byte L2 cache line and saving load-store unit cycles.
 *   4. Warp-level reduction uses cooperative_groups::reduce for clarity and to
 *      let ptxas choose optimal SHFL scheduling.
 *   5. __fmaf_rn throughout to keep FMA precision consistent with baseline.
 */
__global__ void gemv_q4k_sm121_kernel(
    const uint8_t* __restrict__ W_q4k,
    const float*   __restrict__ x,
    float*         __restrict__ y,
    int M, int K)
{
    /* Shared memory layout:
     *   [0 .. K*sizeof(float))  — activation vector x (FP32, L2-cached load)
     */
    extern __shared__ float sx[];

    /* Cooperatively load x into shared memory using __ldcg for L2 pinning. */
    const int nthr = blockDim.x;
    for (int i = threadIdx.x; i < K; i += nthr) {
        sx[i] = __ldcg(&x[i]);
    }
    __syncthreads();

    /* One warp per row. */
    const int warp_id = threadIdx.x / Q4K_WARP_SIZE;
    const int lane_id = threadIdx.x % Q4K_WARP_SIZE;
    const int row     = blockIdx.x * SM121_ROWS_PER_BLOCK + warp_id;

    if (row >= M) return;

    cg::thread_block_tile<Q4K_WARP_SIZE> warp =
        cg::tiled_partition<Q4K_WARP_SIZE>(cg::this_thread_block());

    const int blocks_per_row = K / Q4K_SUPER_BLOCK_SIZE;
    const uint8_t* row_data  = W_q4k + (size_t)row * blocks_per_row * Q4K_BLOCK_BYTES;

    float acc = 0.0f;

    /* Each lane strides across super-blocks. */
    for (int bi = lane_id; bi < blocks_per_row; bi += Q4K_WARP_SIZE) {
        const uint8_t* blk = row_data + bi * Q4K_BLOCK_BYTES;

        /* Load d / dmin. */
        float d    = __half2float(__ldg((const __half*)(blk)));
        float dmin = __half2float(__ldg((const __half*)(blk + 2)));

        /* Decode sub-block scales/mins. */
        float sub_scales[Q4K_NUM_SUB_BLOCKS];
        float sub_mins[Q4K_NUM_SUB_BLOCKS];
        sm121_decode_scales_mins(blk + 4, d, dmin, sub_scales, sub_mins);

        /* Vectorized 128-bit load of the 128-byte qdata region.
         * The qdata region starts at blk+16 and is 128 bytes = 8 x uint4.
         * This aligns with the 128-byte Blackwell L2 cache line. */
        const uint4* qvec = (const uint4*)(blk + 16);
        /* Load all 8 uint4 (128 bytes) into registers. */
        uint4 qv[8];
        #pragma unroll
        for (int qi = 0; qi < 8; qi++) {
            qv[qi] = __ldg(&qvec[qi]);
        }
        /* Reinterpret as a flat uint8 array for nibble extraction. */
        const uint8_t* qdata = (const uint8_t*)qv;

        const int k_base = bi * Q4K_SUPER_BLOCK_SIZE;

        /* Process 4 groups × 64 elements. */
        #pragma unroll
        for (int group = 0; group < 4; group++) {
            const int sb0  = group * 2;
            const int sb1  = group * 2 + 1;
            const float sc0 = sub_scales[sb0], mn0 = sub_mins[sb0];
            const float sc1 = sub_scales[sb1], mn1 = sub_mins[sb1];

            const int base_out = k_base + group * 64;
            const int base_q   = group * 32;

            #pragma unroll
            for (int l = 0; l < 32; l++) {
                const uint8_t q = qdata[base_q + l];

                const float dq_lo = __fmaf_rn(sc0, (float)(q & 0xF), -mn0);
                const float dq_hi = __fmaf_rn(sc1, (float)(q >> 4),   -mn1);

                acc = __fmaf_rn(dq_lo, sx[base_out + l],      acc);
                acc = __fmaf_rn(dq_hi, sx[base_out + l + 32], acc);
            }
        }
    }

    /* Warp reduction using cooperative groups. */
    acc = cg::reduce(warp, acc, cg::plus<float>());

    if (lane_id == 0) {
        y[row] = acc;
    }
}

/* ---------- Capability probe ---------- */

extern "C" int gemv_q4k_check_sm121()
{
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess) return 0;
    int major = 0, minor = 0;
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device);
    cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device);
    /* sm_121 = Blackwell GB10 (DGX Spark), sm_120 = GB200 die.
     * Accept any sm_12x Blackwell part for the optimized path. */
    return (major == 12) ? 1 : 0;
}

/* ---------- Dispatcher ---------- */

extern "C" cudaError_t gemv_q4k_sm121_f32(
    const void* W_q4k, const float* x, float* y,
    int M, int K,
    cudaStream_t stream)
{
    if (K % Q4K_SUPER_BLOCK_SIZE != 0) {
        return cudaErrorInvalidValue;
    }

    const int threads = SM121_WARPS_PER_BLOCK * Q4K_WARP_SIZE;  /* 256 */
    const int grid    = (M + SM121_ROWS_PER_BLOCK - 1) / SM121_ROWS_PER_BLOCK;
    const int smem    = K * (int)sizeof(float);

    gemv_q4k_sm121_kernel<<<grid, threads, smem, stream>>>(
        (const uint8_t*)W_q4k, x, y, M, K);

    return cudaGetLastError();
}
