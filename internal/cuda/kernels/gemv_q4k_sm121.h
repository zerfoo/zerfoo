/* Q4_K GEMV kernel optimized for sm_121 (Blackwell GB10 / DGX Spark).
 *
 * sm_121 architecture improvements over the baseline:
 *   - 8 warps per block (256 threads) for higher occupancy on Blackwell's
 *     128-SM die; 4 warps filled only half the warp slots.
 *   - Vectorized 128-bit loads via uint4 for the Q4_K data region to align
 *     with Blackwell's L2 cache line (128 bytes) and reduce load transactions.
 *   - __ldcg (cache-global) on activation vector loads to keep x in L2 across
 *     multiple GEMV calls (decode loop reuses x across layer rows).
 *   - Warp-level reduction via __shfl_down_sync with full-mask (same as
 *     baseline) — already optimal; cooperative groups add no benefit here.
 *   - Block-level partial sums through shared memory allow 8 warps to all
 *     contribute, avoiding the 1-warp-per-row serialization of the baseline.
 *
 * Same external C interface as gemv_q4k.h.
 *
 * Computes: y[m] = sum_k( dequant(W_q4k[m,k]) * x[k] )
 */

#ifndef GEMV_Q4K_SM121_H
#define GEMV_Q4K_SM121_H

#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

/* gemv_q4k_sm121_f32 — sm_121 optimized Q4_K fused dequant-GEMV.
 *
 * Same semantics as gemv_q4k_f32.  K must be a multiple of 256.
 * On non-sm_121 hardware the dispatcher will fall back to the baseline kernel.
 */
cudaError_t gemv_q4k_sm121_f32(
    const void* W_q4k, const float* x, float* y,
    int M, int K,
    cudaStream_t stream);

/* gemv_q4k_check_sm121 returns 1 if the current device is sm_121. */
int gemv_q4k_check_sm121(void);

#ifdef __cplusplus
}
#endif

#endif /* GEMV_Q4K_SM121_H */
