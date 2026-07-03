/* Fused on-device AdamW mixed-precision update kernel interface.
 *
 * Performs the full AdamW step in place on device (ADR 070 end state, ADR 075
 * lever L1): param, m, grad are f32 device buffers; v is an f64 device sidecar.
 * No host<->device round-trip of any optimizer state. The second moment stays
 * in f64 to preserve f32 training stability (the GPU "CrossAsset cliff" fix).
 */
#ifndef FUSED_ADAMW_H
#define FUSED_ADAMW_H

#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

/* fused_adamw_f32 applies one AdamW step in one kernel launch, in place.
 *
 * param:         device pointer to f32[n] -- updated in place.
 * m:             device pointer to f32[n] first moment -- updated in place.
 * v:             device pointer to f64[n] second moment -- updated in place.
 * grad:          device pointer to f32[n] gradient -- read then zeroed in place.
 * The scalar Adam hyperparameters cross as raw IEEE-754 double BIT PATTERNS in
 * unsigned long long (integer) params -- NOT `double` -- because zerfoo's purego
 * kernel dispatch marshals every arg through integer registers and never the
 * NEON FP registers; the launcher reinterprets the bits to double.
 *
 * beta1Bits, beta2Bits:                 bits of the Adam decay rates.
 * oneMinusBeta1Bits, oneMinusBeta2Bits: bits of the precomputed (1-beta) terms.
 * epsBits:    bits of epsilon (added to sqrt(v)).
 * alphaBits:  bits of lr * bias-correction (numer/denom), precomputed on host.
 * lrWdBits:   bits of lr * weightDecay, precomputed on host.
 * n:          number of elements.
 * stream:     CUDA stream (pass NULL for default stream).
 */
cudaError_t fused_adamw_f32(
    float* param, float* m, double* v, float* grad,
    unsigned long long beta1Bits, unsigned long long beta2Bits,
    unsigned long long oneMinusBeta1Bits, unsigned long long oneMinusBeta2Bits,
    unsigned long long epsBits, unsigned long long alphaBits,
    unsigned long long lrWdBits,
    int n, cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif /* FUSED_ADAMW_H */
