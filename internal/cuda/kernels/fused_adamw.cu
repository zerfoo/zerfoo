// fused_adamw.cu -- CUDA kernel for the on-device AdamW mixed-precision update.
//
// Performs the full AdamW step IN PLACE on device, so param/grad/m never
// round-trip to host (ADR 070 end state, ADR 075 lever L1). The second moment
// (v) is held in float64 to preserve the f32-stability fix from the host
// stepMixedV path: sqrt(v)+eps never collapses to eps under f32 underflow on
// near-zero gradients.
//
// Layout per parameter (all device-resident, allocated once and persisted):
//   param: f32[n]  -- updated in place
//   m:     f32[n]  -- first moment, updated in place
//   v:     f64[n]  -- second moment (sidecar), updated in place
//   grad:  f32[n]  -- read, then zeroed in place
//
// The arithmetic mirrors stepMixedV exactly: the first moment and the param
// step round to f32 the same way the host f64 update does, while v and the
// sqrt(v)+eps denominator stay in f64. Bias-correction is folded into `alpha`
// and `lrWd` on the host and passed in as scalars, matching the host code so
// the equivalence gate holds bit-for-bit modulo f32 rounding.

#include <cuda_runtime.h>

__global__ void kernel_fused_adamw(float* __restrict__ param,
                                   float* __restrict__ m,
                                   double* __restrict__ v,
                                   float* __restrict__ grad,
                                   double beta1, double beta2,
                                   double oneMinusBeta1, double oneMinusBeta2,
                                   double eps, double alpha, double lrWd,
                                   int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    double g = (double)__ldg(&grad[idx]);

    // First moment (rounded to f32, as in the host path: mData[i] = f32(mNew)).
    double mOld = (double)__ldg(&m[idx]);
    double mNew = beta1 * mOld + oneMinusBeta1 * g;
    float mNewF = (float)mNew;
    m[idx] = mNewF;

    // Second moment in f64 (the precision the GPU f32 path needs for stability).
    double vNew = beta2 * v[idx] + oneMinusBeta2 * g * g;
    v[idx] = vNew;

    // Update uses the rounded-to-f32 first moment, matching stepMixedV which
    // reads mNew (the f64 value before rounding) -- see note below. The host
    // code uses `mNew` (f64) in the numerator, so we use mNew (f64) here too.
    double denomI = sqrt(vNew) + eps;
    double update = alpha * mNew / denomI;

    double pv = (double)__ldg(&param[idx]);
    pv = pv - update - lrWd * pv;
    param[idx] = (float)pv;

    // Zero the gradient in place (same buffer, no realloc) -- the host path
    // zeroes grad after reading it; doing it here keeps grad device-resident.
    grad[idx] = 0.0f;
}

// ---------- Launcher function (extern "C" for CGO / purego) ----------
//
// ABI NOTE (critical): zerfoo's purego kernel dispatch (internal/cuda
// ccall / ccall_wrapper) marshals EVERY argument through INTEGER registers
// (AAPCS64 x0-x7 + stack) -- it never populates the NEON v0-v7 FP registers.
// A launcher declared with `double`/`float` params would therefore receive
// garbage in the FP registers and shift the integer args (n, stream) into the
// wrong slots, launching the kernel with a bogus element count -> SIGSEGV.
// So the scalars cross the boundary as their raw IEEE-754 BIT PATTERNS in
// `unsigned long long` (integer) params and are reinterpreted to double here.

static inline double bits_to_double(unsigned long long b) {
    double d;
    __builtin_memcpy(&d, &b, sizeof(d));
    return d;
}

extern "C" {

cudaError_t fused_adamw_f32(float* param, float* m, double* v, float* grad,
                            unsigned long long beta1Bits, unsigned long long beta2Bits,
                            unsigned long long oneMinusBeta1Bits, unsigned long long oneMinusBeta2Bits,
                            unsigned long long epsBits, unsigned long long alphaBits,
                            unsigned long long lrWdBits,
                            int n, cudaStream_t stream) {
    double beta1 = bits_to_double(beta1Bits);
    double beta2 = bits_to_double(beta2Bits);
    double oneMinusBeta1 = bits_to_double(oneMinusBeta1Bits);
    double oneMinusBeta2 = bits_to_double(oneMinusBeta2Bits);
    double eps = bits_to_double(epsBits);
    double alpha = bits_to_double(alphaBits);
    double lrWd = bits_to_double(lrWdBits);

    int block = 256;
    int grid = (n + block - 1) / block;
    kernel_fused_adamw<<<grid, block, 0, stream>>>(
        param, m, v, grad,
        beta1, beta2, oneMinusBeta1, oneMinusBeta2,
        eps, alpha, lrWd, n);
    return cudaGetLastError();
}

} // extern "C"
