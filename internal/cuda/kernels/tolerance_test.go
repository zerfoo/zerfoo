package kernels

// gemvReductionRelTol is the standing relative-error gate for the GEMV
// kernel family's fp32 accumulation (sgemv_m1.cu, gemv_q4k.cu /
// gemv_q4k_sm121.cu). See docs/kernel-tolerances.md for the full per-op
// tolerance table and the T135.3 oracle-gate sweep it belongs to
// (docs/plan-gpu-training-hardening.md T3.3).
//
// Both kernels reduce K/N elements with a FIXED, deterministic order (each
// warp lane sequentially accumulates a strided subset, then a 5-level
// warp-shuffle tree combines the 32 lane partials) -- this is not
// nondeterministic accumulation. It is, however, a DIFFERENT valid
// parenthesization of the sum than the naive left-to-right CPU reference
// (cpuSgemv / buildQ4KTestData's float64 reference) used by these tests, so
// fp32 rounding differs between the two orders. The synthetic sin()-based
// test data also produces occasional near-zero row sums (catastrophic
// cancellation), which inflates RELATIVE error for those rows even though
// absolute error stays tiny.
//
// Measured on the GB10 (2026-07-03, T135.3, ref 1082cced): max observed
// relative error 7.55e-4 at K=512 (TestGemvQ4KF32_MultipleSizes/medium_64x512)
// and 2.36e-4 at N=512 (TestSgemvM1_MultipleSizes/medium_128x512), both well
// under 1e-3. A real kernel bug (wrong index, dropped term, a fast-math-class
// blowup like the tanh overflow in ztensor#125) produces errors one to
// several orders of magnitude larger and stays caught at this bar.
const gemvReductionRelTol = 1e-3
