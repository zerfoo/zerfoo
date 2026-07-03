package kernels

import (
	"math"
	"testing"
)

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

// checkGemvRelError scans the FULL output array against the reference and
// reports the TRUE maximum relative error, then asserts it once against tol.
//
// The original per-element loop called t.Errorf + `break` at the first
// offending index, which meant maxRelErr only ever reflected the error at
// (or before) that first-broken element -- NOT the true dataset-wide max.
// That under-reporting bug surfaced directly during T135.3: raising the
// tolerance from 1e-4 to 1e-3 changed which element the loop broke on
// (a smaller-index, smaller-error element no longer tripped the earlier,
// tighter bar), which made the logged "max relative error" jump around
// between runs and looked like kernel nondeterminism. It was not -- the
// kernel is bit-reproducible; only the test's early-exit reporting was
// unstable. Scan to completion so the reported max is honest regardless of
// where the tolerance line sits.
func checkGemvRelError(t *testing.T, got, ref []float32, tol float64) {
	t.Helper()

	maxRelErr := 0.0
	maxIdx := -1
	badCount := 0
	const maxReported = 5
	for i := range got {
		absRef := math.Abs(float64(ref[i]))
		diff := math.Abs(float64(got[i] - ref[i]))
		var relErr float64
		if absRef > 1e-6 {
			relErr = diff / absRef
		} else {
			relErr = diff
		}
		if relErr > maxRelErr {
			maxRelErr = relErr
			maxIdx = i
		}
		if relErr > tol {
			badCount++
			if badCount <= maxReported {
				t.Errorf("y[%d] = %f, want %f (rel err %e)", i, got[i], ref[i], relErr)
			}
		}
	}
	if badCount > maxReported {
		t.Errorf("... and %d more elements exceeded tol %e", badCount-maxReported, tol)
	}
	t.Logf("max relative error: %e (at index %d)", maxRelErr, maxIdx)
}
