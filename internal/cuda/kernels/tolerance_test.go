package kernels

import (
	"math"
	"testing"
)

// gemvReductionAbsTol and gemvReductionRelTol are the standing
// numpy-allclose-style tolerance gate for the GEMV kernel family's fp32
// accumulation (sgemv_m1.cu, gemv_q4k.cu / gemv_q4k_sm121.cu): an element
// passes when
//
//	|got - want| <= gemvReductionAbsTol + gemvReductionRelTol*|want|
//
// See docs/kernel-tolerances.md for the full per-op tolerance table and the
// T135.3 oracle-gate sweep it belongs to (docs/plan-gpu-training-hardening.md
// T3.3).
//
// Both kernels reduce K/N elements with a FIXED, deterministic order (each
// warp lane sequentially accumulates a strided subset, then a 5-level
// warp-shuffle tree combines the 32 lane partials) -- this is not
// nondeterministic accumulation. It is, however, a DIFFERENT valid
// parenthesization of the sum than the naive left-to-right CPU/float64
// reference (cpuSgemv / buildQ4KTestData's reference) used by these tests, so
// fp32 rounding legitimately differs between the two orders.
//
// A pure RELATIVE bound is the wrong shape for this failure mode: the
// synthetic sin()-based test data produces occasional near-zero row sums
// (catastrophic cancellation), and for those rows the ABSOLUTE error stays a
// few micro-units while the RELATIVE error explodes because the denominator
// (the reference value) is itself tiny. Measured on the GB10 (2026-07-03,
// T135.3, ref 08531b5f), the true (full-array, not first-failure) worst
// cases were:
//   - TestSgemvM1_MultipleSizes/large_4096x4096: y[3862] rel err 7.32e-3,
//     but |diff| = 5e-6 against want=6.30e-4.
//   - TestSgemvM1_MultipleSizes/gemma3_1b_6144x1536: y[1791] rel err 3.96e-3,
//     |diff| = 6e-6 against want=-1.521e-3.
//   - TestGemvQ4KF32_MultipleSizes/medium_64x512: rel err 7.55e-4,
//     |diff| ~ 6e-7 against want=7.94e-4.
//
// In every case |diff| stayed at or below ~6e-6. gemvReductionAbsTol=1e-5
// covers all of them with margin while gemvReductionRelTol=1e-4 keeps the
// original tight relative bound for normal-magnitude elements (at |want|~1,
// the combined bound is ~1.1e-4, essentially unchanged from the original flat
// 1e-4 test). A real kernel bug (wrong index, dropped term, a
// fast-math-class blowup like the tanh overflow in ztensor#125) produces
// absolute errors many orders of magnitude above 1e-5 and stays caught.
const (
	gemvReductionAbsTol = 1e-5
	gemvReductionRelTol = 1e-4
)

// checkGemvRelError scans the FULL output array against the reference using
// the combined absolute+relative bound (see gemvReductionAbsTol /
// gemvReductionRelTol above), reports the true maximum relative error found,
// and asserts once at the end.
//
// The original per-element loop called t.Errorf + `break` at the first
// offending index, which meant the logged "max relative error" only ever
// reflected the error at (or before) that first-broken element -- NOT the
// true dataset-wide max. That under-reporting bug surfaced directly during
// T135.3 tolerance tuning: changing the bound changed WHICH element the loop
// broke on, so the logged max jumped between runs in a way that looked like
// kernel nondeterminism but was actually just the test giving up early at a
// different point each time (the kernel itself is bit-reproducible). Scan to
// completion so the reported max is honest regardless of where the
// tolerance line sits.
func checkGemvRelError(t *testing.T, got, ref []float32, absTol, relTol float64) {
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

		if diff > absTol+relTol*absRef {
			badCount++
			if badCount <= maxReported {
				t.Errorf("y[%d] = %f, want %f (diff %e > %e + %e*%e)",
					i, got[i], ref[i], diff, absTol, relTol, absRef)
			}
		}
	}
	if badCount > maxReported {
		t.Errorf("... and %d more elements exceeded tol", badCount-maxReported)
	}
	t.Logf("max relative error: %e (at index %d)", maxRelErr, maxIdx)
}
