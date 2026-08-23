package testutil

import (
	"math"
	"testing"
)

// CompareSlices compares two float32 slices with tolerance.
// Returns the number of mismatches and the max absolute difference.
func CompareSlices(got, want []float32, tol float64) (mismatches int, maxDiff float64) {
	if len(got) != len(want) {
		return len(got) + len(want), math.Inf(1)
	}
	for i := range got {
		diff := math.Abs(float64(got[i] - want[i]))
		if diff > maxDiff {
			maxDiff = diff
		}
		if diff > tol {
			mismatches++
		}
	}
	return
}

// AssertSensitive fails when the tolerance is so wide relative to the expected
// values that an all-zero output would satisfy the comparison. Such a bound
// proves nothing: the layer under test can be deleted outright and the
// assertion still passes.
//
// This is the sensitivity control lore L-0009 asks for, hoisted into the shared
// helper so every golden-driven parity test carries it. T152.3 found exactly
// one golden violating it (ssm_mamba: tolerance 1e-3 against expected values
// whose largest magnitude is 1.13e-4), and TestParity_MambaBlock duly passed
// with MambaBlock.Forward replaced by zeros.
func AssertSensitive(t *testing.T, label string, want []float32, tol float64) {
	t.Helper()
	if len(want) == 0 {
		t.Fatalf("%s: nothing to compare against (empty expected slice)", label)
	}
	maxWant := 0.0
	for _, w := range want {
		if a := math.Abs(float64(w)); a > maxWant {
			maxWant = a
		}
	}
	if maxWant <= tol {
		t.Fatalf("%s: vacuous tolerance: tol=%g is >= the largest expected magnitude (%g), "+
			"so an all-zero output would pass. Size the tolerance to the measured error instead.",
			label, tol, maxWant)
	}
}

// AssertClose compares output data against expected with tolerance. It first
// asserts its own sensitivity (see AssertSensitive) so a tolerance that could
// never fail is reported as a broken test rather than a passing one.
func AssertClose(t *testing.T, label string, got, want []float32, tol float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: length mismatch: got %d, want %d", label, len(got), len(want))
	}
	AssertSensitive(t, label, want, tol)
	mismatches, maxDiff := CompareSlices(got, want, tol)
	if mismatches > 0 {
		// Show first few mismatches.
		shown := 0
		for i := range got {
			diff := math.Abs(float64(got[i] - want[i]))
			if diff > tol {
				t.Errorf("%s[%d]: got %g, want %g (diff=%g)", label, i, got[i], want[i], diff)
				shown++
				if shown >= 5 {
					break
				}
			}
		}
		t.Errorf("%s: %d/%d values exceed tolerance %g (maxDiff=%g)", label, mismatches, len(got), tol, maxDiff)
	}
}
