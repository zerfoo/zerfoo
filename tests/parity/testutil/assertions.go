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

// AssertClose compares output data against expected with tolerance.
func AssertClose(t *testing.T, label string, got, want []float32, tol float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: length mismatch: got %d, want %d", label, len(got), len(want))
	}
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
