package timeseries

import (
	"strings"
	"testing"

	"github.com/zerfoo/ztensor/tensor"
)

// mustTensor is a tiny test helper that fails the test on construction error.
func mustTensor(t *testing.T, shape []int, data []float32) *tensor.TensorNumeric[float32] {
	t.Helper()
	tt, err := tensor.New[float32](shape, data)
	if err != nil {
		t.Fatalf("tensor.New(shape=%v, len=%d): %v", shape, len(data), err)
	}
	return tt
}

// expectPanic runs fn and returns the recovered value, or fails the test if
// fn did not panic. The substring, when non-empty, must appear in the
// stringified panic value.
func expectPanic(t *testing.T, substr string, fn func()) {
	t.Helper()
	defer func() {
		r := recover()
		if r == nil {
			t.Fatalf("expected panic containing %q, got none", substr)
		}
		if substr == "" {
			return
		}
		msg, ok := r.(string)
		if !ok {
			// fmt.Sprintf panic values are strings, but be defensive.
			t.Fatalf("panic value not a string: %#v", r)
		}
		if !strings.Contains(msg, substr) {
			t.Fatalf("panic message %q does not contain %q", msg, substr)
		}
	}()
	fn()
}

// makeGrads builds a minimal gpuGrads whose allParamTensors() returns exactly
// 5 entries (patchEmbW, patchEmbB, posEmb, headW, headB) — no encoder layers.
// The caller supplies the backing tensors so each test can inject mismatches.
func makeGrads(pw, pb, pe, hw, hb *tensor.TensorNumeric[float32]) *gpuGrads {
	return &gpuGrads{
		patchEmbW: pw,
		patchEmbB: pb,
		posEmb:    pe,
		headW:     hw,
		headB:     hb,
	}
}

// TestGradTsSentinel exercises verifyGradTsAliasing across the four regression
// classes the strengthened sentinel is meant to catch.
func TestGradTsSentinel(t *testing.T) {
	t.Run("HappyPathAliased", func(t *testing.T) {
		pw := mustTensor(t, []int{2, 2}, []float32{1, 2, 3, 4})
		pb := mustTensor(t, []int{2}, []float32{5, 6})
		pe := mustTensor(t, []int{1, 2}, []float32{7, 8})
		hw := mustTensor(t, []int{2, 2}, []float32{9, 10, 11, 12})
		hb := mustTensor(t, []int{2}, []float32{13, 14})

		grads := makeGrads(pw, pb, pe, hw, hb)
		// gradTs is the cached flat slice taken at training start. In the
		// happy path it holds the same wrappers as allParamTensors() and so
		// shares Data()[0] pointers by construction.
		gradTs := grads.allParamTensors()

		// Must not panic.
		verifyGradTsAliasing(grads, gradTs)
	})

	t.Run("ArenaMismatchPanics", func(t *testing.T) {
		pw := mustTensor(t, []int{2, 2}, []float32{1, 2, 3, 4})
		pb := mustTensor(t, []int{2}, []float32{5, 6})
		pe := mustTensor(t, []int{1, 2}, []float32{7, 8})
		hw := mustTensor(t, []int{2, 2}, []float32{9, 10, 11, 12})
		hb := mustTensor(t, []int{2}, []float32{13, 14})
		grads := makeGrads(pw, pb, pe, hw, hb)

		gradTs := grads.allParamTensors()
		// Replace index 3 (headW) with a different tensor of identical shape
		// but an independent backing slice — simulates an arena realloc that
		// left the wrapper identity intact in callers but rerouted storage.
		gradTs[3] = mustTensor(t, []int{2, 2}, []float32{9, 10, 11, 12})

		expectPanic(t, "backing-slice mismatch at index 3", func() {
			verifyGradTsAliasing(grads, gradTs)
		})
	})

	t.Run("LenMismatchPanics", func(t *testing.T) {
		pw := mustTensor(t, []int{2, 2}, []float32{1, 2, 3, 4})
		pb := mustTensor(t, []int{2}, []float32{5, 6})
		pe := mustTensor(t, []int{1, 2}, []float32{7, 8})
		hw := mustTensor(t, []int{2, 2}, []float32{9, 10, 11, 12})
		hb := mustTensor(t, []int{2}, []float32{13, 14})
		grads := makeGrads(pw, pb, pe, hw, hb)

		// Drop the final entry so the outer slice lengths disagree.
		gradTs := grads.allParamTensors()[:4]

		expectPanic(t, "len(grads.allParamTensors())=5 != len(gradTs)=4", func() {
			verifyGradTsAliasing(grads, gradTs)
		})
	})

	t.Run("ZeroLengthAsymmetryPanics", func(t *testing.T) {
		pw := mustTensor(t, []int{2, 2}, []float32{1, 2, 3, 4})
		pb := mustTensor(t, []int{2}, []float32{5, 6})
		pe := mustTensor(t, []int{1, 2}, []float32{7, 8})
		hw := mustTensor(t, []int{2, 2}, []float32{9, 10, 11, 12})
		hb := mustTensor(t, []int{2}, []float32{13, 14})
		grads := makeGrads(pw, pb, pe, hw, hb)

		gradTs := grads.allParamTensors()
		// Swap index 1 (patchEmbB) for a zero-length tensor while leaving the
		// live gpuGrads entry non-empty. The per-index Data()-length check
		// must fire rather than silently skipping via the zero-length branch.
		gradTs[1] = mustTensor(t, []int{0}, []float32{})

		expectPanic(t, "len mismatch at index 1", func() {
			verifyGradTsAliasing(grads, gradTs)
		})
	})
}
