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

// TestGradTsSentinel exercises verifyGradTsAliasing across the regression
// classes the v3 Storage-identity sentinel is meant to catch.
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
		// shares Storage identity by construction.
		gradTs := grads.allParamTensors()

		// Must not panic.
		verifyGradTsAliasing(grads, gradTs)
	})

	t.Run("WrapperMismatchPanics", func(t *testing.T) {
		pw := mustTensor(t, []int{2, 2}, []float32{1, 2, 3, 4})
		pb := mustTensor(t, []int{2}, []float32{5, 6})
		pe := mustTensor(t, []int{1, 2}, []float32{7, 8})
		hw := mustTensor(t, []int{2, 2}, []float32{9, 10, 11, 12})
		hb := mustTensor(t, []int{2}, []float32{13, 14})
		grads := makeGrads(pw, pb, pe, hw, hb)

		gradTs := grads.allParamTensors()
		// Replace index 3 (headW) with a different tensor wrapper with
		// identical shape but an independent backing Storage — simulates the
		// class of bug where the cached gradTs slice goes stale (e.g. the
		// pre-T1.2 arena realloc scenario).
		gradTs[3] = mustTensor(t, []int{2, 2}, []float32{9, 10, 11, 12})

		expectPanic(t, "wrapper mismatch at index 3", func() {
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

	// HappyPath_SharedStorage verifies the v3 fix's core invariant: two
	// distinct *TensorNumeric wrappers that share the SAME backing Storage
	// would fail the wrapper-identity pre-filter (and so panic) — but the
	// point of the v3 fix is that identical wrappers with an EPHEMERAL
	// Data() slice (as GPUStorage.Slice() produces via D2H copy) must NOT
	// panic. We can't allocate GPUStorage without CUDA, but we can assert
	// the analogous property on the CPU path: the happy path (same wrapper,
	// same Storage) never panics even though two independent .Data() calls
	// would return slice headers with different addresses if the storage
	// were non-contiguous — CPUStorage returns the same backing array, so
	// this test is primarily a guard against future regressions of the
	// Data()-pointer comparison.
	t.Run("HappyPath_RepeatedDataCallsOK", func(t *testing.T) {
		pw := mustTensor(t, []int{2, 2}, []float32{1, 2, 3, 4})
		pb := mustTensor(t, []int{2}, []float32{5, 6})
		pe := mustTensor(t, []int{1, 2}, []float32{7, 8})
		hw := mustTensor(t, []int{2, 2}, []float32{9, 10, 11, 12})
		hb := mustTensor(t, []int{2}, []float32{13, 14})

		grads := makeGrads(pw, pb, pe, hw, hb)
		gradTs := grads.allParamTensors()

		// Touch Data() a few times on both sides to mimic the GPU path's
		// fresh-slice-per-call behavior. Must still not panic because
		// Storage identity is preserved.
		_ = pw.Data()
		_ = gradTs[0].Data()
		_ = hw.Data()
		_ = gradTs[3].Data()

		verifyGradTsAliasing(grads, gradTs)
	})

	// StorageMismatch_ConstructedViaPublicAPI: not achievable through the
	// public tensor API, since there is no supported way to swap a tensor's
	// backing Storage without also replacing the wrapper. On GPU the flip
	// happens inside ztensor (makeGPUResult.SetStorage) and requires real
	// CUDA to reproduce — covered by the T5.3 Wave 5 DGX regression run.
	// The WrapperMismatchPanics case above covers the common stale-cache
	// class; the Storage-identity check guards the (CUDA-only) flip class.
}
