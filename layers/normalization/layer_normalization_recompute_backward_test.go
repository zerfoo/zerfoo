package normalization

import (
	"context"
	"math"
	"testing"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/ztensor/types"
)

// These are the regression tests for the GPU f32 "CrossAsset cliff"
// (zerfoo#842) and its ADR 006 (T2.3) follow-up. LayerNorm Backward must
// recompute mean/variance/normedInput from the live input the graph passes
// in, NOT from tensors cached during Forward: on the GPU arena those caches
// get overwritten by downstream forward ops before Backward runs, so the
// cached variance comes back negative and sigma = sqrt(var+eps) is NaN.
//
// Since T2.3 the layer no longer has stat caches at all, so the test
// simulates the arena hazard directly: it overwrites the INPUT buffer in
// place between Forward and Backward (exactly what arena reuse does to any
// cached tensor) and asserts the gradients match a reference computed from
// scratch on the new values. A backward reading any stale forward-time
// state would disagree.

func layerNormLiveInputCase[T tensor.Numeric](
	t *testing.T,
	engine compute.Engine[T],
	ops numeric.Arithmetic[T],
	wantMixed bool,
) {
	t.Helper()
	ctx := context.Background()
	const nFeat = 4

	ln, err := NewLayerNormalization[T](engine, nFeat)
	if err != nil {
		t.Fatalf("NewLayerNormalization: %v", err)
	}
	if ln.useMixedBackward != wantMixed {
		t.Fatalf("useMixedBackward = %v, want %v", ln.useMixedBackward, wantMixed)
	}

	mk := func(vals []float64) *tensor.TensorNumeric[T] {
		data := make([]T, len(vals))
		for i, v := range vals {
			data[i] = ops.FromFloat64(v)
		}
		tt, err := tensor.New[T]([]int{2, nFeat}, data)
		if err != nil {
			t.Fatalf("tensor.New: %v", err)
		}
		return tt
	}

	original := []float64{0.5, -1.2, 3.4, 40.0, 2.0, 2.0, 2.0, 2.1}
	perturbed := []float64{-3.0, 7.5, 0.25, -11.0, 1.0, 0.5, -0.5, 9.0}
	input := mk(original)
	dOut := mk([]float64{0.1, -0.2, 0.3, -0.4, 0.05, 0.06, -0.07, 0.08})

	if _, err := ln.Forward(ctx, input); err != nil {
		t.Fatalf("Forward: %v", err)
	}

	// Simulate arena reuse: overwrite the input buffer in place between
	// Forward and Backward.
	newData := mk(perturbed).Data()
	copy(input.Data(), newData)

	gotGrads, err := ln.Backward(ctx, types.FullBackprop, dOut, input)
	if err != nil {
		t.Fatalf("Backward: %v", err)
	}
	got := gotGrads[0].Data()

	// Reference: a fresh layer that only ever saw the perturbed values.
	ref, err := NewLayerNormalization[T](engine, nFeat)
	if err != nil {
		t.Fatalf("NewLayerNormalization (ref): %v", err)
	}
	refInput := mk(perturbed)
	if _, err := ref.Forward(ctx, refInput); err != nil {
		t.Fatalf("Forward (ref): %v", err)
	}
	refGrads, err := ref.Backward(ctx, types.FullBackprop, dOut, refInput)
	if err != nil {
		t.Fatalf("Backward (ref): %v", err)
	}
	refData := refGrads[0].Data()

	const tol = 1e-5
	for i := range got {
		g := lnNumericToFloat64(got[i])
		w := lnNumericToFloat64(refData[i])
		if math.IsNaN(g) || math.IsInf(g, 0) {
			t.Fatalf("gradient %d non-finite: %v (backward read stale forward state)", i, g)
		}
		if d := math.Abs(g - w); d > tol {
			t.Fatalf("gradient %d = %v, want %v (diff %g > tol): backward used "+
				"stale forward-time state instead of the live input", i, g, w, d)
		}
	}
}

// TestLayerNorm_MixedBackward_ReadsLiveInput covers the float32 mixed-precision
// path (zerfoo#842 recompute fix).
func TestLayerNorm_MixedBackward_ReadsLiveInput(t *testing.T) {
	engine := compute.NewCPUEngine[float32](&numeric.Float32Ops{})
	layerNormLiveInputCase[float32](t, engine, engine.Ops(), true)
}

// TestLayerNorm_GenericBackward_ReadsLiveInput covers the generic engine-op
// path (float64), migrated to live-input recompute by T2.3: before the
// migration this path still read ln.mean/ln.variance/ln.normedInput caches.
func TestLayerNorm_GenericBackward_ReadsLiveInput(t *testing.T) {
	engine := compute.NewCPUEngine[float64](&numeric.Float64Ops{})
	layerNormLiveInputCase[float64](t, engine, engine.Ops(), false)
}
