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

// TestLayerNorm_MixedBackward_RecomputesStatsFromInput is the regression test
// for the residual GPU f32 "CrossAsset cliff" (Wolf docs/plan-gpu-f32-residual,
// S-GR.4.1). The mixed-precision (f64) backward must recompute mean/variance
// from the forward input, NOT trust the cached ln.mean/ln.variance tensors:
// on the GPU arena those caches get overwritten by downstream forward ops
// before Backward runs, so the cached variance comes back negative and
// sigma = sqrt(var+eps) becomes NaN.
//
// We simulate that corruption by clobbering the cached variance with garbage
// (including a negative value) after Forward, then assert the backward still
// produces finite gradients that match an uncorrupted reference.
func TestLayerNorm_MixedBackward_RecomputesStatsFromInput(t *testing.T) {
	engine := compute.NewCPUEngine[float32](&numeric.Float32Ops{})
	ctx := context.Background()
	const nFeat = 4

	ln, err := NewLayerNormalization[float32](engine, nFeat)
	if err != nil {
		t.Fatalf("NewLayerNormalization: %v", err)
	}
	if !ln.useMixedBackward {
		t.Fatal("expected useMixedBackward=true for float32 on CPU engine")
	}

	inputData := []float32{0.5, -1.2, 3.4, 40.0, 2.0, 2.0, 2.0, 2.1}
	input, err := tensor.New[float32]([]int{2, nFeat}, inputData)
	if err != nil {
		t.Fatalf("tensor.New input: %v", err)
	}
	dOutData := []float32{0.1, -0.2, 0.3, -0.4, 0.05, 0.06, -0.07, 0.08}
	dOut, err := tensor.New[float32]([]int{2, nFeat}, dOutData)
	if err != nil {
		t.Fatalf("tensor.New dOut: %v", err)
	}

	// Reference gradients from a clean forward+backward (uncorrupted cache).
	if _, err := ln.Forward(ctx, input); err != nil {
		t.Fatalf("Forward (reference): %v", err)
	}
	refGrads, err := ln.Backward(ctx, types.FullBackprop, dOut, input)
	if err != nil {
		t.Fatalf("Backward (reference): %v", err)
	}
	refData := append([]float32(nil), refGrads[0].Data()...)
	for i, v := range refData {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("reference gradient non-finite at %d: %v", i, v)
		}
	}

	// Now corrupt the cached stats the way the GPU arena does: overwrite the
	// cached variance with garbage that includes a negative value (which would
	// make sigma = sqrt(var+eps) NaN if the backward trusted the cache).
	if _, err := ln.Forward(ctx, input); err != nil {
		t.Fatalf("Forward (corrupt case): %v", err)
	}
	varData := ln.variance.Data()
	for i := range varData {
		varData[i] = -7.5 // impossible for a real variance; sqrt(<0) = NaN
	}
	meanData := ln.mean.Data()
	for i := range meanData {
		meanData[i] = 999.0 // garbage mean
	}
	normedData := ln.normedInput.Data()
	for i := range normedData {
		normedData[i] = float32(math.NaN())
	}

	gotGrads, err := ln.Backward(ctx, types.FullBackprop, dOut, input)
	if err != nil {
		t.Fatalf("Backward (corrupt case): %v", err)
	}
	got := gotGrads[0].Data()

	// The backward must ignore the corrupted cache and recompute from input:
	// gradients stay finite and match the reference.
	const tol = 1e-5
	for i := range got {
		if math.IsNaN(float64(got[i])) || math.IsInf(float64(got[i]), 0) {
			t.Fatalf("gradient %d non-finite despite corrupted cache: %v "+
				"(backward did not recompute stats from input)", i, got[i])
		}
		if d := math.Abs(float64(got[i] - refData[i])); d > tol {
			t.Fatalf("gradient %d = %v, want %v (diff %g > tol): backward used "+
				"the corrupted cache instead of recomputing", i, got[i], refData[i], d)
		}
	}
}
