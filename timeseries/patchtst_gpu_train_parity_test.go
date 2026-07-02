package timeseries

import (
	"math"
	"math/rand"
	"testing"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
)

// TestGPUSingleStepParity is a single-step diagnostic regression test for the
// GPU training path. It runs exactly one forward + backward + AdamW step on
// the GPU code path and asserts that the trainable parameters mutate by at
// least a small epsilon. If weights come back byte-identical (||post - pre||_inf
// == 0), the optimizer is writing into a throwaway slice instead of live
// device memory — this is the exact regression documented in
// docs/plans/gpu-training-convergence-regression.md (Epic E4 T4.1).
//
// On CPU-only machines / CI, NewGPUEngine fails and the test skips cleanly.
// On a pre-fix GPU checkout this test is designed to FAIL with the message
// "weights static after optimizer step"; on a fixed GPU checkout it passes.
// Do not convert this to a build-tagged file — the runtime skip is intentional
// so the test is discoverable via `go test -run TestGPUSingleStepParity ./...`.
func TestGPUSingleStepParity(t *testing.T) {
	ops := numeric.Float32Ops{}
	gpu, err := compute.NewGPUEngine[float32](ops, 0)
	if err != nil {
		t.Skipf("GPU not available: %v", err)
	}
	_ = gpu

	// Deterministic RNG so both the model init and the inputs are reproducible.
	rand.Seed(1)

	// Tiny PatchTST config: 2 samples * 1 channel * small embedding.
	cfg := PatchTSTConfig{
		InputLength: 4,
		PatchLength: 2,
		Stride:      2,
		DModel:      4,
		NHeads:      2,
		NLayers:     1,
		OutputDim:   1,
	}

	m, err := NewPatchTST(cfg, gpu, ops)
	if err != nil {
		t.Fatalf("NewPatchTST: %v", err)
	}

	// Exactly 2 samples * 1 channel * InputLength. Deterministic values.
	windows := [][][]float64{
		{{0.1, 0.2, 0.3, 0.4}},
		{{0.5, 0.6, 0.7, 0.8}},
	}
	// labels: nSamples * OutputDim = 2 * 1 = 2.
	labels := []float64{1.0, -1.0}

	// Snapshot BEFORE the optimizer step.
	// patchEmb.weights is a float32 tensor; on GPU Data() returns a fresh
	// device->host copy, so we copy into an independent slice.
	pre := make([]float32, len(m.patchEmb.weights.Data()))
	copy(pre, m.patchEmb.weights.Data())

	// ONE step: Epochs=1, BatchSize=nSamples=2 -> one optimizer update.
	tc := TrainConfig{
		Epochs:      1,
		LR:          1e-2,
		BatchSize:   2,
		Beta1:       0.9,
		Beta2:       0.999,
		Epsilon:     1e-8,
		WeightDecay: 0.0,
	}
	if _, err := m.TrainWindowed(windows, labels, tc); err != nil {
		t.Fatalf("TrainWindowed (GPU): %v", err)
	}

	// Snapshot AFTER the optimizer step.
	post := make([]float32, len(m.patchEmb.weights.Data()))
	copy(post, m.patchEmb.weights.Data())

	if len(pre) != len(post) {
		t.Fatalf("weight slice length changed: pre=%d post=%d", len(pre), len(post))
	}

	// Inf-norm of (post - pre).
	var maxDelta float32
	for i := range pre {
		d := float32(math.Abs(float64(post[i] - pre[i])))
		if d > maxDelta {
			maxDelta = d
		}
	}

	const eps float32 = 1e-7
	if maxDelta <= eps {
		t.Fatalf("weights static after optimizer step: ||post - pre||_inf = %g (<= %g); "+
			"GPU AdamW wrote to a throwaway buffer instead of live device memory "+
			"(see docs/plans/gpu-training-convergence-regression.md, T4.1)",
			maxDelta, eps)
	}
	t.Logf("patchEmb weight delta (inf-norm) after 1 step: %g", maxDelta)
}
