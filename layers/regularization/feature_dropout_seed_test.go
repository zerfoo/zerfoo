package regularization

import (
	"context"
	"math/rand/v2"
	"testing"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
)

// runFeatureDropoutForward runs a single training-mode Forward over a constant
// input of shape [rows, cols] and returns the cached broadcast mask.
func runFeatureDropoutForward(t *testing.T, d *FeatureDropout[float32], rows, cols int) []float32 {
	t.Helper()
	ctx := context.Background()

	size := rows * cols
	inputData := make([]float32, size)
	for i := range inputData {
		inputData[i] = 1.0
	}
	input, err := tensor.New([]int{rows, cols}, inputData)
	if err != nil {
		t.Fatalf("failed to create input tensor: %v", err)
	}

	if _, err := d.Forward(ctx, input); err != nil {
		t.Fatalf("Forward returned error: %v", err)
	}
	if d.mask == nil {
		t.Fatal("expected a cached mask after training-mode Forward")
	}

	out := make([]float32, size)
	copy(out, d.mask.Data())

	return out
}

func newSeededFeatureDropout(rate float32, seed uint64) *FeatureDropout[float32] {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine(ops)
	d := NewFeatureDropout(engine, ops, rate, WithFeatureDropoutSeed[float32](seed))
	d.SetTraining(true)

	return d
}

// TestFeatureDropout_SeededReproducible verifies same seed -> identical masks.
func TestFeatureDropout_SeededReproducible(t *testing.T) {
	const (
		rows = 4
		cols = 64
		seed = uint64(0xBEEF)
	)

	maskA := runFeatureDropoutForward(t, newSeededFeatureDropout(0.5, seed), rows, cols)
	maskB := runFeatureDropoutForward(t, newSeededFeatureDropout(0.5, seed), rows, cols)

	if !maskEqual(maskA, maskB) {
		t.Fatal("same seed must produce identical feature-dropout masks")
	}

	// Sanity: at least one column dropped and one retained.
	zeros := 0
	for _, v := range maskA {
		if v == 0 {
			zeros++
		}
	}
	if zeros == 0 || zeros == len(maskA) {
		t.Fatalf("seeded feature mask is degenerate: %d zeros of %d", zeros, len(maskA))
	}
}

// TestFeatureDropout_SeededDiffersBySeed verifies different seeds -> different masks.
func TestFeatureDropout_SeededDiffersBySeed(t *testing.T) {
	const (
		rows = 4
		cols = 64
	)

	maskA := runFeatureDropoutForward(t, newSeededFeatureDropout(0.5, 1), rows, cols)
	maskB := runFeatureDropoutForward(t, newSeededFeatureDropout(0.5, 2), rows, cols)

	if maskEqual(maskA, maskB) {
		t.Fatal("different seeds should produce different feature-dropout masks")
	}
}

// TestFeatureDropout_WithSourceReproducible verifies WithFeatureDropoutSource.
func TestFeatureDropout_WithSourceReproducible(t *testing.T) {
	const (
		rows = 3
		cols = 50
	)

	build := func() *FeatureDropout[float32] {
		ops := numeric.Float32Ops{}
		engine := compute.NewCPUEngine(ops)
		src := rand.NewPCG(101, 202)
		d := NewFeatureDropout(engine, ops, float32(0.4), WithFeatureDropoutSource[float32](src))
		d.SetTraining(true)

		return d
	}

	maskA := runFeatureDropoutForward(t, build(), rows, cols)
	maskB := runFeatureDropoutForward(t, build(), rows, cols)

	if !maskEqual(maskA, maskB) {
		t.Fatal("identically seeded rand.Source must produce identical feature masks")
	}
}

// TestFeatureDropout_DefaultUnseeded verifies the default constructor is unseeded.
func TestFeatureDropout_DefaultUnseeded(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine(ops)
	d := NewFeatureDropout(engine, ops, float32(0.5))
	if d.rng != nil {
		t.Fatal("default NewFeatureDropout must not install a seeded source (backward compatible)")
	}
}

// TestFeatureDropout_SeededEvalModeUnaffected verifies eval mode passes input
// through unchanged regardless of seeding.
func TestFeatureDropout_SeededEvalModeUnaffected(t *testing.T) {
	ctx := context.Background()
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine(ops)
	d := NewFeatureDropout(engine, ops, float32(0.5), WithFeatureDropoutSeed[float32](42))
	// Eval mode (training=false) by default.

	inputData := []float32{1, 2, 3, 4, 5, 6}
	input, err := tensor.New([]int{2, 3}, inputData)
	if err != nil {
		t.Fatalf("failed to create input tensor: %v", err)
	}

	output, err := d.Forward(ctx, input)
	if err != nil {
		t.Fatalf("Forward returned error: %v", err)
	}
	if output != input {
		t.Error("eval mode must return the same pointer as input regardless of seeding")
	}
}

// TestFeatureDropout_WithSourceNilIgnored verifies nil source is a no-op.
func TestFeatureDropout_WithSourceNilIgnored(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine(ops)
	d := NewFeatureDropout(engine, ops, float32(0.5), WithFeatureDropoutSource[float32](nil))
	if d.rng != nil {
		t.Fatal("nil source should leave rng unset (default unseeded behavior)")
	}
}
