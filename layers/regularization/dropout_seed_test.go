package regularization

import (
	"context"
	"math/rand/v2"
	"testing"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/ztensor/types"
)

// newSeededDropout builds a training-mode Dropout seeded with the given value.
func newSeededDropout(rate float32, seed uint64) *Dropout[float32] {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine(ops)
	d := NewDropout(engine, ops, rate, WithDropoutSeed[float32](seed))
	d.SetTraining(true)

	return d
}

// runDropoutForward runs a single training-mode Forward over a constant-ones
// input and returns the resulting mask (recovered from the cached mask tensor).
func runDropoutForward(t *testing.T, d *Dropout[float32], size int) []float32 {
	t.Helper()
	ctx := context.Background()

	inputData := make([]float32, size)
	for i := range inputData {
		inputData[i] = 1.0
	}
	input, err := tensor.New([]int{size}, inputData)
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

func maskEqual(a, b []float32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}

	return true
}

// TestDropout_SeededReproducible verifies the core reproducibility contract:
// two layers seeded identically produce byte-identical masks for the same
// input and call sequence.
func TestDropout_SeededReproducible(t *testing.T) {
	const (
		size = 256
		seed = uint64(0xC0FFEE)
	)

	maskA := runDropoutForward(t, newSeededDropout(0.5, seed), size)
	maskB := runDropoutForward(t, newSeededDropout(0.5, seed), size)

	if !maskEqual(maskA, maskB) {
		t.Fatal("same seed must produce identical dropout masks")
	}

	// Sanity: the mask is a real dropout mask (mix of zeros and scaled survivors).
	zeros := 0
	for _, v := range maskA {
		if v == 0 {
			zeros++
		}
	}
	if zeros == 0 || zeros == size {
		t.Fatalf("seeded mask is degenerate: %d zeros of %d", zeros, size)
	}
}

// TestDropout_SeededDiffersBySeed verifies that different seeds yield different
// masks (so the seed actually drives the draw).
func TestDropout_SeededDiffersBySeed(t *testing.T) {
	const size = 256

	maskA := runDropoutForward(t, newSeededDropout(0.5, 1), size)
	maskB := runDropoutForward(t, newSeededDropout(0.5, 2), size)

	if maskEqual(maskA, maskB) {
		t.Fatal("different seeds should (with overwhelming probability) produce different masks")
	}
}

// TestDropout_SeededSequenceAdvances verifies that successive Forward calls on
// the same seeded layer advance the source (masks differ batch-to-batch) and
// that the full sequence is reproducible across two identically seeded layers.
func TestDropout_SeededSequenceAdvances(t *testing.T) {
	const (
		size = 128
		seed = uint64(7)
	)

	d1 := newSeededDropout(0.5, seed)
	d2 := newSeededDropout(0.5, seed)

	var seq1, seq2 [][]float32
	for i := 0; i < 4; i++ {
		seq1 = append(seq1, runDropoutForward(t, d1, size))
		seq2 = append(seq2, runDropoutForward(t, d2, size))
	}

	// Each step of the two identically seeded layers must match.
	for i := range seq1 {
		if !maskEqual(seq1[i], seq2[i]) {
			t.Fatalf("step %d: identically seeded layers diverged", i)
		}
	}

	// Consecutive steps should differ (the source advances; not the same mask).
	if maskEqual(seq1[0], seq1[1]) {
		t.Fatal("consecutive seeded Forward calls produced an identical mask; source did not advance")
	}
}

// TestDropout_WithSourceReproducible verifies the WithDropoutSource option is
// honored and reproducible when fed an identically seeded rand.Source.
func TestDropout_WithSourceReproducible(t *testing.T) {
	const size = 200

	build := func() *Dropout[float32] {
		ops := numeric.Float32Ops{}
		engine := compute.NewCPUEngine(ops)
		src := rand.NewPCG(11, 22)
		d := NewDropout(engine, ops, float32(0.4), WithDropoutSource[float32](src))
		d.SetTraining(true)

		return d
	}

	maskA := runDropoutForward(t, build(), size)
	maskB := runDropoutForward(t, build(), size)

	if !maskEqual(maskA, maskB) {
		t.Fatal("identically seeded rand.Source must produce identical masks")
	}
}

// TestDropout_WithSourceNilIgnored verifies a nil source leaves the default
// (unseeded) behavior in place rather than panicking.
func TestDropout_WithSourceNilIgnored(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine(ops)
	d := NewDropout(engine, ops, float32(0.5), WithDropoutSource[float32](nil))
	if d.rng != nil {
		t.Fatal("nil source should leave rng unset (default unseeded behavior)")
	}
	d.SetTraining(true)
	// Should still run without panic.
	_ = runDropoutForward(t, d, 64)
}

// TestDropout_DefaultUnseeded verifies the default constructor preserves the
// historical unseeded behavior: no seeded source is installed.
func TestDropout_DefaultUnseeded(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine(ops)
	d := NewDropout(engine, ops, float32(0.5))
	if d.rng != nil {
		t.Fatal("default NewDropout must not install a seeded source (backward compatible)")
	}
}

// TestDropout_SeededEvalModeUnaffected verifies the seed has no effect in eval
// mode: the input is returned unchanged (the eval-mode parity contract holds
// regardless of seeding).
func TestDropout_SeededEvalModeUnaffected(t *testing.T) {
	ctx := context.Background()
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine(ops)
	d := NewDropout(engine, ops, float32(0.5), WithDropoutSeed[float32](99))
	// Default mode is eval (training=false); do not enable training.

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
	for i, v := range output.Data() {
		if v != inputData[i] {
			t.Errorf("element %d: got %v, want %v", i, v, inputData[i])
		}
	}
}

// TestDropout_SeededBackwardConsistent verifies the seeded forward mask is the
// same one applied in Backward (no gradient w.r.t. the mask; the gate here is
// forward/backward determinism, not gradcheck of the random mask).
func TestDropout_SeededBackwardConsistent(t *testing.T) {
	ctx := context.Background()
	d := newSeededDropout(0.5, 5)

	size := 64
	mask := runDropoutForward(t, d, size)

	dOutData := make([]float32, size)
	for i := range dOutData {
		dOutData[i] = 1.0
	}
	dOut, err := tensor.New([]int{size}, dOutData)
	if err != nil {
		t.Fatalf("failed to create dOut tensor: %v", err)
	}

	input, _ := tensor.New([]int{size}, make([]float32, size))
	grads, err := d.Backward(ctx, types.FullBackprop, dOut, input)
	if err != nil {
		t.Fatalf("Backward returned error: %v", err)
	}

	for i, g := range grads[0].Data() {
		if g != mask[i] {
			t.Errorf("element %d: backward grad %v != forward mask %v", i, g, mask[i])
		}
	}
}
