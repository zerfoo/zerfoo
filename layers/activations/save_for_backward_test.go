package activations

import (
	"context"
	"math"
	"testing"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/ztensor/types"
)

// recordingSaver is a stub graph.Saver that records every tensor a node
// registers via SaveForBackward. It stands in for the graph-provided,
// arena-pinning Saver (ztensor ADR 006) in CPU-only tests.
type recordingSaver[T tensor.Numeric] struct {
	saved []*tensor.TensorNumeric[T]
}

func (r *recordingSaver[T]) SaveForBackward(ts ...*tensor.TensorNumeric[T]) {
	r.saved = append(r.saved, ts...)
}

var _ graph.Saver[float32] = (*recordingSaver[float32])(nil)

// TestSoftmax_SaveForBackward_RegistersOutput asserts the SAVE migration:
// Softmax registers its cached forward output (expensive to recompute) with
// the save-for-backward contract, so arena-backed storage stays pinned
// until Backward consumes it (zerfoo#842 bug class).
func TestSoftmax_SaveForBackward_RegistersOutput(t *testing.T) {
	ctx := context.Background()
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	sm := NewSoftmax[float32](engine, -1)

	saver := &recordingSaver[float32]{}
	sm.SetSaver(saver)

	input, err := tensor.New[float32]([]int{2, 3}, []float32{1, 2, 3, 0.5, -1, 2})
	if err != nil {
		t.Fatalf("create input: %v", err)
	}

	out, err := sm.Forward(ctx, input)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}

	if len(saver.saved) != 1 {
		t.Fatalf("expected 1 saved tensor, got %d", len(saver.saved))
	}
	if saver.saved[0] != out {
		t.Fatalf("saved tensor is not the forward output: saved %p, output %p", saver.saved[0], out)
	}

	// Backward must still work and consume the registered tensor.
	dOut, err := tensor.New[float32]([]int{2, 3}, []float32{1, 1, 1, 1, 1, 1})
	if err != nil {
		t.Fatalf("create dOut: %v", err)
	}
	grads, err := sm.Backward(ctx, types.FullBackprop, dOut, input)
	if err != nil {
		t.Fatalf("Backward: %v", err)
	}
	if len(grads) != 1 {
		t.Fatalf("expected 1 gradient, got %d", len(grads))
	}
}

// TestSwiGLU_SaveForBackward_RegistersGateAndSilu asserts SwiGLU registers
// both backward-consumed intermediates (gate and silu(x1)).
func TestSwiGLU_SaveForBackward_RegistersGateAndSilu(t *testing.T) {
	ctx := context.Background()
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	sw := NewSwiGLU[float32](engine, engine.Ops())

	saver := &recordingSaver[float32]{}
	sw.SetSaver(saver)

	input, err := tensor.New[float32]([]int{1, 4}, []float32{0.5, -1, 2, 0.25})
	if err != nil {
		t.Fatalf("create input: %v", err)
	}
	if _, err := sw.Forward(ctx, input); err != nil {
		t.Fatalf("Forward: %v", err)
	}

	// The composed Sigmoid saves its output (the gate) and SwiGLU saves
	// gate + siluX1; at minimum both backward-consumed intermediates must
	// be registered.
	foundGate, foundSilu := false, false
	for _, ts := range saver.saved {
		if ts == sw.gate {
			foundGate = true
		}
		if ts == sw.siluX1 {
			foundSilu = true
		}
	}
	if !foundGate || !foundSilu {
		t.Fatalf("expected gate and siluX1 registered; gate=%v silu=%v (saved %d tensors)",
			foundGate, foundSilu, len(saver.saved))
	}
}

// TestGelu_Backward_ReadsLiveInputs asserts the RECOMPUTE migration: Gelu's
// Backward derives the gradient from the live `inputs ...` the graph passes
// in, not from a stale forward-time cache. The input buffer is mutated
// in place between Forward and Backward (simulating arena reuse); the
// gradient must match a fresh layer that only ever saw the new values.
func TestGelu_Backward_ReadsLiveInputs(t *testing.T) {
	ctx := context.Background()
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	ops := engine.Ops()

	input, err := tensor.New[float32]([]int{1, 3}, []float32{0.1, -0.2, 0.3})
	if err != nil {
		t.Fatalf("create input: %v", err)
	}

	gelu := NewGelu(engine, ops)
	if _, err := gelu.Forward(ctx, input); err != nil {
		t.Fatalf("Forward: %v", err)
	}

	// Overwrite the input buffer in place: this is exactly what the GPU
	// arena does to cached forward tensors before Backward runs.
	perturbed := []float32{1.5, -0.7, 2.0}
	copy(input.Data(), perturbed)

	grad, err := tensor.New[float32]([]int{1, 3}, []float32{1, 1, 1})
	if err != nil {
		t.Fatalf("create grad: %v", err)
	}
	got, err := gelu.Backward(ctx, types.FullBackprop, grad, input)
	if err != nil {
		t.Fatalf("Backward: %v", err)
	}

	// Reference: a fresh layer that ran Forward on the perturbed values.
	ref := NewGelu(engine, ops)
	refInput, err := tensor.New[float32]([]int{1, 3}, perturbed)
	if err != nil {
		t.Fatalf("create ref input: %v", err)
	}
	if _, err := ref.Forward(ctx, refInput); err != nil {
		t.Fatalf("ref Forward: %v", err)
	}
	want, err := ref.Backward(ctx, types.FullBackprop, grad, refInput)
	if err != nil {
		t.Fatalf("ref Backward: %v", err)
	}

	for i := range got[0].Data() {
		g, w := got[0].Data()[i], want[0].Data()[i]
		if math.Abs(float64(g-w)) > 1e-6 {
			t.Errorf("grad[%d] = %v, want %v (backward read a stale cache?)", i, g, w)
		}
	}
}
