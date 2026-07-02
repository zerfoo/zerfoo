package regularization

import (
	"context"
	"testing"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
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

// TestDropout_SaveForBackward_RegistersMask asserts the SAVE migration:
// the dropout mask is random and therefore cannot be recomputed in
// Backward; it must be registered with the save-for-backward contract so
// arena-backed storage stays pinned until Backward consumes it
// (zerfoo#842 / Wolf QK-norm bug class).
func TestDropout_SaveForBackward_RegistersMask(t *testing.T) {
	ctx := context.Background()
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	d := NewDropout[float32](engine, engine.Ops(), 0.5)
	d.SetTraining(true)

	saver := &recordingSaver[float32]{}
	d.SetSaver(saver)

	input, err := tensor.New[float32]([]int{4, 8}, nil)
	if err != nil {
		t.Fatalf("create input: %v", err)
	}
	for i := range input.Data() {
		input.Data()[i] = 1.0
	}

	if _, err := d.Forward(ctx, input); err != nil {
		t.Fatalf("Forward: %v", err)
	}

	if len(saver.saved) != 1 {
		t.Fatalf("expected 1 saved tensor (the mask), got %d", len(saver.saved))
	}
	if saver.saved[0] != d.mask {
		t.Fatalf("saved tensor is not the dropout mask")
	}

	// Eval mode must not register anything (mask is nil, Forward is identity).
	saver.saved = nil
	d.SetTraining(false)
	if _, err := d.Forward(ctx, input); err != nil {
		t.Fatalf("Forward (eval): %v", err)
	}
	if len(saver.saved) != 0 {
		t.Fatalf("eval-mode Forward must not save tensors, saved %d", len(saver.saved))
	}
}

// TestFeatureDropout_SaveForBackward_RegistersMask is the FeatureDropout
// counterpart of TestDropout_SaveForBackward_RegistersMask.
func TestFeatureDropout_SaveForBackward_RegistersMask(t *testing.T) {
	ctx := context.Background()
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	d := NewFeatureDropout[float32](engine, engine.Ops(), 0.5)
	d.SetTraining(true)

	saver := &recordingSaver[float32]{}
	d.SetSaver(saver)

	input, err := tensor.New[float32]([]int{4, 8}, nil)
	if err != nil {
		t.Fatalf("create input: %v", err)
	}
	for i := range input.Data() {
		input.Data()[i] = 1.0
	}

	if _, err := d.Forward(ctx, input); err != nil {
		t.Fatalf("Forward: %v", err)
	}

	if len(saver.saved) != 1 {
		t.Fatalf("expected 1 saved tensor (the mask), got %d", len(saver.saved))
	}
	if saver.saved[0] != d.mask {
		t.Fatalf("saved tensor is not the feature-dropout mask")
	}
}
