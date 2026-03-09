package codegen

import (
	"testing"

	"github.com/zerfoo/zerfoo/internal/cuda"
)

func TestLoadMegakernelBadPath(t *testing.T) {
	if !cuda.Available() {
		t.Skip("CUDA not available")
	}
	_, err := LoadMegakernel("/nonexistent/path/megakernel.so")
	if err == nil {
		t.Fatal("expected error for nonexistent .so path")
	}
}

func TestPrepareWorkspaceLayout(t *testing.T) {
	if !cuda.Available() {
		t.Skip("CUDA not available")
	}

	cfg := MegakernelConfig{
		SlotShapes: [][]int{
			{4},    // slot 0: input (4 elements)
			{4},    // slot 1: intermediate
			{2},    // slot 2: output
			{4},    // slot 3: frozen weight
		},
		FrozenSlots: []FrozenSlotMeta{{SlotIdx: 3}},
		InputSlots:  []int{0},
		OutputSlot:  2,
	}

	r := &MegakernelRunner{}
	frozenData := [][]float32{{1.0, 2.0, 3.0, 4.0}}

	if err := r.PrepareWorkspace(cfg, frozenData); err != nil {
		t.Fatalf("PrepareWorkspace: %v", err)
	}
	defer func() { _ = r.Close() }()

	// Verify layout was computed correctly.
	if r.layout.TotalSize == 0 {
		t.Error("expected non-zero workspace size")
	}
	if r.layout.OutputSize != 2 {
		t.Errorf("expected output size 2, got %d", r.layout.OutputSize)
	}
	if r.workspace == nil {
		t.Error("expected non-nil workspace pointer")
	}
	if r.frozenPtrs == nil {
		t.Error("expected non-nil frozenPtrs pointer")
	}
	if len(r.frozenBufs) != 1 {
		t.Errorf("expected 1 frozen buf, got %d", len(r.frozenBufs))
	}
}

func TestPrepareWorkspaceFrozenMismatch(t *testing.T) {
	if !cuda.Available() {
		t.Skip("CUDA not available")
	}

	cfg := MegakernelConfig{
		SlotShapes:  [][]int{{4}, {4}},
		FrozenSlots: []FrozenSlotMeta{{SlotIdx: 1}},
		InputSlots:  []int{0},
		OutputSlot:  0,
	}

	r := &MegakernelRunner{}
	// Wrong number of frozen data slices.
	err := r.PrepareWorkspace(cfg, [][]float32{{1.0}, {2.0}})
	if err == nil {
		t.Fatal("expected error for mismatched frozenData length")
	}
}
