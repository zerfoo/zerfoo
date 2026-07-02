package nas

import (
	"testing"
)

func TestDiscretizeArgmax(t *testing.T) {
	// 3 nodes → 3 edges, 3 ops → alpha has 9 elements.
	ops := []OpType{OpSkipConnect, OpConv3x3, OpZero}
	space := NewSearchSpaceWithOps(3, ops)

	// Edge 0 (0→1): op1 (conv3x3) has highest alpha.
	// Edge 1 (0→2): op0 (skip) has highest alpha.
	// Edge 2 (1→2): op2 (zero) has highest alpha.
	alpha := []float32{
		0.1, 0.9, 0.0, // edge 0 → conv3x3
		0.8, 0.1, 0.1, // edge 1 → skip
		0.0, 0.2, 0.8, // edge 2 → zero
	}

	arch, err := Discretize(alpha, space, 0)
	if err != nil {
		t.Fatalf("Discretize: %v", err)
	}

	if !arch.Cell.Valid() {
		t.Fatal("discretized cell is not valid")
	}

	wantOps := []OpType{OpConv3x3, OpSkipConnect, OpZero}
	if len(arch.Cell.Edges) != len(wantOps) {
		t.Fatalf("got %d edges, want %d", len(arch.Cell.Edges), len(wantOps))
	}
	for i, e := range arch.Cell.Edges {
		if e.Op != wantOps[i] {
			t.Errorf("edge %d: got op %s, want %s", i, e.Op, wantOps[i])
		}
	}
}

func TestDiscretizeMaxParamsExceeded(t *testing.T) {
	ops := []OpType{OpConv3x3, OpZero}
	space := NewSearchSpaceWithOps(3, ops)

	// All edges select conv3x3 (index 0 has highest alpha).
	alpha := []float32{
		0.9, 0.1,
		0.9, 0.1,
		0.9, 0.1,
	}

	// conv3x3 params = 9 * 64 * 64 = 36864, 3 edges → 110592 total.
	// Set maxParams to something smaller.
	_, err := Discretize(alpha, space, 1000)
	if err == nil {
		t.Fatal("expected error for exceeding max_params, got nil")
	}
}

func TestDiscretizeMaxParamsOK(t *testing.T) {
	ops := []OpType{OpConv3x3, OpZero}
	space := NewSearchSpaceWithOps(3, ops)

	// All edges select zero (index 1 has highest alpha) → 0 params.
	alpha := []float32{
		0.1, 0.9,
		0.1, 0.9,
		0.1, 0.9,
	}

	arch, err := Discretize(alpha, space, 1000)
	if err != nil {
		t.Fatalf("Discretize: %v", err)
	}
	if arch.TotalParams != 0 {
		t.Errorf("got TotalParams=%d, want 0", arch.TotalParams)
	}
}

func TestDiscretizeNilSpace(t *testing.T) {
	_, err := Discretize([]float32{1.0}, nil, 0)
	if err == nil {
		t.Fatal("expected error for nil space")
	}
}

func TestDiscretizeAlphaLengthMismatch(t *testing.T) {
	space := NewSearchSpace(3)
	_, err := Discretize([]float32{1.0, 2.0}, space, 0)
	if err == nil {
		t.Fatal("expected error for alpha length mismatch")
	}
}

func TestDiscretizeNoConstraint(t *testing.T) {
	// maxParams=0 means no constraint.
	ops := []OpType{OpConv5x5, OpConv3x3}
	space := NewSearchSpaceWithOps(3, ops)

	alpha := []float32{
		0.9, 0.1, // edge 0 → conv5x5
		0.9, 0.1, // edge 1 → conv5x5
		0.9, 0.1, // edge 2 → conv5x5
	}

	arch, err := Discretize(alpha, space, 0)
	if err != nil {
		t.Fatalf("Discretize: %v", err)
	}

	// conv5x5 params = 25 * 64 * 64 = 102400, 3 edges → 307200.
	wantParams := int64(3 * 25 * 64 * 64)
	if arch.TotalParams != wantParams {
		t.Errorf("got TotalParams=%d, want %d", arch.TotalParams, wantParams)
	}
}

func TestDiscretizeTieBreaking(t *testing.T) {
	// When multiple ops have equal alpha, the first one wins (argmax from left).
	ops := []OpType{OpSkipConnect, OpZero}
	space := NewSearchSpaceWithOps(3, ops)

	alpha := []float32{
		0.5, 0.5, // tie → skip_connect (index 0)
		0.5, 0.5,
		0.5, 0.5,
	}

	arch, err := Discretize(alpha, space, 0)
	if err != nil {
		t.Fatalf("Discretize: %v", err)
	}

	for i, e := range arch.Cell.Edges {
		if e.Op != OpSkipConnect {
			t.Errorf("edge %d: got op %s, want %s on tie", i, e.Op, OpSkipConnect)
		}
	}
}
