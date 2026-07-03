package gguf

import (
	"os"
	"path/filepath"
	"testing"
)

// attackDimensions is the exact tensor shape from deep-review 002, finding F1:
// two dimensions that multiply to exactly 1<<34 (passing the old strict ">"
// cap check), followed by a third dimension whose multiply overflows int64
// to a negative value that also passes "> 1<<34". Every call site must
// reject this shape with an error instead of computing a negative element
// count that later panics in allocation code.
var attackDimensions = []uint64{131072, 131072, 2147483647}

// TestComputeNumElements_AttackShape verifies the shared helper rejects the
// F1 attack shape directly, and does so BEFORE the overflow would occur
// (i.e. it must not rely on ever observing a negative product).
func TestComputeNumElements_AttackShape(t *testing.T) {
	n, err := computeNumElements("test.attack", attackDimensions)
	if err == nil {
		t.Fatalf("expected error for attack shape, got numElements=%d", n)
	}
	if n != 0 {
		t.Errorf("numElements = %d on error, want 0", n)
	}
}

// TestComputeNumElements_ExactCapBoundary verifies the check-before-multiply
// rewrite still allows a legitimate product that lands exactly on the cap
// (1<<34), which is the boundary the old strict ">" check also allowed --
// confirming the new predicate is not stricter than necessary.
func TestComputeNumElements_ExactCapBoundary(t *testing.T) {
	n, err := computeNumElements("test.exact_cap", []uint64{131072, 131072})
	if err != nil {
		t.Fatalf("computeNumElements: unexpected error at exact cap: %v", err)
	}
	if n != 1<<34 {
		t.Errorf("numElements = %d, want %d", n, int64(1<<34))
	}
}

// TestComputeNumElements_OneOverCap verifies a product one element over the
// cap is rejected.
func TestComputeNumElements_OneOverCap(t *testing.T) {
	_, err := computeNumElements("test.over_cap", []uint64{131072, 131073})
	if err == nil {
		t.Fatal("expected error for total elements > 1<<34")
	}
}

// TestComputeNumElements_ZeroDimension verifies a zero-sized dimension is
// rejected rather than silently collapsing the tensor to zero elements.
func TestComputeNumElements_ZeroDimension(t *testing.T) {
	_, err := computeNumElements("test.zero_dim", []uint64{4096, 0})
	if err == nil {
		t.Fatal("expected error for zero dimension")
	}
}

// TestComputeNumElements_DimensionExceedsMaxInt32 verifies a single dimension
// above math.MaxInt32 is rejected.
func TestComputeNumElements_DimensionExceedsMaxInt32(t *testing.T) {
	_, err := computeNumElements("test.huge_dim", []uint64{1 << 32})
	if err == nil {
		t.Fatal("expected error for dimension exceeding MaxInt32")
	}
}

// TestComputeNumElements_LegitimateShapes is a table test of realistic
// tensor shapes drawn from common model architectures. None of these should
// be affected by the check-before-multiply rewrite -- the new predicate is
// strictly tighter than the old one only at the two failure modes above, and
// every shape here stays comfortably under the cap.
func TestComputeNumElements_LegitimateShapes(t *testing.T) {
	tests := []struct {
		name       string
		dimensions []uint64
		want       int64
	}{
		{"scalar (rank 0)", []uint64{}, 1},
		{"1D bias", []uint64{4096}, 4096},
		{"small dense weight", []uint64{4096, 4096}, 4096 * 4096},
		{"attention qkv proj", []uint64{4096, 12288}, 4096 * 12288},
		{"ffn up proj", []uint64{4096, 14336}, 4096 * 14336},
		{"llama-3 vocab embedding", []uint64{128256, 4096}, 128256 * 4096},
		{"gemma-4-edge PLE embedding", []uint64{262144, 256}, 262144 * 256},
		{"3D conv-like kernel", []uint64{64, 3, 3}, 64 * 3 * 3},
		{"large single dim near MaxInt32", []uint64{1 << 20}, 1 << 20},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := computeNumElements("test."+tt.name, tt.dimensions)
			if err != nil {
				t.Fatalf("computeNumElements: unexpected error: %v", err)
			}
			if got != tt.want {
				t.Errorf("numElements = %d, want %d", got, tt.want)
			}
		})
	}
}

// TestLoadTensors_AttackShape confirms the LoadTensors (heap, single-file)
// call site returns an error -- not a panic -- for the exact F1 attack
// shape.
func TestLoadTensors_AttackShape(t *testing.T) {
	tensors := []TensorInfo{{
		Name:       "test.attack",
		Dimensions: attackDimensions,
		Type:       GGMLTypeF32,
		Offset:     0,
	}}
	r := buildGGUFWithTensors(t, tensors, nil)
	f, err := Parse(r)
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}

	defer func() {
		if rec := recover(); rec != nil {
			t.Fatalf("LoadTensors panicked instead of returning an error: %v", rec)
		}
	}()

	_, err = LoadTensors(f, r)
	if err == nil {
		t.Fatal("expected error for F1 attack shape, got nil")
	}
}

// TestLoadTensorsMmap_AttackShape confirms the LoadTensorsMmap call site
// (the default mmap load path) returns an error -- not a panic -- for the
// exact F1 attack shape.
func TestLoadTensorsMmap_AttackShape(t *testing.T) {
	gf := &File{
		DataOffset: 0,
		Tensors: []TensorInfo{{
			Name:       "test.attack",
			Dimensions: attackDimensions,
			Type:       GGMLTypeF32,
			Offset:     0,
		}},
	}
	mapped := make([]byte, 64)

	defer func() {
		if rec := recover(); rec != nil {
			t.Fatalf("LoadTensorsMmap panicked instead of returning an error: %v", rec)
		}
	}()

	_, err := LoadTensorsMmap(gf, mapped)
	if err == nil {
		t.Fatal("expected error for F1 attack shape, got nil")
	}
}

// TestLoadTensorsMmapSplit_AttackShape confirms the split-file mmap load
// site returns an error -- not a panic -- for the exact F1 attack shape.
func TestLoadTensorsMmapSplit_AttackShape(t *testing.T) {
	shard := &File{DataOffset: 0}
	sf := &SplitFile{
		File: &File{
			Tensors: []TensorInfo{{
				Name:       "test.attack",
				Dimensions: attackDimensions,
				Type:       GGMLTypeF32,
				Offset:     0,
			}},
		},
		Shards:     []*File{shard},
		ShardIndex: map[string]int{"test.attack": 0},
	}
	mappedShards := [][]byte{make([]byte, 64)}

	defer func() {
		if rec := recover(); rec != nil {
			t.Fatalf("LoadTensorsMmapSplit panicked instead of returning an error: %v", rec)
		}
	}()

	_, err := LoadTensorsMmapSplit(sf, mappedShards)
	if err == nil {
		t.Fatal("expected error for F1 attack shape, got nil")
	}
}

// TestLoadTensorsSplit_AttackShape confirms the split-file heap load site
// returns an error -- not a panic -- for the exact F1 attack shape.
func TestLoadTensorsSplit_AttackShape(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "shard0.gguf")
	if err := os.WriteFile(path, make([]byte, 64), 0o600); err != nil {
		t.Fatal(err)
	}
	r, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()

	shard := &File{DataOffset: 0}
	sf := &SplitFile{
		File: &File{
			Tensors: []TensorInfo{{
				Name:       "test.attack",
				Dimensions: attackDimensions,
				Type:       GGMLTypeF32,
				Offset:     0,
			}},
		},
		Shards:     []*File{shard},
		ShardIndex: map[string]int{"test.attack": 0},
	}

	defer func() {
		if rec := recover(); rec != nil {
			t.Fatalf("LoadTensorsSplit panicked instead of returning an error: %v", rec)
		}
	}()

	_, err = LoadTensorsSplit(sf, []*os.File{r})
	if err == nil {
		t.Fatal("expected error for F1 attack shape, got nil")
	}
}
