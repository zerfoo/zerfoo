package residual

import (
	"context"
	"math"
	"testing"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
)

func TestBlockAttnResForward(t *testing.T) {
	ctx := context.Background()
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	dim := 4
	blockSize := 3
	epsilon := float32(1e-6)

	bar, err := NewBlockAttnRes[float32](engine, ops, blockSize, dim, epsilon)
	if err != nil {
		t.Fatalf("NewBlockAttnRes: %v", err)
	}

	tests := []struct {
		name      string
		query     []float32
		blocks    [][]float32
		partial   []float32
		wantShape []int
	}{
		{
			name:  "3 completed blocks + 1 partial",
			query: []float32{1.0, 0.5, -0.3, 0.8},
			blocks: [][]float32{
				{0.1, 0.2, 0.3, 0.4},
				{0.5, 0.6, 0.7, 0.8},
				{0.9, 1.0, 1.1, 1.2},
			},
			partial:   []float32{0.3, 0.4, 0.5, 0.6},
			wantShape: []int{dim},
		},
		{
			name:  "1 completed block + 1 partial",
			query: []float32{0.2, 0.3, 0.4, 0.5},
			blocks: [][]float32{
				{1.0, 2.0, 3.0, 4.0},
			},
			partial:   []float32{0.5, 0.5, 0.5, 0.5},
			wantShape: []int{dim},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			query, err := tensor.New[float32]([]int{dim}, tc.query)
			if err != nil {
				t.Fatalf("create query: %v", err)
			}

			blocks := make([]*tensor.TensorNumeric[float32], len(tc.blocks))
			for i, bd := range tc.blocks {
				blocks[i], err = tensor.New[float32]([]int{dim}, bd)
				if err != nil {
					t.Fatalf("create block %d: %v", i, err)
				}
			}

			partial, err := tensor.New[float32]([]int{dim}, tc.partial)
			if err != nil {
				t.Fatalf("create partial: %v", err)
			}

			out, err := bar.Forward(ctx, query, blocks, partial)
			if err != nil {
				t.Fatalf("Forward: %v", err)
			}

			if !equalShape(out.Shape(), tc.wantShape) {
				t.Errorf("shape = %v, want %v", out.Shape(), tc.wantShape)
			}

			// Output should be finite.
			for i, v := range out.Data() {
				if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
					t.Errorf("output[%d] = %v, want finite", i, v)
				}
			}
		})
	}
}

func TestBlockAttnResBlockBoundary(t *testing.T) {
	// Verify that intra-block is additive: with 0 completed blocks and
	// a partial block, the output should be the partial block itself
	// (single-element softmax -> weight 1.0 -> identity).
	ctx := context.Background()
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	dim := 4
	epsilon := float32(1e-6)

	bar, err := NewBlockAttnRes[float32](engine, ops, 3, dim, epsilon)
	if err != nil {
		t.Fatalf("NewBlockAttnRes: %v", err)
	}

	partialData := []float32{1.0, 2.0, 3.0, 4.0}
	query, err := tensor.New[float32]([]int{dim}, []float32{0.5, 0.5, 0.5, 0.5})
	if err != nil {
		t.Fatalf("create query: %v", err)
	}
	partial, err := tensor.New[float32]([]int{dim}, partialData)
	if err != nil {
		t.Fatalf("create partial: %v", err)
	}

	// No completed blocks -- only partial block.
	out, err := bar.Forward(ctx, query, nil, partial)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}

	// With a single block, softmax gives weight=1.0, so output = partial.
	outData := out.Data()
	for i, want := range partialData {
		if diff := math.Abs(float64(outData[i] - want)); diff > 1e-5 {
			t.Errorf("output[%d] = %v, want %v (diff=%v)", i, outData[i], want, diff)
		}
	}
}

func TestBlockAttnResEmptyBlocks(t *testing.T) {
	// Edge case: 0 completed blocks, only partial block.
	ctx := context.Background()
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	dim := 8
	epsilon := float32(1e-6)

	bar, err := NewBlockAttnRes[float32](engine, ops, 4, dim, epsilon)
	if err != nil {
		t.Fatalf("NewBlockAttnRes: %v", err)
	}

	queryData := []float32{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8}
	partialData := []float32{1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0}

	query, err := tensor.New[float32]([]int{dim}, queryData)
	if err != nil {
		t.Fatalf("create query: %v", err)
	}
	partial, err := tensor.New[float32]([]int{dim}, partialData)
	if err != nil {
		t.Fatalf("create partial: %v", err)
	}

	out, err := bar.Forward(ctx, query, nil, partial)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}

	// Shape should match query.
	if !equalShape(out.Shape(), []int{dim}) {
		t.Errorf("shape = %v, want [%d]", out.Shape(), dim)
	}

	// Single block -> output = partial block exactly.
	outData := out.Data()
	for i, want := range partialData {
		if diff := math.Abs(float64(outData[i] - want)); diff > 1e-5 {
			t.Errorf("output[%d] = %v, want %v", i, outData[i], want)
		}
	}
}

func TestBlockAttnResAttentionWeights(t *testing.T) {
	ctx := context.Background()
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	dim := 4
	epsilon := float32(1e-6)

	bar, err := NewBlockAttnRes[float32](engine, ops, 2, dim, epsilon)
	if err != nil {
		t.Fatalf("NewBlockAttnRes: %v", err)
	}

	tests := []struct {
		name     string
		query    []float32
		blocks   [][]float32
		partial  []float32
		nBlocks  int // expected number of attention weights
	}{
		{
			name:    "2 blocks + partial",
			query:   []float32{1.0, 0.0, 0.0, 0.0},
			blocks:  [][]float32{{1, 0, 0, 0}, {0, 1, 0, 0}},
			partial: []float32{0, 0, 1, 0},
			nBlocks: 3,
		},
		{
			name:    "0 blocks + partial",
			query:   []float32{1.0, 1.0, 1.0, 1.0},
			blocks:  nil,
			partial: []float32{0.5, 0.5, 0.5, 0.5},
			nBlocks: 1,
		},
		{
			name:    "1 block + partial",
			query:   []float32{0.3, 0.7, 0.1, 0.9},
			blocks:  [][]float32{{0.5, 0.5, 0.5, 0.5}},
			partial: []float32{0.2, 0.8, 0.2, 0.8},
			nBlocks: 2,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			query, err := tensor.New[float32]([]int{dim}, tc.query)
			if err != nil {
				t.Fatalf("create query: %v", err)
			}

			blocks := make([]*tensor.TensorNumeric[float32], len(tc.blocks))
			for i, bd := range tc.blocks {
				blocks[i], err = tensor.New[float32]([]int{dim}, bd)
				if err != nil {
					t.Fatalf("create block %d: %v", i, err)
				}
			}

			partial, err := tensor.New[float32]([]int{dim}, tc.partial)
			if err != nil {
				t.Fatalf("create partial: %v", err)
			}

			weights, err := bar.AttentionWeights(ctx, query, blocks, partial)
			if err != nil {
				t.Fatalf("AttentionWeights: %v", err)
			}

			data := weights.Data()
			if len(data) != tc.nBlocks {
				t.Fatalf("got %d weights, want %d", len(data), tc.nBlocks)
			}

			// Verify weights sum to 1.
			var sum float64
			for _, w := range data {
				sum += float64(w)
				if w < 0 {
					t.Errorf("negative weight: %v", w)
				}
			}
			if diff := math.Abs(sum - 1.0); diff > 1e-5 {
				t.Errorf("weights sum = %v, want 1.0", sum)
			}
		})
	}
}

func TestBlockAttnResInvalidBlockSize(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	_, err := NewBlockAttnRes[float32](engine, ops, 0, 4, 1e-6)
	if err == nil {
		t.Error("expected error for blockSize=0, got nil")
	}

	_, err = NewBlockAttnRes[float32](engine, ops, -1, 4, 1e-6)
	if err == nil {
		t.Error("expected error for blockSize=-1, got nil")
	}
}

func equalShape(a, b []int) bool {
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
