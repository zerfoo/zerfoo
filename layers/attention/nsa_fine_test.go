package attention

import (
	"context"
	"math"
	"testing"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
)

func TestNSAFineSelection(t *testing.T) {
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	ctx := context.Background()

	tests := []struct {
		name       string
		batch      int
		numHeads   int
		numKVHeads int
		headDim    int
		seqQ       int
		seqKV      int
		topTokens  int
		qData      []float32
		kData      []float32
		vData      []float32
		wantIdx    []int // expected top-f indices per query position
	}{
		{
			name:       "single head, select top-2 of 4",
			batch:      1,
			numHeads:   1,
			numKVHeads: 1,
			headDim:    2,
			seqQ:       1,
			seqKV:      4,
			topTokens:  2,
			// Q = [[1, 0]]  → dot with K rows: k0=1, k1=0, k2=2, k3=-1
			// Top-2 by score: indices 2 (score=2), 0 (score=1)
			qData: []float32{1, 0},
			kData: []float32{
				1, 0, // k0: dot=1
				0, 1, // k1: dot=0
				2, 0, // k2: dot=2
				-1, 0, // k3: dot=-1
			},
			vData: []float32{
				10, 20,
				30, 40,
				50, 60,
				70, 80,
			},
			wantIdx: []int{2, 0},
		},
		{
			name:       "two query positions",
			batch:      1,
			numHeads:   1,
			numKVHeads: 1,
			headDim:    2,
			seqQ:       2,
			seqKV:      3,
			topTokens:  2,
			// Q[0] = [1,0]: scores = [1, 0, 3] → top-2: {2, 0}
			// Q[1] = [0,1]: scores = [0, 2, 0] → top-2: {1, 0} or {1, 2} (tied)
			qData: []float32{
				1, 0,
				0, 1,
			},
			kData: []float32{
				1, 0,
				0, 2,
				3, 0,
			},
			vData: []float32{
				1, 1,
				2, 2,
				3, 3,
			},
			wantIdx: []int{2, 0, 1, 0},
		},
		{
			name:       "GQA: 2 query heads, 1 KV head",
			batch:      1,
			numHeads:   2,
			numKVHeads: 1,
			headDim:    2,
			seqQ:       1,
			seqKV:      3,
			topTokens:  2,
			// Both query heads share the same KV head.
			// Q head0 = [1,0]: scores against K = [1, 0, 2] → top-2: {2, 0}
			// Q head1 = [0,1]: scores against K = [0, 1, 0] → top-2: {1, 0} or {1, 2}
			qData: []float32{
				1, 0, // head 0
				0, 1, // head 1
			},
			kData: []float32{
				1, 0,
				0, 1,
				2, 0,
			},
			vData: []float32{
				10, 20,
				30, 40,
				50, 60,
			},
			wantIdx: []int{2, 0, 1, 0},
		},
		{
			name:       "topTokens exceeds seqKV",
			batch:      1,
			numHeads:   1,
			numKVHeads: 1,
			headDim:    2,
			seqQ:       1,
			seqKV:      2,
			topTokens:  5,
			qData:      []float32{1, 0},
			kData:      []float32{1, 0, 0, 1},
			vData:      []float32{10, 20, 30, 40},
			wantIdx:    []int{0, 1},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			nsa := NewNSAFineSelection[float32](
				engine, tt.topTokens, tt.numHeads, tt.numKVHeads, tt.headDim,
			)

			Q, err := tensor.New[float32]([]int{tt.batch, tt.numHeads, tt.seqQ, tt.headDim}, tt.qData)
			if err != nil {
				t.Fatalf("create Q: %v", err)
			}
			K, err := tensor.New[float32]([]int{tt.batch, tt.numKVHeads, tt.seqKV, tt.headDim}, tt.kData)
			if err != nil {
				t.Fatalf("create K: %v", err)
			}
			V, err := tensor.New[float32]([]int{tt.batch, tt.numKVHeads, tt.seqKV, tt.headDim}, tt.vData)
			if err != nil {
				t.Fatalf("create V: %v", err)
			}

			// Verify selected indices match expectations.
			gotIdx := nsa.SelectedIndices(Q, K)
			if len(gotIdx) != len(tt.wantIdx) {
				t.Fatalf("index count: got %d, want %d", len(gotIdx), len(tt.wantIdx))
			}
			for i, want := range tt.wantIdx {
				if gotIdx[i] != want {
					t.Errorf("index[%d]: got %d, want %d", i, gotIdx[i], want)
				}
			}

			// Verify Forward produces correct shape and reasonable values.
			out, err := nsa.Forward(ctx, Q, K, V)
			if err != nil {
				t.Fatalf("Forward: %v", err)
			}
			wantShape := []int{tt.batch, tt.numHeads, tt.seqQ, tt.headDim}
			gotShape := out.Shape()
			for i := range wantShape {
				if gotShape[i] != wantShape[i] {
					t.Fatalf("shape[%d]: got %d, want %d", i, gotShape[i], wantShape[i])
				}
			}

			// Output should be a convex combination of selected V rows,
			// so each element should be finite.
			for i, v := range out.Data() {
				if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
					t.Errorf("output[%d] is not finite: %v", i, v)
				}
			}
		})
	}
}

func TestNSAFineSelectionForwardValues(t *testing.T) {
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	ctx := context.Background()

	// Single head, 1 query, 2 KV positions, top-1.
	// With top-1, output should equal the V row of the highest-scoring K.
	nsa := NewNSAFineSelection[float32](engine, 1, 1, 1, 2)

	// Q=[1,0], K=[[1,0],[0,1]], V=[[10,20],[30,40]]
	// Scores: [1/sqrt(2), 0] → top-1 is index 0.
	// Softmax over single element = 1.0.
	// Output = V[0] = [10, 20].
	Q, _ := tensor.New[float32]([]int{1, 1, 1, 2}, []float32{1, 0})
	K, _ := tensor.New[float32]([]int{1, 1, 2, 2}, []float32{1, 0, 0, 1})
	V, _ := tensor.New[float32]([]int{1, 1, 2, 2}, []float32{10, 20, 30, 40})

	out, err := nsa.Forward(ctx, Q, K, V)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}

	got := out.Data()
	want := []float32{10, 20}
	const tol = 1e-5
	for i := range want {
		if diff := math.Abs(float64(got[i] - want[i])); diff > tol {
			t.Errorf("output[%d]: got %f, want %f (diff %f)", i, got[i], want[i], diff)
		}
	}
}

func TestNSAFineSelectionBatch(t *testing.T) {
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	ctx := context.Background()

	// batch=2, 1 head, 1 query pos, 3 KV positions, top-1.
	nsa := NewNSAFineSelection[float32](engine, 1, 1, 1, 2)

	Q, _ := tensor.New[float32]([]int{2, 1, 1, 2}, []float32{
		1, 0, // batch 0
		0, 1, // batch 1
	})
	K, _ := tensor.New[float32]([]int{2, 1, 3, 2}, []float32{
		// batch 0: scores = [1, 0, 2]/sqrt(2) → top: idx 2
		1, 0, 0, 1, 2, 0,
		// batch 1: scores = [0, 1, 0]/sqrt(2) → top: idx 1
		1, 0, 0, 1, 2, 0,
	})
	V, _ := tensor.New[float32]([]int{2, 1, 3, 2}, []float32{
		10, 20, 30, 40, 50, 60, // batch 0
		70, 80, 90, 100, 110, 120, // batch 1
	})

	out, err := nsa.Forward(ctx, Q, K, V)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}

	got := out.Data()
	// batch 0: top-1 is idx 2 → V[2] = [50, 60]
	// batch 1: top-1 is idx 1 → V[1] = [90, 100]
	want := []float32{50, 60, 90, 100}
	const tol = 1e-5
	for i := range want {
		if diff := math.Abs(float64(got[i] - want[i])); diff > tol {
			t.Errorf("output[%d]: got %f, want %f", i, got[i], want[i])
		}
	}
}
