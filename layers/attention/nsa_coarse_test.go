package attention

import (
	"context"
	"math"
	"testing"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
)

func TestNSACoarseCompression_Forward_OutputShape(t *testing.T) {
	tests := []struct {
		name       string
		batch      int
		numHeads   int
		numKVHeads int
		seqQ       int
		seqKV      int
		headDim    int
		blockSize  int
		topBlocks  int
	}{
		{
			name:       "basic",
			batch:      1,
			numHeads:   4,
			numKVHeads: 4,
			seqQ:       4,
			seqKV:      8,
			headDim:    8,
			blockSize:  2,
			topBlocks:  2,
		},
		{
			name:       "GQA heads",
			batch:      2,
			numHeads:   8,
			numKVHeads: 2,
			seqQ:       4,
			seqKV:      16,
			headDim:    16,
			blockSize:  4,
			topBlocks:  2,
		},
		{
			name:       "single query position",
			batch:      1,
			numHeads:   4,
			numKVHeads: 4,
			seqQ:       1,
			seqKV:      8,
			headDim:    8,
			blockSize:  2,
			topBlocks:  3,
		},
		{
			name:       "topBlocks exceeds numBlocks",
			batch:      1,
			numHeads:   2,
			numKVHeads: 2,
			seqQ:       2,
			seqKV:      4,
			headDim:    4,
			blockSize:  2,
			topBlocks:  10,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ctx := context.Background()
			ops := numeric.Float32Ops{}
			engine := compute.NewCPUEngine[float32](ops)

			nsa := NewNSACoarseCompression[float32](
				engine, ops, tt.blockSize, tt.topBlocks,
				tt.numHeads, tt.numKVHeads, tt.headDim,
			)

			qSize := tt.batch * tt.numHeads * tt.seqQ * tt.headDim
			kSize := tt.batch * tt.numKVHeads * tt.seqKV * tt.headDim
			vSize := kSize

			qData := make([]float32, qSize)
			kData := make([]float32, kSize)
			vData := make([]float32, vSize)
			for i := range qData {
				qData[i] = float32(i%7) * 0.1
			}
			for i := range kData {
				kData[i] = float32(i%5) * 0.1
			}
			for i := range vData {
				vData[i] = float32(i%11) * 0.1
			}

			Q, err := tensor.New[float32]([]int{tt.batch, tt.numHeads, tt.seqQ, tt.headDim}, qData)
			if err != nil {
				t.Fatalf("create Q: %v", err)
			}
			K, err := tensor.New[float32]([]int{tt.batch, tt.numKVHeads, tt.seqKV, tt.headDim}, kData)
			if err != nil {
				t.Fatalf("create K: %v", err)
			}
			V, err := tensor.New[float32]([]int{tt.batch, tt.numKVHeads, tt.seqKV, tt.headDim}, vData)
			if err != nil {
				t.Fatalf("create V: %v", err)
			}

			out, err := nsa.Forward(ctx, Q, K, V)
			if err != nil {
				t.Fatalf("Forward: %v", err)
			}

			wantShape := []int{tt.batch, tt.numHeads, tt.seqQ, tt.headDim}
			gotShape := out.Shape()
			if len(gotShape) != len(wantShape) {
				t.Fatalf("output rank: got %d, want %d", len(gotShape), len(wantShape))
			}
			for i := range wantShape {
				if gotShape[i] != wantShape[i] {
					t.Errorf("output shape[%d]: got %d, want %d", i, gotShape[i], wantShape[i])
				}
			}
		})
	}
}

func TestNSACoarseCompression_BlockSelection(t *testing.T) {
	// Verify that the top-k block selection picks the correct blocks.
	// Set up K so that block 1 has much larger key values than block 0,
	// making it the clear winner for selection with topBlocks=1.
	ctx := context.Background()
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	batch := 1
	numHeads := 1
	numKVHeads := 1
	seqQ := 1
	seqKV := 4
	headDim := 2
	blockSize := 2
	topBlocks := 1

	nsa := NewNSACoarseCompression[float32](
		engine, ops, blockSize, topBlocks,
		numHeads, numKVHeads, headDim,
	)

	// Q = [1, 1] — a simple query vector.
	Q, err := tensor.New[float32]([]int{batch, numHeads, seqQ, headDim}, []float32{1, 1})
	if err != nil {
		t.Fatalf("create Q: %v", err)
	}

	// K: block 0 = [[0.1, 0.1], [0.1, 0.1]], block 1 = [[10, 10], [10, 10]]
	// Block 1 has much higher dot product with Q.
	K, err := tensor.New[float32]([]int{batch, numKVHeads, seqKV, headDim}, []float32{
		0.1, 0.1, 0.1, 0.1, 10, 10, 10, 10,
	})
	if err != nil {
		t.Fatalf("create K: %v", err)
	}

	// V: block 0 = [[1, 0], [1, 0]], block 1 = [[0, 1], [0, 1]]
	// If block 1 is selected, output should be close to [0, 1].
	V, err := tensor.New[float32]([]int{batch, numKVHeads, seqKV, headDim}, []float32{
		1, 0, 1, 0, 0, 1, 0, 1,
	})
	if err != nil {
		t.Fatalf("create V: %v", err)
	}

	out, err := nsa.Forward(ctx, Q, K, V)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}

	outData := out.Data()
	// With topBlocks=1 and block 1 selected, all attention goes to block 1's V.
	// Both tokens in block 1 have V=[0,1], so output = [0, 1].
	if math.Abs(float64(outData[0])) > 1e-4 {
		t.Errorf("output[0]: got %f, want ~0", outData[0])
	}
	if math.Abs(float64(outData[1])-1.0) > 1e-4 {
		t.Errorf("output[1]: got %f, want ~1", outData[1])
	}
}

func TestNSACoarseCompression_TopKIndices(t *testing.T) {
	ops := numeric.Float32Ops{}

	tests := []struct {
		name    string
		scores  []float32
		k       int
		wantLen int
	}{
		{
			name:    "basic top-2",
			scores:  []float32{1.0, 3.0, 2.0, 5.0},
			k:       2,
			wantLen: 2,
		},
		{
			name:    "k exceeds length",
			scores:  []float32{1.0, 2.0},
			k:       5,
			wantLen: 2,
		},
		{
			name:    "k equals length",
			scores:  []float32{3.0, 1.0, 2.0},
			k:       3,
			wantLen: 3,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			indices := topKIndicesT(tt.scores, tt.k, ops)
			if len(indices) != tt.wantLen {
				t.Fatalf("topKIndices length: got %d, want %d", len(indices), tt.wantLen)
			}

			// Verify returned indices point to top-k values.
			for i := 1; i < len(indices); i++ {
				if tt.scores[indices[i]] > tt.scores[indices[i-1]] {
					t.Errorf("topKIndices not sorted by score: score[%d]=%f > score[%d]=%f",
						indices[i], tt.scores[indices[i]], indices[i-1], tt.scores[indices[i-1]])
				}
			}
		})
	}
}

func TestNSACoarseCompression_Backward_StraightThrough(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	nsa := NewNSACoarseCompression[float32](engine, ops, 2, 1, 2, 2, 4)

	dOutData := make([]float32, 2*2*3*4)
	for i := range dOutData {
		dOutData[i] = float32(i) * 0.01
	}
	dOut, err := tensor.New[float32]([]int{2, 2, 3, 4}, dOutData)
	if err != nil {
		t.Fatalf("create dOut: %v", err)
	}

	grads, err := nsa.Backward(context.Background(), 0, dOut)
	if err != nil {
		t.Fatalf("Backward: %v", err)
	}

	if len(grads) != 3 {
		t.Fatalf("expected 3 gradients, got %d", len(grads))
	}

	// dQ should equal dOut (straight-through).
	if grads[0] != dOut {
		t.Error("dQ should be the same tensor as dOut (straight-through)")
	}

	// dK and dV should be nil.
	if grads[1] != nil {
		t.Error("dK should be nil")
	}
	if grads[2] != nil {
		t.Error("dV should be nil")
	}
}

func TestNSACoarseCompression_Metadata(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	nsa := NewNSACoarseCompression[float32](engine, ops, 4, 2, 8, 2, 64)

	if nsa.OpType() != "NSACoarseCompression" {
		t.Errorf("OpType: got %q, want %q", nsa.OpType(), "NSACoarseCompression")
	}

	attrs := nsa.Attributes()
	if attrs["block_size"] != 4 {
		t.Errorf("block_size: got %v, want 4", attrs["block_size"])
	}
	if attrs["top_blocks"] != 2 {
		t.Errorf("top_blocks: got %v, want 2", attrs["top_blocks"])
	}
	if attrs["num_heads"] != 8 {
		t.Errorf("num_heads: got %v, want 8", attrs["num_heads"])
	}

	if params := nsa.Parameters(); params != nil {
		t.Errorf("Parameters: got %v, want nil", params)
	}
}
