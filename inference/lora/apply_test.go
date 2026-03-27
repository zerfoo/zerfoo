package lora

import (
	"math"
	"testing"
)

func TestApply(t *testing.T) {
	tests := []struct {
		name        string
		x           []float32
		baseOutput  []float32
		layer       *Layer
		scaleFactor float64
		batch       int
		inDim       int
		outDim      int
		want        []float32
	}{
		{
			name: "zero adapter produces no change",
			x:    []float32{1, 2, 3},
			baseOutput: []float32{10, 20},
			layer: &Layer{
				A: [][]float32{{0, 0, 0}},       // [rank=1, inDim=3]
				B: [][]float32{{0}, {0}},         // [outDim=2, rank=1]
			},
			scaleFactor: 1.0,
			batch:       1,
			inDim:       3,
			outDim:      2,
			want:        []float32{10, 20},
		},
		{
			name: "known values 2x3",
			// x = [1, 2]
			// A = [[1, 0], [0, 1]]  (rank=2, inDim=2)
			// B = [[1, 0], [0, 1], [1, 1]]  (outDim=3, rank=2)
			// hidden = x @ A^T = [1, 2]
			// delta = hidden @ B^T = [1*1+2*0, 1*0+2*1, 1*1+2*1] = [1, 2, 3]
			// scale = 0.5
			// output = [0, 0, 0] + 0.5 * [1, 2, 3] = [0.5, 1.0, 1.5]
			x:          []float32{1, 2},
			baseOutput: []float32{0, 0, 0},
			layer: &Layer{
				A: [][]float32{{1, 0}, {0, 1}},
				B: [][]float32{{1, 0}, {0, 1}, {1, 1}},
			},
			scaleFactor: 0.5,
			batch:       1,
			inDim:       2,
			outDim:      3,
			want:        []float32{0.5, 1.0, 1.5},
		},
		{
			name: "identity-like adapter adds scaled input",
			// rank=2, inDim=2, outDim=2
			// A = I, B = I → delta = x
			// output = base + scale * x
			x:          []float32{3, 4},
			baseOutput: []float32{10, 20},
			layer: &Layer{
				A: [][]float32{{1, 0}, {0, 1}},
				B: [][]float32{{1, 0}, {0, 1}},
			},
			scaleFactor: 2.0,
			batch:       1,
			inDim:       2,
			outDim:      2,
			want:        []float32{16, 28}, // [10+2*3, 20+2*4]
		},
		{
			name: "batch of two elements",
			// Same adapter as identity-like, batch=2
			x:          []float32{1, 2, 3, 4},
			baseOutput: []float32{0, 0, 0, 0},
			layer: &Layer{
				A: [][]float32{{1, 0}, {0, 1}},
				B: [][]float32{{1, 0}, {0, 1}},
			},
			scaleFactor: 1.0,
			batch:       2,
			inDim:       2,
			outDim:      2,
			want:        []float32{1, 2, 3, 4},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Copy baseOutput so we can verify in-place modification.
			out := make([]float32, len(tt.baseOutput))
			copy(out, tt.baseOutput)

			got := Apply(out, tt.x, tt.layer, tt.scaleFactor, tt.batch, tt.inDim, tt.outDim)

			if len(got) != len(tt.want) {
				t.Fatalf("length mismatch: got %d, want %d", len(got), len(tt.want))
			}
			for i := range got {
				if diff := math.Abs(float64(got[i] - tt.want[i])); diff > 1e-5 {
					t.Errorf("got[%d] = %f, want %f (diff %e)", i, got[i], tt.want[i], diff)
				}
			}
		})
	}
}

func TestApplyBatch_MultipleElements(t *testing.T) {
	// rank=2, inDim=3, outDim=2, batch=3
	layer := &Layer{
		A: [][]float32{{1, 0, 0}, {0, 1, 0}}, // picks first two dims
		B: [][]float32{{1, 1}, {0, 1}},        // outDim=2, rank=2
	}
	scale := 1.0
	batch, inDim, outDim := 3, 3, 2

	// x rows: [1,2,3], [4,5,6], [7,8,9]
	x := []float32{1, 2, 3, 4, 5, 6, 7, 8, 9}
	baseOutput := make([]float32, batch*outDim)

	ApplyBatch(baseOutput, x, layer, scale, batch, inDim, outDim)

	// hidden[b] = [x[b][0], x[b][1]]  (A picks first two dims)
	// delta[b][0] = hidden[0]*1 + hidden[1]*1
	// delta[b][1] = hidden[0]*0 + hidden[1]*1
	// Row 0: hidden=[1,2], delta=[3,2]
	// Row 1: hidden=[4,5], delta=[9,5]
	// Row 2: hidden=[7,8], delta=[15,8]
	want := []float32{3, 2, 9, 5, 15, 8}

	for i := range baseOutput {
		if diff := math.Abs(float64(baseOutput[i] - want[i])); diff > 1e-5 {
			t.Errorf("baseOutput[%d] = %f, want %f", i, baseOutput[i], want[i])
		}
	}
}

func TestApply_MergedWeightParity(t *testing.T) {
	// Verify that LoRA apply produces the same result as merging weights.
	//
	// W_merged = W_base + scale * B @ A
	// output_merged = x @ W_merged^T
	// output_lora   = (x @ W_base^T) then Apply()
	//
	// These must match within 1e-5.

	inDim, outDim, rank := 4, 3, 2
	scale := 0.75

	// W_base [outDim, inDim]
	wBase := [][]float32{
		{1, 2, 3, 4},
		{5, 6, 7, 8},
		{0.5, 1.5, 2.5, 3.5},
	}

	layer := &Layer{
		A: [][]float32{
			{0.1, 0.2, 0.3, 0.4},
			{0.5, 0.6, 0.7, 0.8},
		},
		B: [][]float32{
			{1.0, 0.5},
			{0.3, 0.7},
			{0.2, 0.9},
		},
	}

	x := []float32{1, 0.5, -1, 2}
	batch := 1

	// Compute W_merged = W_base + scale * B @ A
	wMerged := make([][]float32, outDim)
	for o := 0; o < outDim; o++ {
		wMerged[o] = make([]float32, inDim)
		copy(wMerged[o], wBase[o])
		for k := 0; k < inDim; k++ {
			var ba float32
			for r := 0; r < rank; r++ {
				ba += layer.B[o][r] * layer.A[r][k]
			}
			wMerged[o][k] += float32(scale) * ba
		}
	}

	// output_merged = x @ W_merged^T
	outputMerged := make([]float32, outDim)
	for o := 0; o < outDim; o++ {
		for k := 0; k < inDim; k++ {
			outputMerged[o] += x[k] * wMerged[o][k]
		}
	}

	// output_lora = x @ W_base^T, then Apply
	outputLora := make([]float32, outDim)
	for o := 0; o < outDim; o++ {
		for k := 0; k < inDim; k++ {
			outputLora[o] += x[k] * wBase[o][k]
		}
	}
	Apply(outputLora, x, layer, scale, batch, inDim, outDim)

	for i := 0; i < outDim; i++ {
		diff := math.Abs(float64(outputMerged[i] - outputLora[i]))
		if diff > 1e-5 {
			t.Errorf("output[%d]: merged=%f, lora=%f, diff=%e", i, outputMerged[i], outputLora[i], diff)
		}
	}
}
