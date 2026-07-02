package inference

import (
	"testing"

	"github.com/zerfoo/zerfoo/layers/attention"
)

func TestBuildCausalSlidingWindowMask(t *testing.T) {
	tests := []struct {
		name       string
		seqLen     int
		windowSize int
		// wantZero[i][j] = true means position (i,j) should be 0 (allowed)
		wantZero [][]bool
	}{
		{
			name:       "window=2, seqLen=4",
			seqLen:     4,
			windowSize: 2,
			// Row i: attend to positions j where j <= i AND i-j < 2
			// i=0: j=0 → [0, -neg, -neg, -neg]
			// i=1: j=0,1 → [0, 0, -neg, -neg]
			// i=2: j=1,2 → [-neg, 0, 0, -neg]
			// i=3: j=2,3 → [-neg, -neg, 0, 0]
			wantZero: [][]bool{
				{true, false, false, false},
				{true, true, false, false},
				{false, true, true, false},
				{false, false, true, true},
			},
		},
		{
			name:       "window=3, seqLen=4",
			seqLen:     4,
			windowSize: 3,
			// i=0: j=0 → [0, -neg, -neg, -neg]
			// i=1: j=0,1 → [0, 0, -neg, -neg]
			// i=2: j=0,1,2 → [0, 0, 0, -neg]
			// i=3: j=1,2,3 (j=0: i-j=3 NOT < 3) → [-neg, 0, 0, 0]
			wantZero: [][]bool{
				{true, false, false, false},
				{true, true, false, false},
				{true, true, true, false},
				{false, true, true, true},
			},
		},
		{
			name:       "window=1, seqLen=3",
			seqLen:     3,
			windowSize: 1,
			// Each position only attends to itself
			wantZero: [][]bool{
				{true, false, false},
				{false, true, false},
				{false, false, true},
			},
		},
		{
			name:       "window larger than seqLen acts as full causal",
			seqLen:     3,
			windowSize: 10,
			wantZero: [][]bool{
				{true, false, false},
				{true, true, false},
				{true, true, true},
			},
		},
		{
			name:       "seqLen=1 always attends to self",
			seqLen:     1,
			windowSize: 4096,
			wantZero: [][]bool{
				{true},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			mask := attention.BuildCausalSlidingWindowMask[float32](tt.seqLen, tt.windowSize)
			if mask == nil {
				t.Fatal("mask is nil")
			}

			shape := mask.Shape()
			wantShape := []int{1, 1, tt.seqLen, tt.seqLen}
			if len(shape) != 4 || shape[0] != wantShape[0] || shape[1] != wantShape[1] || shape[2] != wantShape[2] || shape[3] != wantShape[3] {
				t.Fatalf("shape = %v, want %v", shape, wantShape)
			}

			data := mask.Data()
			for i := range tt.seqLen {
				for j := range tt.seqLen {
					val := data[i*tt.seqLen+j]
					if tt.wantZero[i][j] {
						if val != 0 {
							t.Errorf("mask[%d][%d] = %v, want 0", i, j, val)
						}
					} else {
						if val >= 0 {
							t.Errorf("mask[%d][%d] = %v, want large negative", i, j, val)
						}
					}
				}
			}
		})
	}
}
