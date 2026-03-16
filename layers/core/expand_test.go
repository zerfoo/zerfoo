package core

import (
	"context"
	"testing"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
)

func TestExpandCPUParity(t *testing.T) {
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	ctx := context.Background()

	tests := []struct {
		name        string
		srcShape    []int
		srcData     []float32
		targetShape []float32
		wantShape   []int
		wantData    []float32
	}{
		{
			name:        "1x5x1_to_3x5x7",
			srcShape:    []int{1, 5, 1},
			srcData:     []float32{1, 2, 3, 4, 5},
			targetShape: []float32{3, 5, 7},
			wantShape:   []int{3, 5, 7},
			wantData: func() []float32 {
				out := make([]float32, 3*5*7)
				for b := 0; b < 3; b++ {
					for r := 0; r < 5; r++ {
						for c := 0; c < 7; c++ {
							out[b*5*7+r*7+c] = float32(r + 1)
						}
					}
				}
				return out
			}(),
		},
		{
			name:        "1D_to_2D",
			srcShape:    []int{3},
			srcData:     []float32{10, 20, 30},
			targetShape: []float32{2, 3},
			wantShape:   []int{2, 3},
			wantData:    []float32{10, 20, 30, 10, 20, 30},
		},
		{
			name:        "scalar_to_2x3",
			srcShape:    []int{1},
			srcData:     []float32{7},
			targetShape: []float32{2, 3},
			wantShape:   []int{2, 3},
			wantData:    []float32{7, 7, 7, 7, 7, 7},
		},
		{
			name:        "same_shape_noop",
			srcShape:    []int{2, 3},
			srcData:     []float32{1, 2, 3, 4, 5, 6},
			targetShape: []float32{2, 3},
			wantShape:   []int{2, 3},
			wantData:    []float32{1, 2, 3, 4, 5, 6},
		},
		{
			name:        "broadcast_middle_dim",
			srcShape:    []int{2, 1, 3},
			srcData:     []float32{1, 2, 3, 4, 5, 6},
			targetShape: []float32{2, 4, 3},
			wantShape:   []int{2, 4, 3},
			wantData: func() []float32 {
				out := make([]float32, 2*4*3)
				src := []float32{1, 2, 3, 4, 5, 6}
				for b := 0; b < 2; b++ {
					for r := 0; r < 4; r++ {
						for c := 0; c < 3; c++ {
							out[b*4*3+r*3+c] = src[b*3+c]
						}
					}
				}
				return out
			}(),
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			input, err := tensor.New(tc.srcShape, tc.srcData)
			if err != nil {
				t.Fatalf("create input: %v", err)
			}
			shapeTensor, err := tensor.New([]int{len(tc.targetShape)}, tc.targetShape)
			if err != nil {
				t.Fatalf("create shape tensor: %v", err)
			}

			expand := &Expand[float32]{engine: engine}
			got, err := expand.Forward(ctx, input, shapeTensor)
			if err != nil {
				t.Fatalf("Forward: %v", err)
			}

			gotShape := got.Shape()
			if len(gotShape) != len(tc.wantShape) {
				t.Fatalf("shape len = %d, want %d", len(gotShape), len(tc.wantShape))
			}
			for i := range gotShape {
				if gotShape[i] != tc.wantShape[i] {
					t.Fatalf("shape[%d] = %d, want %d", i, gotShape[i], tc.wantShape[i])
				}
			}

			gotData := got.Data()
			if len(gotData) != len(tc.wantData) {
				t.Fatalf("data len = %d, want %d", len(gotData), len(tc.wantData))
			}
			for i := range gotData {
				if gotData[i] != tc.wantData[i] {
					t.Errorf("data[%d] = %v, want %v", i, gotData[i], tc.wantData[i])
				}
			}
		})
	}
}
