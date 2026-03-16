package core

import (
	"context"
	"testing"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
)

func TestTile_Forward(t *testing.T) {
	eng := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	node := &Tile[float32]{engine: eng}

	tests := []struct {
		name      string
		inShape   []int
		inData    []float32
		repeats   []float32
		wantShape []int
		wantData  []float32
	}{
		{
			name:      "1D repeat 3",
			inShape:   []int{2},
			inData:    []float32{1, 2},
			repeats:   []float32{3},
			wantShape: []int{6},
			wantData:  []float32{1, 2, 1, 2, 1, 2},
		},
		{
			name:      "2D repeat rows",
			inShape:   []int{1, 3},
			inData:    []float32{1, 2, 3},
			repeats:   []float32{2, 1},
			wantShape: []int{2, 3},
			wantData:  []float32{1, 2, 3, 1, 2, 3},
		},
		{
			name:      "2D repeat cols",
			inShape:   []int{2, 1},
			inData:    []float32{1, 2},
			repeats:   []float32{1, 3},
			wantShape: []int{2, 3},
			wantData:  []float32{1, 1, 1, 2, 2, 2},
		},
		{
			name:      "2D repeat both",
			inShape:   []int{2, 2},
			inData:    []float32{1, 2, 3, 4},
			repeats:   []float32{2, 2},
			wantShape: []int{4, 4},
			wantData: []float32{
				1, 2, 1, 2,
				3, 4, 3, 4,
				1, 2, 1, 2,
				3, 4, 3, 4,
			},
		},
		{
			name:      "no repeat",
			inShape:   []int{2, 3},
			inData:    []float32{1, 2, 3, 4, 5, 6},
			repeats:   []float32{1, 1},
			wantShape: []int{2, 3},
			wantData:  []float32{1, 2, 3, 4, 5, 6},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			in, _ := tensor.New[float32](tc.inShape, tc.inData)
			rep, _ := tensor.New[float32]([]int{len(tc.repeats)}, tc.repeats)
			out, err := node.Forward(context.Background(), in, rep)
			if err != nil {
				t.Fatalf("Forward failed: %v", err)
			}
			gotShape := out.Shape()
			if len(gotShape) != len(tc.wantShape) {
				t.Fatalf("shape = %v, want %v", gotShape, tc.wantShape)
			}
			for i := range gotShape {
				if gotShape[i] != tc.wantShape[i] {
					t.Errorf("shape[%d] = %d, want %d", i, gotShape[i], tc.wantShape[i])
				}
			}
			got := out.Data()
			if len(got) != len(tc.wantData) {
				t.Fatalf("data length = %d, want %d", len(got), len(tc.wantData))
			}
			for i := range got {
				if got[i] != tc.wantData[i] {
					t.Errorf("data[%d] = %v, want %v", i, got[i], tc.wantData[i])
				}
			}
		})
	}
}

func TestTile_WrongInputCount(t *testing.T) {
	eng := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	node := &Tile[float32]{engine: eng}
	a, _ := tensor.New[float32]([]int{3}, nil)
	_, err := node.Forward(context.Background(), a)
	if err == nil {
		t.Error("expected error for 1 input")
	}
}

func TestTile_OpType(t *testing.T) {
	node := &Tile[float32]{}
	if got := node.OpType(); got != "Tile" {
		t.Errorf("OpType() = %q, want %q", got, "Tile")
	}
}
