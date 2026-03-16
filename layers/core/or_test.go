package core

import (
	"context"
	"testing"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
)

func TestOr_Forward(t *testing.T) {
	eng := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	ops := numeric.Float32Ops{}
	node := &Or[float32]{engine: eng, ops: ops}

	tests := []struct {
		name   string
		aShape []int
		aData  []float32
		bShape []int
		bData  []float32
		want   []float32
	}{
		{
			name:   "both nonzero",
			aShape: []int{3}, aData: []float32{1, 2, 3},
			bShape: []int{3}, bData: []float32{4, 5, 6},
			want: []float32{1, 1, 1},
		},
		{
			name:   "both zero",
			aShape: []int{3}, aData: []float32{0, 0, 0},
			bShape: []int{3}, bData: []float32{0, 0, 0},
			want: []float32{0, 0, 0},
		},
		{
			name:   "a zero b nonzero",
			aShape: []int{3}, aData: []float32{0, 0, 0},
			bShape: []int{3}, bData: []float32{1, 2, 3},
			want: []float32{1, 1, 1},
		},
		{
			name:   "mixed",
			aShape: []int{4}, aData: []float32{0, 1, 0, 1},
			bShape: []int{4}, bData: []float32{0, 0, 1, 1},
			want: []float32{0, 1, 1, 1},
		},
		{
			name:   "scalar broadcast b",
			aShape: []int{3}, aData: []float32{0, 1, 0},
			bShape: []int{1}, bData: []float32{0},
			want: []float32{0, 1, 0},
		},
		{
			name:   "scalar broadcast a",
			aShape: []int{1}, aData: []float32{0},
			bShape: []int{3}, bData: []float32{0, 1, 0},
			want: []float32{0, 1, 0},
		},
		{
			name:   "scalar broadcast b nonzero",
			aShape: []int{3}, aData: []float32{0, 0, 0},
			bShape: []int{1}, bData: []float32{1},
			want: []float32{1, 1, 1},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			a, _ := tensor.New[float32](tc.aShape, tc.aData)
			b, _ := tensor.New[float32](tc.bShape, tc.bData)
			out, err := node.Forward(context.Background(), a, b)
			if err != nil {
				t.Fatalf("Forward failed: %v", err)
			}
			got := out.Data()
			if len(got) != len(tc.want) {
				t.Fatalf("output length %d, want %d", len(got), len(tc.want))
			}
			for i := range got {
				if got[i] != tc.want[i] {
					t.Errorf("output[%d] = %v, want %v", i, got[i], tc.want[i])
				}
			}
		})
	}
}

func TestOr_Broadcasting(t *testing.T) {
	eng := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	ops := numeric.Float32Ops{}
	node := &Or[float32]{engine: eng, ops: ops}

	tests := []struct {
		name      string
		aShape    []int
		aData     []float32
		bShape    []int
		bData     []float32
		wantShape []int
		want      []float32
	}{
		{
			name:      "3D broadcast [1,5,1] x [1,1,3]",
			aShape:    []int{1, 5, 1},
			aData:     []float32{0, 1, 0, 1, 0},
			bShape:    []int{1, 1, 3},
			bData:     []float32{0, 1, 0},
			wantShape: []int{1, 5, 3},
			want: []float32{
				0, 1, 0, // row0: a=0, b=[0,1,0]
				1, 1, 1, // row1: a=1, b=[0,1,0]
				0, 1, 0, // row2: a=0
				1, 1, 1, // row3: a=1
				0, 1, 0, // row4: a=0
			},
		},
		{
			name:      "scalar vs 2D [1] x [2,3]",
			aShape:    []int{1},
			aData:     []float32{1},
			bShape:    []int{2, 3},
			bData:     []float32{0, 0, 0, 1, 1, 1},
			wantShape: []int{2, 3},
			want:      []float32{1, 1, 1, 1, 1, 1},
		},
		{
			name:      "2D broadcast [3,1] x [1,4]",
			aShape:    []int{3, 1},
			aData:     []float32{0, 1, 0},
			bShape:    []int{1, 4},
			bData:     []float32{0, 1, 0, 1},
			wantShape: []int{3, 4},
			want: []float32{
				0, 1, 0, 1,
				1, 1, 1, 1,
				0, 1, 0, 1,
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			a, _ := tensor.New[float32](tc.aShape, tc.aData)
			b, _ := tensor.New[float32](tc.bShape, tc.bData)
			out, err := node.Forward(context.Background(), a, b)
			if err != nil {
				t.Fatalf("Forward failed: %v", err)
			}
			gotShape := out.Shape()
			for i := range gotShape {
				if gotShape[i] != tc.wantShape[i] {
					t.Fatalf("shape = %v, want %v", gotShape, tc.wantShape)
				}
			}
			got := out.Data()
			if len(got) != len(tc.want) {
				t.Fatalf("output length %d, want %d", len(got), len(tc.want))
			}
			for i := range got {
				if got[i] != tc.want[i] {
					t.Errorf("output[%d] = %v, want %v", i, got[i], tc.want[i])
				}
			}
		})
	}
}

func TestOr_WrongInputCount(t *testing.T) {
	eng := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	ops := numeric.Float32Ops{}
	node := &Or[float32]{engine: eng, ops: ops}

	a, _ := tensor.New[float32]([]int{3}, nil)
	_, err := node.Forward(context.Background(), a)
	if err == nil {
		t.Error("expected error for 1 input")
	}
}

func TestOr_SizeMismatch(t *testing.T) {
	eng := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	ops := numeric.Float32Ops{}
	node := &Or[float32]{engine: eng, ops: ops}

	a, _ := tensor.New[float32]([]int{3}, nil)
	b, _ := tensor.New[float32]([]int{4}, nil)
	_, err := node.Forward(context.Background(), a, b)
	if err == nil {
		t.Error("expected error for mismatched sizes")
	}
}

func TestOr_OpType(t *testing.T) {
	node := &Or[float32]{}
	if got := node.OpType(); got != "Or" {
		t.Errorf("OpType() = %q, want %q", got, "Or")
	}
}
