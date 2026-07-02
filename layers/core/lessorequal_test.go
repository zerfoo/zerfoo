package core

import (
	"context"
	"testing"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
)

func TestLessOrEqual_Forward(t *testing.T) {
	eng := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	ops := numeric.Float32Ops{}
	node := &LessOrEqual[float32]{engine: eng, ops: ops}

	tests := []struct {
		name   string
		aShape []int
		aData  []float32
		bShape []int
		bData  []float32
		want   []float32
	}{
		{
			name:   "equal values",
			aShape: []int{3}, aData: []float32{1, 2, 3},
			bShape: []int{3}, bData: []float32{1, 2, 3},
			want: []float32{1, 1, 1},
		},
		{
			name:   "less than",
			aShape: []int{3}, aData: []float32{1, 2, 3},
			bShape: []int{3}, bData: []float32{2, 3, 4},
			want: []float32{1, 1, 1},
		},
		{
			name:   "greater than",
			aShape: []int{3}, aData: []float32{3, 4, 5},
			bShape: []int{3}, bData: []float32{1, 2, 3},
			want: []float32{0, 0, 0},
		},
		{
			name:   "mixed",
			aShape: []int{4}, aData: []float32{1, 3, 2, 4},
			bShape: []int{4}, bData: []float32{2, 2, 2, 2},
			want: []float32{1, 0, 1, 0},
		},
		{
			name:   "scalar broadcast b",
			aShape: []int{3}, aData: []float32{1, 2, 3},
			bShape: []int{1}, bData: []float32{2},
			want: []float32{1, 1, 0},
		},
		{
			name:   "scalar broadcast a",
			aShape: []int{1}, aData: []float32{2},
			bShape: []int{3}, bData: []float32{1, 2, 3},
			want: []float32{0, 1, 1},
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

func TestLessOrEqual_WrongInputCount(t *testing.T) {
	eng := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	ops := numeric.Float32Ops{}
	node := &LessOrEqual[float32]{engine: eng, ops: ops}

	a, _ := tensor.New[float32]([]int{3}, nil)
	_, err := node.Forward(context.Background(), a)
	if err == nil {
		t.Error("expected error for 1 input")
	}
}

func TestLessOrEqual_SizeMismatch(t *testing.T) {
	eng := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	ops := numeric.Float32Ops{}
	node := &LessOrEqual[float32]{engine: eng, ops: ops}

	a, _ := tensor.New[float32]([]int{3}, nil)
	b, _ := tensor.New[float32]([]int{4}, nil)
	_, err := node.Forward(context.Background(), a, b)
	if err == nil {
		t.Error("expected error for mismatched sizes")
	}
}

func TestLessOrEqual_OpType(t *testing.T) {
	node := &LessOrEqual[float32]{}
	if got := node.OpType(); got != "LessOrEqual" {
		t.Errorf("OpType() = %q, want %q", got, "LessOrEqual")
	}
}
