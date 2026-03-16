package core

import (
	"context"
	"math"
	"testing"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
)

func TestMod_Forward(t *testing.T) {
	eng := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	node := &Mod[float32]{engine: eng}

	tests := []struct {
		name   string
		aShape []int
		aData  []float32
		bShape []int
		bData  []float32
		want   []float32
	}{
		{
			name:   "basic",
			aShape: []int{4}, aData: []float32{7, 8, 9, 10},
			bShape: []int{4}, bData: []float32{3, 3, 3, 3},
			want: []float32{1, 2, 0, 1},
		},
		{
			name:   "scalar broadcast b",
			aShape: []int{3}, aData: []float32{5, 7, 10},
			bShape: []int{1}, bData: []float32{3},
			want: []float32{2, 1, 1},
		},
		{
			name:   "scalar broadcast a",
			aShape: []int{1}, aData: []float32{10},
			bShape: []int{3}, bData: []float32{3, 4, 7},
			want: []float32{1, 2, 3},
		},
		{
			name:   "float mod",
			aShape: []int{2}, aData: []float32{5.5, 7.2},
			bShape: []int{2}, bData: []float32{2.5, 3.0},
			want: []float32{float32(math.Mod(5.5, 2.5)), float32(math.Mod(7.2, 3.0))},
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
				t.Fatalf("length = %d, want %d", len(got), len(tc.want))
			}
			for i := range got {
				if math.Abs(float64(got[i]-tc.want[i])) > 1e-5 {
					t.Errorf("[%d] = %v, want %v", i, got[i], tc.want[i])
				}
			}
		})
	}
}

func TestMod_WrongInputCount(t *testing.T) {
	eng := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	node := &Mod[float32]{engine: eng}
	a, _ := tensor.New[float32]([]int{3}, nil)
	_, err := node.Forward(context.Background(), a)
	if err == nil {
		t.Error("expected error for 1 input")
	}
}

func TestMod_SizeMismatch(t *testing.T) {
	eng := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	node := &Mod[float32]{engine: eng}
	a, _ := tensor.New[float32]([]int{3}, nil)
	b, _ := tensor.New[float32]([]int{4}, nil)
	_, err := node.Forward(context.Background(), a, b)
	if err == nil {
		t.Error("expected error for mismatched sizes")
	}
}

func TestMod_OpType(t *testing.T) {
	node := &Mod[float32]{}
	if got := node.OpType(); got != "Mod" {
		t.Errorf("OpType() = %q, want %q", got, "Mod")
	}
}
