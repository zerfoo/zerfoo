package core

import (
	"context"
	"testing"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
)

func TestSqueeze_Forward(t *testing.T) {
	eng := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	tests := []struct {
		name      string
		axes      []int
		inShape   []int
		wantShape []int
	}{
		{
			name:      "squeeze all",
			axes:      nil,
			inShape:   []int{1, 3, 1, 4},
			wantShape: []int{3, 4},
		},
		{
			name:      "squeeze axis 0",
			axes:      []int{0},
			inShape:   []int{1, 3, 4},
			wantShape: []int{3, 4},
		},
		{
			name:      "squeeze axis -1",
			axes:      []int{-1},
			inShape:   []int{3, 4, 1},
			wantShape: []int{3, 4},
		},
		{
			name:      "squeeze multiple axes",
			axes:      []int{0, 2},
			inShape:   []int{1, 3, 1, 4},
			wantShape: []int{3, 4},
		},
		{
			name:      "squeeze to scalar",
			axes:      []int{0},
			inShape:   []int{1},
			wantShape: nil, // scalar (0D tensor)
		},
		{
			name:      "no size-1 dims to squeeze",
			axes:      nil,
			inShape:   []int{3, 4},
			wantShape: []int{3, 4},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			n := 1
			for _, d := range tc.inShape {
				n *= d
			}
			data := make([]float32, n)
			for i := range data {
				data[i] = float32(i)
			}
			in, _ := tensor.New[float32](tc.inShape, data)
			node := &Squeeze[float32]{engine: eng, axes: tc.axes}
			out, err := node.Forward(context.Background(), in)
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
			// Data should be preserved.
			if len(out.Data()) != n {
				t.Errorf("data length = %d, want %d", len(out.Data()), n)
			}
		})
	}
}

func TestSqueeze_AxesFromInput(t *testing.T) {
	eng := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	node := &Squeeze[float32]{engine: eng}

	in, _ := tensor.New[float32]([]int{1, 3, 1, 4}, make([]float32, 12))
	axes, _ := tensor.New[float32]([]int{2}, []float32{0, 2})

	out, err := node.Forward(context.Background(), in, axes)
	if err != nil {
		t.Fatalf("Forward failed: %v", err)
	}
	gotShape := out.Shape()
	wantShape := []int{3, 4}
	if len(gotShape) != len(wantShape) {
		t.Fatalf("shape = %v, want %v", gotShape, wantShape)
	}
	for i := range gotShape {
		if gotShape[i] != wantShape[i] {
			t.Errorf("shape[%d] = %d, want %d", i, gotShape[i], wantShape[i])
		}
	}
}

func TestSqueeze_NonOneDimError(t *testing.T) {
	eng := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	node := &Squeeze[float32]{engine: eng, axes: []int{1}}

	in, _ := tensor.New[float32]([]int{1, 3, 4}, make([]float32, 12))
	_, err := node.Forward(context.Background(), in)
	if err == nil {
		t.Error("expected error for squeezing non-1 dim")
	}
}

func TestSqueeze_OpType(t *testing.T) {
	node := &Squeeze[float32]{}
	if got := node.OpType(); got != "Squeeze" {
		t.Errorf("OpType() = %q, want %q", got, "Squeeze")
	}
}
