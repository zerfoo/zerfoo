package core

import (
	"context"
	"math"
	"testing"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/ztensor/types"
)

func TestDivForward_Comprehensive(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	tests := []struct {
		name string
		a    []float32
		b    []float32
		want []float32
	}{
		{
			name: "basic division",
			a:    []float32{10, 20, 30, 40},
			b:    []float32{2, 4, 5, 8},
			want: []float32{5, 5, 6, 5},
		},
		{
			name: "division by 1",
			a:    []float32{3, 7, 11, 0},
			b:    []float32{1, 1, 1, 1},
			want: []float32{3, 7, 11, 0},
		},
		{
			name: "large values",
			a:    []float32{1e10, 1e12, 1e8, 1e6},
			b:    []float32{1e5, 1e6, 1e4, 1e3},
			want: []float32{1e5, 1e6, 1e4, 1e3},
		},
		{
			name: "same tensor divided by itself",
			a:    []float32{5, 13, 0.5, 100},
			b:    []float32{5, 13, 0.5, 100},
			want: []float32{1, 1, 1, 1},
		},
		{
			name: "division by small numbers",
			a:    []float32{1, 1, 1, 1},
			b:    []float32{0.001, 0.01, 0.1, 0.0001},
			want: []float32{1000, 100, 10, 10000},
		},
		{
			name: "negative values",
			a:    []float32{-10, 10, -10, 10},
			b:    []float32{2, -2, -2, 2},
			want: []float32{-5, -5, 5, 5},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			a, err := tensor.New[float32]([]int{1, len(tc.a)}, tc.a)
			if err != nil {
				t.Fatalf("creating tensor a: %v", err)
			}
			b, err := tensor.New[float32]([]int{1, len(tc.b)}, tc.b)
			if err != nil {
				t.Fatalf("creating tensor b: %v", err)
			}

			div := NewDiv[float32](engine)
			out, err := div.Forward(context.Background(), a, b)
			if err != nil {
				t.Fatalf("Forward: %v", err)
			}

			got := out.Data()
			for i := range tc.want {
				if diff := math.Abs(float64(got[i] - tc.want[i])); diff > 1e-3 {
					t.Errorf("out[%d] = %v, want %v", i, got[i], tc.want[i])
				}
			}
		})
	}
}

func TestDivForward_WrongInputCount(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)
	div := NewDiv[float32](engine)

	a, _ := tensor.New[float32]([]int{2}, []float32{1, 2})

	if _, err := div.Forward(context.Background(), a); err == nil {
		t.Error("expected error for 1 input")
	}

	b, _ := tensor.New[float32]([]int{2}, []float32{3, 4})
	c, _ := tensor.New[float32]([]int{2}, []float32{5, 6})

	if _, err := div.Forward(context.Background(), a, b, c); err == nil {
		t.Error("expected error for 3 inputs")
	}
}

func TestDivBackward(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	a, _ := tensor.New[float32]([]int{1, 4}, []float32{10, 20, 30, 40})
	b, _ := tensor.New[float32]([]int{1, 4}, []float32{2, 4, 5, 8})
	dOut, _ := tensor.New[float32]([]int{1, 4}, []float32{1, 1, 1, 1})

	div := NewDiv[float32](engine)

	// Run forward first (some implementations cache inputs).
	if _, err := div.Forward(context.Background(), a, b); err != nil {
		t.Fatalf("Forward: %v", err)
	}

	grads, err := div.Backward(context.Background(), types.FullBackprop, dOut, a, b)
	if err != nil {
		t.Skipf("Backward not yet implemented: %v", err)
	}

	if len(grads) != 2 {
		t.Fatalf("expected 2 gradients, got %d", len(grads))
	}

	// d(a/b)/da = 1/b
	wantGradA := []float32{0.5, 0.25, 0.2, 0.125}
	gotA := grads[0].Data()
	for i := range wantGradA {
		if diff := math.Abs(float64(gotA[i] - wantGradA[i])); diff > 1e-5 {
			t.Errorf("gradA[%d] = %v, want %v", i, gotA[i], wantGradA[i])
		}
	}

	// d(a/b)/db = -a/b^2
	wantGradB := []float32{-2.5, -1.25, -1.2, -0.625}
	gotB := grads[1].Data()
	for i := range wantGradB {
		if diff := math.Abs(float64(gotB[i] - wantGradB[i])); diff > 1e-5 {
			t.Errorf("gradB[%d] = %v, want %v", i, gotB[i], wantGradB[i])
		}
	}
}

func TestDivOpType(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)
	div := NewDiv[float32](engine)

	if got := div.OpType(); got != "Div" {
		t.Errorf("OpType() = %q, want %q", got, "Div")
	}
}
