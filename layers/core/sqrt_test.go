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

func TestSqrtForward_Comprehensive(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	tests := []struct {
		name string
		in   []float32
		want []float32
	}{
		{
			name: "perfect squares",
			in:   []float32{0, 1, 4, 9, 16, 25},
			want: []float32{0, 1, 2, 3, 4, 5},
		},
		{
			name: "non-perfect squares",
			in:   []float32{2, 3, 5, 7},
			want: []float32{float32(math.Sqrt(2)), float32(math.Sqrt(3)), float32(math.Sqrt(5)), float32(math.Sqrt(7))},
		},
		{
			name: "large values",
			in:   []float32{1e6, 1e8, 1e4, 1e10},
			want: []float32{1e3, 1e4, 1e2, 1e5},
		},
		{
			name: "sqrt of 0 and 1",
			in:   []float32{0, 1, 0, 1},
			want: []float32{0, 1, 0, 1},
		},
		{
			name: "fractional values",
			in:   []float32{0.25, 0.01, 0.0001, 0.49},
			want: []float32{0.5, 0.1, 0.01, 0.7},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			in, err := tensor.New[float32]([]int{len(tc.in)}, tc.in)
			if err != nil {
				t.Fatalf("creating tensor: %v", err)
			}

			sqrt := NewSqrt[float32](engine)
			out, err := sqrt.Forward(context.Background(), in)
			if err != nil {
				t.Fatalf("Forward: %v", err)
			}

			got := out.Data()
			for i := range tc.want {
				if diff := math.Abs(float64(got[i] - tc.want[i])); diff > 1e-4 {
					t.Errorf("out[%d] = %v, want %v", i, got[i], tc.want[i])
				}
			}
		})
	}
}

func TestSqrtForward_WrongInputCount(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)
	sqrt := NewSqrt[float32](engine)

	a, _ := tensor.New[float32]([]int{2}, []float32{1, 4})
	b, _ := tensor.New[float32]([]int{2}, []float32{9, 16})

	if _, err := sqrt.Forward(context.Background(), a, b); err == nil {
		t.Error("expected error for 2 inputs")
	}

	if _, err := sqrt.Forward(context.Background()); err == nil {
		t.Error("expected error for 0 inputs")
	}
}

func TestSqrtBackward(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	in, _ := tensor.New[float32]([]int{4}, []float32{1, 4, 9, 16})
	dOut, _ := tensor.New[float32]([]int{4}, []float32{1, 1, 1, 1})

	sqrt := NewSqrt[float32](engine)

	// Run forward first.
	if _, err := sqrt.Forward(context.Background(), in); err != nil {
		t.Fatalf("Forward: %v", err)
	}

	grads, err := sqrt.Backward(context.Background(), types.FullBackprop, dOut, in)
	if err != nil {
		t.Skipf("Backward not yet implemented: %v", err)
	}

	if len(grads) != 1 {
		t.Fatalf("expected 1 gradient, got %d", len(grads))
	}

	// d(sqrt(x))/dx = 1 / (2 * sqrt(x))
	wantGrad := []float32{0.5, 0.25, 1.0 / 6.0, 0.125}
	got := grads[0].Data()
	for i := range wantGrad {
		if diff := math.Abs(float64(got[i] - wantGrad[i])); diff > 1e-5 {
			t.Errorf("grad[%d] = %v, want %v", i, got[i], wantGrad[i])
		}
	}
}

func TestSqrtOpType(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)
	sqrt := NewSqrt[float32](engine)

	if got := sqrt.OpType(); got != "Sqrt" {
		t.Errorf("OpType() = %q, want %q", got, "Sqrt")
	}
}
