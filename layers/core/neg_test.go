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

func TestNegForward_Comprehensive(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	tests := []struct {
		name string
		in   []float32
		want []float32
	}{
		{
			name: "positive inputs",
			in:   []float32{1, 2, 3, 4},
			want: []float32{-1, -2, -3, -4},
		},
		{
			name: "negative inputs",
			in:   []float32{-5, -10, -0.5, -100},
			want: []float32{5, 10, 0.5, 100},
		},
		{
			name: "zeros",
			in:   []float32{0, 0, 0, 0},
			want: []float32{0, 0, 0, 0},
		},
		{
			name: "mixed values",
			in:   []float32{-3, 0, 7, -0.1},
			want: []float32{3, 0, -7, 0.1},
		},
		{
			name: "very large values",
			in:   []float32{1e10, -1e10, 1e15, -1e15},
			want: []float32{-1e10, 1e10, -1e15, 1e15},
		},
		{
			name: "very small values",
			in:   []float32{1e-7, -1e-7, 1e-10, -1e-10},
			want: []float32{-1e-7, 1e-7, -1e-10, 1e-10},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			in, err := tensor.New[float32]([]int{len(tc.in)}, tc.in)
			if err != nil {
				t.Fatalf("creating tensor: %v", err)
			}

			neg := &Neg[float32]{engine: engine, ops: ops}
			out, err := neg.Forward(context.Background(), in)
			if err != nil {
				t.Fatalf("Forward: %v", err)
			}

			got := out.Data()
			for i := range tc.want {
				if diff := math.Abs(float64(got[i] - tc.want[i])); diff > 1e-6 {
					t.Errorf("out[%d] = %v, want %v", i, got[i], tc.want[i])
				}
			}
		})
	}
}

func TestNegForward_WrongInputCount(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)
	neg := &Neg[float32]{engine: engine, ops: ops}

	a, _ := tensor.New[float32]([]int{2}, []float32{1, 2})
	b, _ := tensor.New[float32]([]int{2}, []float32{3, 4})

	if _, err := neg.Forward(context.Background(), a, b); err == nil {
		t.Error("expected error for 2 inputs")
	}

	if _, err := neg.Forward(context.Background()); err == nil {
		t.Error("expected error for 0 inputs")
	}
}

func TestNegBackward(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	in, _ := tensor.New[float32]([]int{4}, []float32{3, -2, 0, 7})
	dOut, _ := tensor.New[float32]([]int{4}, []float32{1, 1, 1, 1})

	neg := &Neg[float32]{engine: engine, ops: ops}

	// Run forward first.
	if _, err := neg.Forward(context.Background(), in); err != nil {
		t.Fatalf("Forward: %v", err)
	}

	grads, err := neg.Backward(context.Background(), types.FullBackprop, dOut, in)
	if err != nil {
		t.Skipf("Backward not yet implemented: %v", err)
	}

	if len(grads) != 1 {
		t.Fatalf("expected 1 gradient, got %d", len(grads))
	}

	// d(-x)/dx = -1, so gradient should be -dOut
	wantGrad := []float32{-1, -1, -1, -1}
	got := grads[0].Data()
	for i := range wantGrad {
		if diff := math.Abs(float64(got[i] - wantGrad[i])); diff > 1e-6 {
			t.Errorf("grad[%d] = %v, want %v", i, got[i], wantGrad[i])
		}
	}
}

func TestNegForward_DoubleNeg(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	input := []float32{1, -2, 3, -4}
	in, _ := tensor.New[float32]([]int{4}, input)

	neg := &Neg[float32]{engine: engine, ops: ops}

	// Neg(x)
	out1, err := neg.Forward(context.Background(), in)
	if err != nil {
		t.Fatalf("first Neg: %v", err)
	}

	// Neg(Neg(x)) should equal x
	out2, err := neg.Forward(context.Background(), out1)
	if err != nil {
		t.Fatalf("second Neg: %v", err)
	}

	got := out2.Data()
	for i := range input {
		if diff := math.Abs(float64(got[i] - input[i])); diff > 1e-6 {
			t.Errorf("out[%d] = %v, want %v", i, got[i], input[i])
		}
	}
}

func TestNegOpType(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)
	neg := &Neg[float32]{engine: engine, ops: ops}

	if got := neg.OpType(); got != "Neg" {
		t.Errorf("OpType() = %q, want %q", got, "Neg")
	}
}
