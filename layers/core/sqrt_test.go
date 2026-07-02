package core

import (
	"context"
	"math"
	"testing"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/ztensor/testing/testutils"
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
	tests := []struct {
		name      string
		a         []float32
		dOut      []float32
		wantGradA []float32
	}{
		{
			name:      "perfect squares",
			a:         []float32{1, 4, 9, 16},
			dOut:      []float32{1, 1, 1, 1},
			wantGradA: []float32{0.5, 0.25, 1.0 / 6.0, 0.125},
		},
		{
			name:      "scaled upstream gradient",
			a:         []float32{4, 9},
			dOut:      []float32{2, 3},
			wantGradA: []float32{0.5, 0.5},
		},
		{
			name: "non-perfect squares",
			a:    []float32{2, 3, 5, 7},
			dOut: []float32{1, 1, 1, 1},
			wantGradA: []float32{
				float32(0.5 / math.Sqrt(2)),
				float32(0.5 / math.Sqrt(3)),
				float32(0.5 / math.Sqrt(5)),
				float32(0.5 / math.Sqrt(7)),
			},
		},
	}

	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			a, _ := tensor.New[float32]([]int{1, len(tc.a)}, tc.a)
			dOut, _ := tensor.New[float32]([]int{1, len(tc.dOut)}, tc.dOut)

			sqrt := NewSqrt[float32](engine)

			grads, err := sqrt.Backward(context.Background(), types.FullBackprop, dOut, a)
			if err != nil {
				t.Fatalf("Backward failed: %v", err)
			}

			if len(grads) != 1 {
				t.Fatalf("Expected 1 gradient, got %d", len(grads))
			}

			testutils.AssertFloat32SliceApproxEqual(t, tc.wantGradA, grads[0].Data(), 1e-5, "gradA incorrect")
		})
	}
}

func TestSqrtBackwardNumericalGradient(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	aVals := []float32{2, 5, 10, 25}

	a, _ := tensor.New[float32]([]int{1, 4}, aVals)
	dOut, _ := tensor.New[float32]([]int{1, 4}, []float32{1, 1, 1, 1})

	sqrt := NewSqrt[float32](engine)

	grads, err := sqrt.Backward(context.Background(), types.FullBackprop, dOut, a)
	if err != nil {
		t.Fatalf("Backward failed: %v", err)
	}

	eps := float64(1e-4)
	for i := range aVals {
		fPlus := math.Sqrt(float64(aVals[i]) + eps)
		fMinus := math.Sqrt(float64(aVals[i]) - eps)
		expectedGrad := (fPlus - fMinus) / (2 * eps)
		if math.Abs(expectedGrad-float64(grads[0].Data()[i])) > 1e-3 {
			t.Errorf("gradA[%d]: analytical=%f, numerical=%f", i, grads[0].Data()[i], expectedGrad)
		}
	}
}

func TestSqrtBackwardWrongInputCount(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	dOut, _ := tensor.New[float32]([]int{1, 2}, []float32{1, 1})
	a, _ := tensor.New[float32]([]int{1, 2}, []float32{1, 2})
	b, _ := tensor.New[float32]([]int{1, 2}, []float32{3, 4})

	sqrt := NewSqrt[float32](engine)

	_, err := sqrt.Backward(context.Background(), types.FullBackprop, dOut, a, b)
	if err == nil {
		t.Fatal("Expected error with wrong input count")
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
