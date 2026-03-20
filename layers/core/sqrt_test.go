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
