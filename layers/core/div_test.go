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

func TestDivBackward(t *testing.T) {
	tests := []struct {
		name      string
		a         []float32
		b         []float32
		dOut      []float32
		wantGradA []float32
		wantGradB []float32
	}{
		{
			name:      "simple division",
			a:         []float32{6, 12, 9, 20},
			b:         []float32{2, 3, 3, 4},
			dOut:      []float32{1, 1, 1, 1},
			wantGradA: []float32{0.5, 1.0 / 3.0, 1.0 / 3.0, 0.25},
			wantGradB: []float32{-1.5, -12.0 / 9.0, -1, -20.0 / 16.0},
		},
		{
			name:      "scaled upstream gradient",
			a:         []float32{4, 9},
			b:         []float32{2, 3},
			dOut:      []float32{2, 3},
			wantGradA: []float32{1, 1},
			wantGradB: []float32{-2, -3},
		},
		{
			name:      "fractional values",
			a:         []float32{1, 1},
			b:         []float32{4, 2},
			dOut:      []float32{1, 1},
			wantGradA: []float32{0.25, 0.5},
			wantGradB: []float32{-1.0 / 16.0, -0.25},
		},
	}

	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			a, _ := tensor.New[float32]([]int{1, len(tc.a)}, tc.a)
			b, _ := tensor.New[float32]([]int{1, len(tc.b)}, tc.b)
			dOut, _ := tensor.New[float32]([]int{1, len(tc.dOut)}, tc.dOut)

			div := NewDiv[float32](engine)

			grads, err := div.Backward(context.Background(), types.FullBackprop, dOut, a, b)
			if err != nil {
				t.Fatalf("Backward failed: %v", err)
			}

			if len(grads) != 2 {
				t.Fatalf("Expected 2 gradients, got %d", len(grads))
			}

			testutils.AssertFloat32SliceApproxEqual(t, tc.wantGradA, grads[0].Data(), 1e-5, "gradA incorrect")
			testutils.AssertFloat32SliceApproxEqual(t, tc.wantGradB, grads[1].Data(), 1e-5, "gradB incorrect")
		})
	}
}

func TestDivBackwardNumericalGradient(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	aVals := []float32{5, 10, 3, 7}
	bVals := []float32{2, 4, 1.5, 3.5}

	a, _ := tensor.New[float32]([]int{1, 4}, aVals)
	b, _ := tensor.New[float32]([]int{1, 4}, bVals)
	dOut, _ := tensor.New[float32]([]int{1, 4}, []float32{1, 1, 1, 1})

	div := NewDiv[float32](engine)

	grads, err := div.Backward(context.Background(), types.FullBackprop, dOut, a, b)
	if err != nil {
		t.Fatalf("Backward failed: %v", err)
	}

	eps := float32(1e-4)
	for i := range aVals {
		expectedGradA := float64((aVals[i]+eps)/bVals[i]-(aVals[i]-eps)/bVals[i]) / float64(2*eps)
		if math.Abs(expectedGradA-float64(grads[0].Data()[i])) > 1e-2 {
			t.Errorf("gradA[%d]: analytical=%f, numerical=%f", i, grads[0].Data()[i], expectedGradA)
		}

		expectedGradB := float64(aVals[i]/(bVals[i]+eps)-aVals[i]/(bVals[i]-eps)) / float64(2*eps)
		if math.Abs(expectedGradB-float64(grads[1].Data()[i])) > 1e-2 {
			t.Errorf("gradB[%d]: analytical=%f, numerical=%f", i, grads[1].Data()[i], expectedGradB)
		}
	}
}

func TestDivBackwardWrongInputCount(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	dOut, _ := tensor.New[float32]([]int{1, 2}, []float32{1, 1})
	a, _ := tensor.New[float32]([]int{1, 2}, []float32{1, 2})

	div := NewDiv[float32](engine)

	_, err := div.Backward(context.Background(), types.FullBackprop, dOut, a)
	if err == nil {
		t.Fatal("Expected error with wrong input count")
	}
}
