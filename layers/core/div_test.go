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

func TestDivOpType(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)
	div := NewDiv[float32](engine)

	if got := div.OpType(); got != "Div" {
		t.Errorf("OpType() = %q, want %q", got, "Div")
	}
}
