package compute

import (
	"context"
	"math"
	"testing"

	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
)

func approxEqualFloats(a, b []float32, tol float32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if float32(math.Abs(float64(a[i]-b[i]))) > tol {
			return false
		}
	}
	return true
}

func TestCPUEngine_Softmax_NegativeAxisMeansLast(t *testing.T) {
	ctx := context.Background()
	engine := NewCPUEngine[float32](numeric.Float32Ops{})

	// 2x3, identical rows so expected softmax per-row is the same
	inp, _ := tensor.New[float32]([]int{2, 3}, []float32{1, 2, 3, 1, 2, 3})

	// Softmax along last axis explicitly
	outLast, err := engine.Softmax(ctx, inp, 1)
	if err != nil {
		t.Fatalf("Softmax last axis failed: %v", err)
	}

	// Softmax with axis = -1 should mean last axis
	outNeg, err := engine.Softmax(ctx, inp, -1)
	if err != nil {
		t.Fatalf("Softmax negative axis failed: %v", err)
	}

	if !approxEqualFloats(outLast.Data(), outNeg.Data(), 1e-5) {
		t.Fatalf("Softmax axis=-1 mismatch with last axis.\nlast=%v\nneg=%v", outLast.Data(), outNeg.Data())
	}

	// Additionally, verify each row sums to 1 for axis=-1 result
	sums, err := engine.Sum(ctx, outNeg, 1, false)
	if err != nil {
		t.Fatalf("Sum over softmax output failed: %v", err)
	}
	expectedOnes := []float32{1, 1}
	if !approxEqualFloats(sums.Data(), expectedOnes, 1e-5) {
		t.Fatalf("Row sums not 1: got %v", sums.Data())
	}
}
