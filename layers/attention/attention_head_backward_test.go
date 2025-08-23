package attention

import (
	"context"
	"testing"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
	"github.com/zerfoo/zerfoo/testing/testutils"
)

func TestAttentionHead_Backward_GradientShapeAndFlow(t *testing.T) {
	engine := compute.NewCPUEngine(numeric.Float32Ops{})

	batchSize := 2
	seqLen := 5
	inputDim := 10
	headDim := 8

	ah := NewAttentionHead[float32](engine, inputDim, headDim)

	// Input
	inp, err := tensor.New[float32]([]int{batchSize, seqLen, inputDim}, nil)
	testutils.AssertNoError(t, err, "create input")
	// Fill deterministic values
	for i := range inp.Data() {
		inp.Data()[i] = float32((i%7)+1) / 10.0
	}

	// Forward to populate caches for SDPA
	out, err := ah.Forward(context.Background(), inp)
	testutils.AssertNoError(t, err, "forward")

	// Upstream gradient same shape as out
	dOut, err := tensor.New[float32](out.Shape(), nil)
	testutils.AssertNoError(t, err, "create dOut")
	for i := range dOut.Data() {
		dOut.Data()[i] = 1.0 // simple all-ones upstream grad
	}

	grads, err := ah.Backward(context.Background(), dOut, inp)
	testutils.AssertNoError(t, err, "backward")
	testutils.AssertEqual(t, 1, len(grads), "grads len")

	gIn := grads[0]
	testutils.AssertTrue(t, testutils.IntSliceEqual(inp.Shape(), gIn.Shape()), "grad shape")

	// Sanity: gradient should be finite and not all zeros
	var sum float32
	for _, v := range gIn.Data() {
		// NaN check
		if !(v == v) {
			t.Fatalf("gradient contains NaN")
		}
		sum += v
	}
	// Not a strict check, just ensure some signal propagated
	testutils.AssertTrue(t, sum != 0, "expected non-zero gradient sum")
}
