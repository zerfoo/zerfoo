package attention

import (
	"context"
	"testing"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
	"github.com/zerfoo/zerfoo/testing/testutils"
	"github.com/zerfoo/zerfoo/types"
)

func TestNewScaledDotProductAttention_FunctionalOptions(t *testing.T) {
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	headDim := 4

	// Test with functional options (even if no specific options are defined yet)
	sdpa := NewScaledDotProductAttention(
		engine,
		headDim,
		// No specific options to pass yet, but demonstrating the pattern
	)

	testutils.AssertNotNil(t, sdpa, "expected ScaledDotProductAttention to not be nil")

	// Test forward pass (simplified)
	batchSize := 1
	seqLenQ := 3
	seqLenK := 3

	qData := []float32{1, 0, 0, 0, 1, 0, 0, 0, 1, 2, 3, 4}
	kData := []float32{1, 0, 0, 0, 1, 0, 0, 0, 1, 5, 6, 7}
	vData := []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 8, 7, 6}

	q, err := tensor.New[float32]([]int{batchSize, seqLenQ, headDim}, qData)
	testutils.AssertNoError(t, err, "failed to create Q tensor")
	k, err := tensor.New[float32]([]int{batchSize, seqLenK, headDim}, kData)
	testutils.AssertNoError(t, err, "failed to create K tensor")
	v, err := tensor.New[float32]([]int{batchSize, seqLenK, headDim}, vData)
	testutils.AssertNoError(t, err, "failed to create V tensor")

	output, err := sdpa.Forward(context.Background(), q, k, v, nil)
	testutils.AssertNoError(t, err, "forward pass failed")
	testutils.AssertNotNil(t, output, "expected output to not be nil")

	expectedOutputShape := []int{batchSize, seqLenQ, headDim}
	testutils.AssertTrue(t, testutils.IntSliceEqual(expectedOutputShape, output.Shape()), "output shape mismatch")

	// Test backward pass (simplified check)
	outputGradData := make([]float32, batchSize*seqLenQ*headDim)
	for i := range outputGradData {
		outputGradData[i] = 1.0
	}

	outputGrad, err := tensor.New[float32](expectedOutputShape, outputGradData)
	testutils.AssertNoError(t, err, "failed to create output gradient tensor")

	inputGrads, err := sdpa.Backward(context.Background(), types.FullBackprop, outputGrad, q, k, v)
	testutils.AssertNoError(t, err, "backward pass failed")
	testutils.AssertTrue(t, len(inputGrads) == 3, "expected 3 input gradients")
	testutils.AssertNotNil(t, inputGrads[0], "expected non-nil dQ")
	testutils.AssertNotNil(t, inputGrads[1], "expected non-nil dK")
	testutils.AssertNotNil(t, inputGrads[2], "expected non-nil dV")
}
