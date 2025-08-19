package core

import (
	"context"
	"testing"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
	"github.com/zerfoo/zerfoo/testing/testutils"
)

func TestDense(t *testing.T) {
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	ops := numeric.Float32Ops{}
	layer, err := NewDense[float32]("dense", engine, ops, 10, 5)
	testutils.AssertNoError(t, err, "expected no error when creating dense layer, got %v")

	inputData := make([]float32, 10)
	for i := range inputData {
		inputData[i] = float32(i)
	}
	input, err := tensor.New[float32]([]int{1, 10}, inputData)
	testutils.AssertNoError(t, err, "expected no error when creating input tensor, got %v")

	// Check forward pass
	output, _ := layer.Forward(context.Background(), input)
	testutils.AssertNotNil(t, output, "expected output to not be nil")

	// Check backward pass
	gradOutput, err := tensor.New[float32]([]int{1, 5}, []float32{1, 1, 1, 1, 1})
	testutils.AssertNoError(t, err, "expected no error when creating gradient output tensor, got %v")
	gradInput, _ := layer.Backward(context.Background(), gradOutput)
	testutils.AssertNotNil(t, gradInput, "expected gradient input to not be nil")

	// Test the SetName method of the dense layer
	layer.SetName("new_dense")
	testutils.AssertEqual(t, "new_dense_weights", layer.linear.weights.Name, "expected weights name %q, got %q")
	testutils.AssertEqual(t, "new_dense_biases", layer.bias.biases.Name, "expected biases name %q, got %q")

	// Test the OutputShape method of the dense layer
	testutils.AssertTrue(t, testutils.IntSliceEqual([]int{1, 5}, layer.OutputShape()), "expected output shape to be equal")

	// Test the Parameters method of the dense layer
	testutils.AssertEqual(t, 2, len(layer.Parameters()), "expected 2 parameters, got %d")
}
