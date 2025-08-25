package activations

import (
	"context"
	"testing"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
	"github.com/zerfoo/zerfoo/testing/testutils"
	"github.com/zerfoo/zerfoo/types"
)

// MockEngine for testing purposes.
type MockEngine[T tensor.Numeric] struct {
	compute.CPUEngine[T]
}

// MockOps for testing purposes.
type MockOps[T tensor.Numeric] struct {
	numeric.Float32Ops
}

func TestNewBaseActivation_FunctionalOptions(t *testing.T) {
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	ops := numeric.Float32Ops{}

	// Define a custom forward operation using a functional option
	customForwardOp := func(val float32) float32 {
		return val * 2 // Example: double the value
	}
	customBackwardOp := func(_ float32) float32 {
		return 1.0 // Example: constant gradient
	}

	// Test with functional options
	activation := NewBaseActivation(
		engine,
		ops,
		"mock_activation",
		WithForwardOp(customForwardOp),
		WithBackwardOp(customBackwardOp),
	)

	testutils.AssertNotNil(t, activation, "expected activation to not be nil")
	// Further assertions can be added here to verify the applied options
	// For example, by calling Forward and checking the output
	input, err := tensor.New[float32]([]int{1, 2}, []float32{1.0, 2.0})
	testutils.AssertNoError(t, err, "failed to create input tensor")

	output, err := activation.Forward(context.Background(), input)
	testutils.AssertNoError(t, err, "forward pass failed")
	testutils.AssertNotNil(t, output, "expected output to not be nil")

	expectedOutputData := []float32{2.0, 4.0} // 1*2, 2*2
	testutils.AssertFloat32SliceApproxEqual(t, expectedOutputData, output.Data(), 1e-6, "forward output mismatch")

	// Test backward pass
	outputGrad, err := tensor.New[float32]([]int{1, 2}, []float32{1.0, 1.0})
	testutils.AssertNoError(t, err, "failed to create output gradient tensor")

	inputGrads, err := activation.Backward(context.Background(), types.FullBackprop, outputGrad, input)
	testutils.AssertNoError(t, err, "backward pass failed")
	testutils.AssertTrue(t, len(inputGrads) == 1, "expected 1 input gradient")
	testutils.AssertNotNil(t, inputGrads[0], "expected non-nil input gradient")

	expectedInputGradData := []float32{1.0, 1.0} // 1.0 * 1.0, 1.0 * 1.0 (since backwardOp returns 1.0)
	testutils.AssertFloat32SliceApproxEqual(t, expectedInputGradData, inputGrads[0].Data(), 1e-6, "backward input gradient mismatch")
}
