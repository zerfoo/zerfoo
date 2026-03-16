package activations

import (
	"context"
	"testing"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/ztensor/testing/testutils"
	"github.com/zerfoo/ztensor/types"
)

func TestNewSwiGLU_FunctionalOptions(t *testing.T) {
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	ops := numeric.Float32Ops{}

	// Test with functional options (even if no specific options are defined yet)
	swiglu := NewSwiGLU(
		engine,
		ops,
		// No specific options to pass yet, but demonstrating the pattern
	)

	testutils.AssertNotNil(t, swiglu, "expected SwiGLU to not be nil")

	// Test forward pass
	inputData := []float32{1.0, 2.0, 3.0, 4.0}
	input, err := tensor.New[float32]([]int{1, 4}, inputData)
	testutils.AssertNoError(t, err, "failed to create input tensor")

	output, err := swiglu.Forward(context.Background(), input)
	testutils.AssertNoError(t, err, "forward pass failed")
	testutils.AssertNotNil(t, output, "expected output to not be nil")

	// Expected output for SwiGLU(x1, x2) = silu(x1) * x2
	// x1 = [1.0, 2.0], x2 = [3.0, 4.0]
	// silu(x) = x * sigmoid(x)
	// silu(1.0) = 1.0 * sigmoid(1.0) = 1.0 * 0.7310586 = 0.7310586
	// silu(2.0) = 2.0 * sigmoid(2.0) = 2.0 * 0.8807970 = 1.7615941
	// output = [0.7310586 * 3.0, 1.7615941 * 4.0] = [2.1931758, 7.0463762]
	expectedOutputData := []float32{2.1931758, 7.0463762}
	testutils.AssertFloat32SliceApproxEqual(t, expectedOutputData, output.Data(), 1e-6, "forward output mismatch")

	// Test backward pass (simplified check)
	outputGradData := []float32{1.0, 1.0}
	outputGrad, err := tensor.New[float32]([]int{1, 2}, outputGradData)
	testutils.AssertNoError(t, err, "failed to create output gradient tensor")

	inputGrads, err := swiglu.Backward(context.Background(), types.FullBackprop, outputGrad, input)
	testutils.AssertNoError(t, err, "backward pass failed")
	testutils.AssertTrue(t, len(inputGrads) == 1, "expected 1 input gradient")
	testutils.AssertNotNil(t, inputGrads[0], "expected non-nil input gradient")

	// The exact backward gradient calculation is complex, so we'll just check for non-nil for now.
	// A more thorough test would involve numerical gradient checking.
}
