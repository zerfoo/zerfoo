// package activations_test tests the activation layers.
package activations_test

import (
	"context"
	"math"
	"testing"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/layers/activations"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
	"github.com/zerfoo/zerfoo/testing/testutils"
)

// TestFastGelu_Forward tests the forward pass of the FastGelu layer.
func TestFastGelu_Forward(t *testing.T) {
	ctx := context.Background()
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](&ops)

	// Input tensor
	inputData := []float32{-2, -1, 0, 1, 2}
	input, _ := tensor.New[float32]([]int{1, 5}, inputData)

	// Create the layer
	layer := activations.NewFastGelu[float32](engine)

	// Execute forward pass
	output, err := layer.Forward(ctx, input)
	testutils.AssertNoError(t, err, "Forward pass failed")

	// Manually calculate the expected output using the fast GELU approximation
	// y = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
	expectedOutputData := make([]float32, len(inputData))
	for i, x := range inputData {
		x3 := x * x * x
		inner := float32(math.Sqrt(2/math.Pi)) * (x + 0.044715*x3)
		expectedOutputData[i] = 0.5 * x * (1 + float32(math.Tanh(float64(inner))))
	}
	expectedOutput, _ := tensor.New[float32]([]int{1, 5}, expectedOutputData)

	if !testutils.CompareTensorsApprox(t, expectedOutput, output, 1e-5) {
		t.Errorf("Output tensor does not match expected. Got %v, want %v", output.Data(), expectedOutput.Data())
	}
}
