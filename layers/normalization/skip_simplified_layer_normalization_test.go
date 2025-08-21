// package normalization_test tests the normalization layers.
package normalization_test

import (
	"context"
	"testing"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/layers/normalization"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
	"github.com/zerfoo/zerfoo/testing/testutils"
)

// TestSkipSimplifiedLayerNormalization_Forward tests the forward pass of the SkipSimplifiedLayerNormalization layer.
func TestSkipSimplifiedLayerNormalization_Forward(t *testing.T) {
	ctx := context.Background()
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](&ops)

	// Input tensor
	inputData := []float32{1, 2, 3, 4}
	input, _ := tensor.New[float32]([]int{1, 4}, inputData)

	// Gain parameter
	gainData := []float32{1.5}
	gain, _ := tensor.New[float32]([]int{1}, gainData)

	// Epsilon attribute
	epsilon := float32(1e-5)

	// Create the layer
	layer, err := normalization.NewSkipSimplifiedLayerNormalization[float32](engine, &ops, gain, epsilon)
	testutils.AssertNoError(t, err, "NewSkipSimplifiedLayerNormalization failed")

	// Execute forward pass
	output, err := layer.Forward(ctx, input)
	testutils.AssertNoError(t, err, "Forward pass failed")

	// Manually calculate the expected output
	// 1. SimplifiedLayerNormalization output: [0.5477225, 1.095445, 1.643167, 2.19089]
	// 2. Add input: [1, 2, 3, 4] + [0.5477225, 1.095445, 1.643167, 2.19089]
	expectedOutputData := []float32{1.5477225, 3.095445, 4.643167, 6.19089}
	expectedOutput, _ := tensor.New[float32]([]int{1, 4}, expectedOutputData)

	if !testutils.CompareTensorsApprox(t, expectedOutput, output, 1e-5) {
		t.Errorf("Output tensor does not match expected. Got %v, want %v", output.Data(), expectedOutput.Data())
	}
}
