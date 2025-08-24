// Package normalization_test tests the normalization layers.
//

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

// TestSimplifiedLayerNormalization_Forward tests the forward pass of the SimplifiedLayerNormalization layer.
//

func TestSimplifiedLayerNormalization_Forward(t *testing.T) {
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
	layer, err := normalization.NewSimplifiedLayerNormalization[float32](engine, &ops, gain, epsilon)
	testutils.AssertNoError(t, err, "NewSimplifiedLayerNormalization failed")

	// Execute forward pass
	output, err := layer.Forward(ctx, input)
	testutils.AssertNoError(t, err, "Forward pass failed")

	// Manually calculate the expected output
	// 1. Square the input: [1, 4, 9, 16]
	// 2. Mean of squares: (1+4+9+16)/4 = 30/4 = 7.5
	// 3. Add epsilon: 7.5 + 1e-5 = 7.50001
	// 4. Square root: sqrt(7.50001) = 2.738614
	// 5. Inverse square root: 1 / 2.738614 = 0.365148
	// 6. Normalize: input * inv_sqrt = [0.365148, 0.730296, 1.095444, 1.460592]
	// 7. Apply gain: result * 1.5 = [0.547722, 1.095444, 1.643166, 2.190888]
	expectedOutputData := []float32{0.5477225, 1.095445, 1.643167, 2.19089}
	expectedOutput, _ := tensor.New[float32]([]int{1, 4}, expectedOutputData)

	if !testutils.CompareTensorsApprox(t, expectedOutput, output, 1e-5) {
		t.Errorf("Output tensor does not match expected. Got %v, want %v", output.Data(), expectedOutput.Data())
	}
}
