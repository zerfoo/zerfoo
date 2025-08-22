package core

import (
	"context"
	"reflect"
	"testing"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
	"github.com/zerfoo/zerfoo/testing/testutils"
)

func TestFFN_Forward(t *testing.T) {
	ctx := context.Background()
	engine := compute.NewCPUEngine[float64](&numeric.Float64Ops{})
	ops := &numeric.Float64Ops{}

	inputDim := 4
	hiddenDim := 8
	outputDim := 4
	batchSize := 1

	// Create a mock FFN with predictable weights for testing
	ffn, err := NewFFN[float64]("test_ffn", engine, ops, inputDim, hiddenDim, outputDim)
	testutils.AssertNoError(t, err, "NewFFN should not return an error")
	testutils.AssertNotNil(t, ffn, "FFN should not be nil")

	// Manually set weights for W1, W2, and W3 (for SwiGLU)
	// W1: inputDim x hiddenDim
	// W2: hiddenDim x outputDim
	// W3: inputDim x hiddenDim (for gate)

	// W1 (inputDim x hiddenDim)
	w1Data := make([]float64, inputDim*hiddenDim)
	for i := range w1Data {
		w1Data[i] = float64(i+1) * 0.1
	}
	ffn.w1.linear.weights.Value = func() *tensor.TensorNumeric[float64] {
		t, _ := tensor.New[float64]([]int{inputDim, hiddenDim}, w1Data)

		return t
	}() // Assuming SetWeights takes *tensor.Tensor

	// W2 (hiddenDim x outputDim)
	w2Data := make([]float64, hiddenDim*outputDim)
	for i := range w2Data {
		w2Data[i] = float64(i+1) * 0.05
	}
	ffn.w2.linear.weights.Value = func() *tensor.TensorNumeric[float64] {
		t, _ := tensor.New[float64]([]int{hiddenDim, outputDim}, w2Data)

		return t
	}() // Assuming SetWeights takes *tensor.Tensor

	// W3 (inputDim x hiddenDim) - for the gate in SwiGLU
	w3Data := make([]float64, inputDim*hiddenDim)
	for i := range w3Data {
		w3Data[i] = float64(i+1) * 0.08
	}
	ffn.w3.linear.weights.Value = func() *tensor.TensorNumeric[float64] {
		t, _ := tensor.New[float64]([]int{inputDim, hiddenDim}, w3Data)

		return t
	}() // Assuming SetWeights takes *tensor.Tensor

	// Input tensor (batchSize, inputDim)
	inputData := []float64{1.0, 2.0, 3.0, 4.0}
	inputTensor, _ := tensor.New[float64]([]int{batchSize, inputDim}, inputData)

	// Expected output calculation (manual or from a known good implementation)
	// This will be complex due to SwiGLU, so we'll use a simplified example or a pre-calculated one.
	// For now, let's just ensure it runs without error and output shape is correct.

	output, err := ffn.Forward(ctx, inputTensor)
	testutils.AssertNoError(t, err, "Forward pass should not return an error")
	testutils.AssertNotNil(t, output, "Output tensor should not be nil")

	// Expected output shape: (batchSize, outputDim)
	expectedShape := []int{batchSize, outputDim}
	testutils.AssertTrue(t, reflect.DeepEqual(output.Shape(), expectedShape), "Output shape mismatch")

	expectedOutput := []float64{131.39999095, 139.59998895, 147.79998695, 155.99998495}
	tolerance := 1e-6
	for i, val := range output.Data() {
		if diff := val - expectedOutput[i]; diff < -tolerance || diff > tolerance {
			t.Errorf("Output value at index %d is incorrect. Got %f, expected %f", i, val, expectedOutput[i])
		}
	}
}

func TestFFN_Backward(t *testing.T) {
	ctx := context.Background()
	engine := compute.NewCPUEngine[float64](&numeric.Float64Ops{})
	ops := &numeric.Float64Ops{}

	inputDim := 4
	hiddenDim := 8
	outputDim := 4
	batchSize := 1

	ffn, err := NewFFN[float64]("test_ffn", engine, ops, inputDim, hiddenDim, outputDim)
	testutils.AssertNoError(t, err, "NewFFN should not return an error")

	// Manually set weights (same as forward for consistency)
	w1Data := make([]float64, inputDim*hiddenDim)
	for i := range w1Data {
		w1Data[i] = float64(i+1) * 0.1
	}
	ffn.w1.linear.weights.Value = func() *tensor.TensorNumeric[float64] {
		t, _ := tensor.New[float64]([]int{inputDim, hiddenDim}, w1Data)

		return t
	}()

	w2Data := make([]float64, hiddenDim*outputDim)
	for i := range w2Data {
		w2Data[i] = float64(i+1) * 0.05
	}
	ffn.w2.linear.weights.Value = func() *tensor.TensorNumeric[float64] {
		t, _ := tensor.New[float64]([]int{hiddenDim, outputDim}, w2Data)

		return t
	}()

	w3Data := make([]float64, inputDim*hiddenDim)
	for i := range w3Data {
		w3Data[i] = float64(i+1) * 0.08
	}
	ffn.w3.linear.weights.Value = func() *tensor.TensorNumeric[float64] {
		t, _ := tensor.New[float64]([]int{inputDim, hiddenDim}, w3Data)

		return t
	}()

	inputData := []float64{1.0, 2.0, 3.0, 4.0}
	inputTensor, _ := tensor.New[float64]([]int{batchSize, inputDim}, inputData)

	// Perform a forward pass to populate cached values for backward
	_, err = ffn.Forward(ctx, inputTensor)
	testutils.AssertNoError(t, err, "Forward pass for backward test should not return an error")

	// Gradient from subsequent layer (dOut)
	dOutData := make([]float64, batchSize*outputDim)
	for i := range dOutData {
		dOutData[i] = float64(i+1) * 0.01
	}
	dOutTensor, _ := tensor.New[float64]([]int{batchSize, outputDim}, dOutData)

	// Perform backward pass
	dInputs, err := ffn.Backward(ctx, dOutTensor)
	testutils.AssertNoError(t, err, "Backward pass should not return an error")
	testutils.AssertEqual(t, len(dInputs), 1, "Expected 1 input gradient")
	testutils.AssertNotNil(t, dInputs[0], "Input gradient should not be nil")

	expectedDInput := []float64{0.39000028, 0.93400106, 1.47800183, 2.02200261}
	tolerance := 1e-6
	for i, val := range dInputs[0].Data() {
		if diff := val - expectedDInput[i]; diff < -tolerance || diff > tolerance {
			t.Errorf("dInput value at index %d is incorrect. Got %f, expected %f", i, val, expectedDInput[i])
		}
	}

	// Check weight gradients
	dW1 := ffn.w1.linear.weights.Gradient.Data()
	expectedDW1 := []float64{0.01499998, 0.03499998, 0.05499999, 0.07499999, 0.095, 0.115, 0.135, 0.155, 0.02999996, 0.06999996, 0.10999997, 0.14999998, 0.18999999, 0.22999999, 0.27, 0.31, 0.04499994, 0.10499994, 0.16499996, 0.22499997, 0.28499999, 0.34499999, 0.405, 0.465, 0.05999993, 0.13999992, 0.21999994, 0.29999997, 0.37999998, 0.45999999, 0.53999999, 0.62}
	for i, val := range dW1 {
		if diff := val - expectedDW1[i]; diff < -tolerance || diff > tolerance {
			t.Errorf("dW1 value at index %d is incorrect. Got %f, expected %f", i, val, expectedDW1[i])
		}
	}

	dW2 := ffn.w2.linear.weights.Gradient.Data()
	expectedDW2 := []float64{0.16999979, 0.33999958, 0.50999937, 0.67999916, 0.1799999, 0.3599998, 0.5399997, 0.7199996, 0.18999995, 0.3799999, 0.56999986, 0.75999981, 0.19999998, 0.39999995, 0.59999993, 0.79999991, 0.20999999, 0.41999998, 0.62999997, 0.83999996, 0.22, 0.43999999, 0.65999999, 0.87999998, 0.23, 0.46, 0.68999999, 0.91999999, 0.24, 0.48, 0.72, 0.96}
	for i, val := range dW2 {
		if diff := val - expectedDW2[i]; diff < -tolerance || diff > tolerance {
			t.Errorf("dW2 value at index %d is incorrect. Got %f, expected %f", i, val, expectedDW2[i])
		}
	}

	dW3 := ffn.w3.linear.weights.Gradient.Data()
	expectedDW3 := []float64{3.16325461e-07, 3.51155541e-07, 2.61721830e-07, 1.68802724e-07, 1.00877790e-07, 5.74827610e-08, 3.16988226e-08, 1.70643160e-08, 6.32650921e-07, 7.02311082e-07, 5.23443660e-07, 3.37605448e-07, 2.01755580e-07, 1.14965522e-07, 6.33976452e-08, 3.41286320e-08, 9.48976382e-07, 1.05346662e-06, 7.85165489e-07, 5.06408172e-07, 3.02633371e-07, 1.72448283e-07, 9.50964677e-08, 5.11929480e-08, 1.26530184e-06, 1.40462216e-06, 1.04688732e-06, 6.75210896e-07, 4.03511161e-07, 2.29931044e-07, 1.26795290e-07, 6.82572641e-08}
	for i, val := range dW3 {
		if diff := val - expectedDW3[i]; diff < -tolerance || diff > tolerance {
			t.Errorf("dW3 value at index %d is incorrect. Got %f, expected %f", i, val, expectedDW3[i])
		}
	}
}

// TestFFN_WithInitializer tests FFN with custom weight initializer option.
func TestFFN_WithInitializer(t *testing.T) {
	engine := compute.NewCPUEngine[float32](&numeric.Float32Ops{})
	ops := &numeric.Float32Ops{}

	// Create custom initializer that sets all weights to 0.5
	customInit := &testutils.TestInitializer[float32]{Value: 0.5}

	ffn, err := NewFFN[float32]("test_ffn", engine, ops, 2, 4, 2, WithFFNInitializer[float32](customInit))
	testutils.AssertNoError(t, err, "NewFFN with custom initializer should not return an error")
	testutils.AssertNotNil(t, ffn, "FFN should not be nil")

	// Check that all weights are initialized to 0.5
	w1Weights := ffn.w1.linear.weights.Value.Data()
	for _, val := range w1Weights {
		testutils.AssertFloatEqual(t, float32(0.5), val, float32(1e-6), "W1 weight should be 0.5")
	}

	w2Weights := ffn.w2.linear.weights.Value.Data()
	for _, val := range w2Weights {
		testutils.AssertFloatEqual(t, float32(0.5), val, float32(1e-6), "W2 weight should be 0.5")
	}

	w3Weights := ffn.w3.linear.weights.Value.Data()
	for _, val := range w3Weights {
		testutils.AssertFloatEqual(t, float32(0.5), val, float32(1e-6), "W3 weight should be 0.5")
	}
}

// TestFFN_WithBias tests FFN with bias option.
func TestFFN_WithBias(t *testing.T) {
	engine := compute.NewCPUEngine[float32](&numeric.Float32Ops{})
	ops := &numeric.Float32Ops{}

	// Test with bias disabled
	ffnNoBias, err := NewFFN[float32]("test_ffn_no_bias", engine, ops, 2, 4, 2, WithFFNBias[float32](false))
	testutils.AssertNoError(t, err, "NewFFN with no bias should not return an error")
	testutils.AssertNotNil(t, ffnNoBias, "FFN should not be nil")

	// Test with bias enabled (default)
	ffnWithBias, err := NewFFN[float32]("test_ffn_with_bias", engine, ops, 2, 4, 2, WithFFNBias[float32](true))
	testutils.AssertNoError(t, err, "NewFFN with bias should not return an error")
	testutils.AssertNotNil(t, ffnWithBias, "FFN should not be nil")

	// Check parameter counts (weights + bias for each layer)
	noBiasParams := ffnNoBias.Parameters()
	withBiasParams := ffnWithBias.Parameters()

	// With bias: 6 parameters (W1 weights, W1 bias, W2 weights, W2 bias, W3 weights, W3 bias)
	// Without bias: 3 parameters (W1 weights, W2 weights, W3 weights)
	testutils.AssertEqual(t, len(noBiasParams), 3, "FFN without bias should have 3 parameter tensors")
	testutils.AssertEqual(t, len(withBiasParams), 6, "FFN with bias should have 6 parameter tensors")
}
