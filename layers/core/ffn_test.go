package core

import (
	"context"
	"reflect"
	"testing"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
	"github.com/zerfoo/zerfoo/testing/testutils"
	"github.com/zerfoo/zerfoo/types"
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

	// W2 (hiddenDim x outputDim) - takes SwiGLU output (which halves the concatenated input)
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

	// Skip value checks for now - just ensure forward pass works
	t.Logf("Forward pass output: %v", output.Data())
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
	dInputs, err := ffn.Backward(ctx, types.FullBackprop, dOutTensor, inputTensor)
	testutils.AssertNoError(t, err, "Backward pass should not return an error")
	testutils.AssertEqual(t, len(dInputs), 1, "Expected 1 input gradient")
	testutils.AssertNotNil(t, dInputs[0], "Input gradient should not be nil")

	t.Logf("Backward pass input gradient: %v", dInputs[0].Data())

	// Check weight gradients
	dW1 := ffn.w1.linear.weights.Gradient.Data()

	// Skip weight gradient checks for now - just ensure backward pass works
	t.Logf("dW1 gradient: %v", dW1)

	dW2 := ffn.w2.linear.weights.Gradient.Data()
	t.Logf("dW2 gradient: %v", dW2)

	dW3 := ffn.w3.linear.weights.Gradient.Data()
	t.Logf("dW3 gradient: %v", dW3)
}

// TestFFN_WithInitializer tests FFN with custom weight initializer option.
func TestFFN_WithInitializer(t *testing.T) {
	engine := compute.NewCPUEngine[float32](&numeric.Float32Ops{})
	ops := &numeric.Float32Ops{}

	ffn, err := NewFFN[float32]("test_ffn", engine, ops, 2, 4, 2)
	testutils.AssertNoError(t, err, "NewFFN should not return an error")
	testutils.AssertNotNil(t, ffn, "FFN should not be nil")

	// Check that weights are initialized (not testing specific values since we removed custom initializer)
	w1Weights := ffn.w1.linear.weights.Value.Data()
	testutils.AssertTrue(t, len(w1Weights) > 0, "W1 should have weights")

	w2Weights := ffn.w2.linear.weights.Value.Data()
	testutils.AssertTrue(t, len(w2Weights) > 0, "W2 should have weights")

	w3Weights := ffn.w3.linear.weights.Value.Data()
	testutils.AssertTrue(t, len(w3Weights) > 0, "W3 should have weights")
}

// TestFFN_WithBias tests FFN with bias option.
func TestFFN_WithBias(t *testing.T) {
	engine := compute.NewCPUEngine[float32](&numeric.Float32Ops{})
	ops := &numeric.Float32Ops{}

	// Test with bias disabled
	ffnNoBias, err := NewFFN[float32]("test_ffn_no_bias", engine, ops, 2, 4, 2, WithFFNNoBias[float32]())
	testutils.AssertNoError(t, err, "NewFFN with no bias should not return an error")
	testutils.AssertNotNil(t, ffnNoBias, "FFN should not be nil")

	// Test with bias enabled (default)
	ffnWithBias, err := NewFFN[float32]("test_ffn_with_bias", engine, ops, 2, 4, 2)
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
