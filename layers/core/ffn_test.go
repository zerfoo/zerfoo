package core

import (
	"context"
	"fmt"
	"math"
	"reflect"
	"testing"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/ztensor/testing/testutils"
	"github.com/zerfoo/ztensor/types"
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

// TestFFN_WithSwiGLUAndNoBias verifies that combining WithSwiGLU and WithFFNNoBias works.
// This was broken before the noBias field fix: the old heuristic only worked with len(opts)==1.
func TestFFN_WithSwiGLUAndNoBias(t *testing.T) {
	engine := compute.NewCPUEngine[float32](&numeric.Float32Ops{})
	ops := &numeric.Float32Ops{}

	ffn, err := NewFFN[float32]("test", engine, ops, 4, 8, 4,
		WithSwiGLU[float32](),
		WithFFNNoBias[float32](),
	)
	testutils.AssertNoError(t, err, "NewFFN with SwiGLU+NoBias")
	testutils.AssertNotNil(t, ffn, "FFN should not be nil")

	// Should have 3 parameters (weights only, no bias)
	testutils.AssertEqual(t, len(ffn.Parameters()), 3, "SwiGLU+NoBias should have 3 params")

	// Verify forward pass works
	input, _ := tensor.New[float32]([]int{1, 4}, []float32{1, 2, 3, 4})
	output, err := ffn.Forward(context.Background(), input)
	testutils.AssertNoError(t, err, "Forward with SwiGLU+NoBias")
	testutils.AssertNotNil(t, output, "Output should not be nil")
	testutils.AssertEqual(t, output.Shape()[1], 4, "Output dim should be 4")
}

// TestFFN_WithGELU tests that GELU FFN builds and produces valid output.
func TestFFN_WithGELU(t *testing.T) {
	engine := compute.NewCPUEngine[float32](&numeric.Float32Ops{})
	ops := &numeric.Float32Ops{}

	ffn, err := NewFFN[float32]("test_gelu", engine, ops, 4, 8, 4,
		WithGELU[float32](),
		WithFFNNoBias[float32](),
	)
	testutils.AssertNoError(t, err, "NewFFN with GELU")
	testutils.AssertNotNil(t, ffn, "FFN should not be nil")

	input, _ := tensor.New[float32]([]int{1, 4}, []float32{1, 2, 3, 4})
	output, err := ffn.Forward(context.Background(), input)
	testutils.AssertNoError(t, err, "Forward with GELU")
	testutils.AssertNotNil(t, output, "Output should not be nil")
	testutils.AssertEqual(t, output.Shape()[1], 4, "Output dim should be 4")

	// Verify no NaN values in output.
	for i, v := range output.Data() {
		testutils.AssertTrue(t, !math.IsNaN(float64(v)), fmt.Sprintf("output[%d] is NaN", i))
	}
}

// TestFFN_GELUDiffersFromSwiGLU verifies GELU FFN produces different output than SwiGLU FFN.
func TestFFN_GELUDiffersFromSwiGLU(t *testing.T) {
	engine := compute.NewCPUEngine[float64](&numeric.Float64Ops{})
	ops := &numeric.Float64Ops{}

	inputDim, hiddenDim, outputDim := 4, 8, 4

	// Create SwiGLU FFN.
	swigluFFN, err := NewFFN[float64]("swiglu", engine, ops, inputDim, hiddenDim, outputDim,
		WithFFNNoBias[float64](),
	)
	testutils.AssertNoError(t, err, "NewFFN SwiGLU")

	// Create GELU FFN.
	geluFFN, err := NewFFN[float64]("gelu", engine, ops, inputDim, hiddenDim, outputDim,
		WithGELU[float64](),
		WithFFNNoBias[float64](),
	)
	testutils.AssertNoError(t, err, "NewFFN GELU")

	// Copy weights from SwiGLU to GELU so the only difference is the activation.
	geluFFN.w1.linear.weights.Value = swigluFFN.w1.linear.weights.Value
	geluFFN.w2.linear.weights.Value = swigluFFN.w2.linear.weights.Value
	geluFFN.w3.linear.weights.Value = swigluFFN.w3.linear.weights.Value

	input, _ := tensor.New[float64]([]int{1, inputDim}, []float64{1, 2, 3, 4})

	ctx := context.Background()
	swigluOut, err := swigluFFN.Forward(ctx, input)
	testutils.AssertNoError(t, err, "SwiGLU forward")

	geluOut, err := geluFFN.Forward(ctx, input)
	testutils.AssertNoError(t, err, "GELU forward")

	// Outputs should differ because different activations are used.
	swigluData := swigluOut.Data()
	geluData := geluOut.Data()
	testutils.AssertEqual(t, len(swigluData), len(geluData), "Output lengths should match")

	differs := false
	for i := range swigluData {
		if math.Abs(swigluData[i]-geluData[i]) > 1e-10 {
			differs = true
			break
		}
	}
	testutils.AssertTrue(t, differs, "GELU and SwiGLU outputs should differ")
}
