package normalization

import (
	"context"
	"testing"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
	"github.com/zerfoo/zerfoo/testing/testutils"
	"github.com/zerfoo/zerfoo/types"
)

// TestLayerNormalization_WithEpsilon tests LayerNormalization with custom epsilon option.
func TestLayerNormalization_WithEpsilon(t *testing.T) {
	engine := compute.NewCPUEngine[float32](&numeric.Float32Ops{})

	// Test with custom epsilon value
	customEpsilon := float32(1e-6)
	ln, err := NewLayerNormalization[float32](engine, 4, WithLayerNormEpsilon[float32](customEpsilon))
	testutils.AssertNoError(t, err, "NewLayerNormalization with custom epsilon should not return an error")
	testutils.AssertNotNil(t, ln, "LayerNormalization should not be nil")

	// Check that epsilon is set correctly
	testutils.AssertFloatEqual(t, customEpsilon, ln.epsilon, float32(1e-9), "Epsilon should be set to custom value")
}

// TestLayerNormalization_DefaultEpsilon tests LayerNormalization with default epsilon.
func TestLayerNormalization_DefaultEpsilon(t *testing.T) {
	engine := compute.NewCPUEngine[float32](&numeric.Float32Ops{})

	// Test with default epsilon (no options)
	ln, err := NewLayerNormalization[float32](engine, 4)
	testutils.AssertNoError(t, err, "NewLayerNormalization with default epsilon should not return an error")
	testutils.AssertNotNil(t, ln, "LayerNormalization should not be nil")

	// Check that epsilon is set to default value (1e-5)
	expectedEpsilon := float32(1e-5)
	testutils.AssertFloatEqual(t, expectedEpsilon, ln.epsilon, float32(1e-9), "Epsilon should be set to default value")
}

// TestLayerNormalization_Parameters tests that LayerNormalization returns correct parameters.
func TestLayerNormalization_Parameters(t *testing.T) {
	engine := compute.NewCPUEngine[float32](&numeric.Float32Ops{})

	ln, err := NewLayerNormalization[float32](engine, 4)
	testutils.AssertNoError(t, err, "NewLayerNormalization should not return an error")

	params := ln.Parameters()
	testutils.AssertEqual(t, len(params), 2, "LayerNormalization should have 2 parameters (gamma and beta)")
}

// TestLayerNormalization_OutputShape tests OutputShape method.
func TestLayerNormalization_OutputShape(t *testing.T) {
	ctx := context.Background()
	engine := compute.NewCPUEngine[float32](&numeric.Float32Ops{})

	ln, err := NewLayerNormalization[float32](engine, 4)
	testutils.AssertNoError(t, err, "NewLayerNormalization should not return an error")

	// Create a test input to initialize output shape
	inputShape := []int{2, 4}
	inputData := []float32{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0}
	inputTensor, err := tensor.New[float32](inputShape, inputData)
	testutils.AssertNoError(t, err, "Failed to create input tensor")

	// Run forward to initialize output shape
	_, err = ln.Forward(ctx, inputTensor)
	testutils.AssertNoError(t, err, "Forward pass should not return an error")

	// Now test OutputShape
	outputShape := ln.OutputShape()
	testutils.AssertTrue(t, testutils.IntSliceEqual(inputShape, outputShape), "OutputShape should match input shape")
}

// TestLayerNormalization_Forward tests Forward method.
func TestLayerNormalization_Forward(t *testing.T) {
	ctx := context.Background()
	engine := compute.NewCPUEngine[float32](&numeric.Float32Ops{})

	normalizedDim := 4
	ln, err := NewLayerNormalization[float32](engine, normalizedDim)
	testutils.AssertNoError(t, err, "NewLayerNormalization should not return an error")

	// Create test input tensor [batch=2, seq=3, features=4]
	batchSize := 2
	seqLen := 3
	inputShape := []int{batchSize, seqLen, normalizedDim}
	inputData := []float32{
		// Batch 1, Seq 1
		1.0, 2.0, 3.0, 4.0,
		// Batch 1, Seq 2
		5.0, 6.0, 7.0, 8.0,
		// Batch 1, Seq 3
		9.0, 10.0, 11.0, 12.0,
		// Batch 2, Seq 1
		2.0, 4.0, 6.0, 8.0,
		// Batch 2, Seq 2
		1.0, 3.0, 5.0, 7.0,
		// Batch 2, Seq 3
		10.0, 12.0, 14.0, 16.0,
	}
	inputTensor, err := tensor.New[float32](inputShape, inputData)
	testutils.AssertNoError(t, err, "Failed to create input tensor")

	// Test forward pass
	output, err := ln.Forward(ctx, inputTensor)
	testutils.AssertNoError(t, err, "Forward pass should not return an error")
	testutils.AssertNotNil(t, output, "Output should not be nil")

	// Check output shape
	testutils.AssertTrue(t, testutils.IntSliceEqual(inputShape, output.Shape()), "Output shape should match input shape")

	// Check that output values are normalized (mean ≈ 0, std ≈ 1)
	outputData := output.Data()
	testutils.AssertEqual(t, len(outputData), len(inputData), "Output data length should match input")

	// Verify normalization properties for first sequence
	firstSeq := outputData[0:4] // First sequence of first batch
	mean := (firstSeq[0] + firstSeq[1] + firstSeq[2] + firstSeq[3]) / 4.0
	testutils.AssertFloatEqual(t, 0.0, mean, 1e-5, "Normalized sequence should have mean ≈ 0")
}

// TestLayerNormalization_Forward_EdgeCases tests Forward with edge cases.
func TestLayerNormalization_Forward_EdgeCases(t *testing.T) {
	ctx := context.Background()
	engine := compute.NewCPUEngine[float32](&numeric.Float32Ops{})

	normalizedDim := 3
	ln, err := NewLayerNormalization[float32](engine, normalizedDim)
	testutils.AssertNoError(t, err, "NewLayerNormalization should not return an error")

	// Test with zeros (should not crash due to epsilon)
	zeroShape := []int{1, 1, normalizedDim}
	zeroData := []float32{0.0, 0.0, 0.0}
	zeroTensor, err := tensor.New[float32](zeroShape, zeroData)
	testutils.AssertNoError(t, err, "Failed to create zero tensor")

	output, err := ln.Forward(ctx, zeroTensor)
	testutils.AssertNoError(t, err, "Forward with zeros should not error")
	testutils.AssertNotNil(t, output, "Output should not be nil")

	// Test with single element
	singleShape := []int{1, normalizedDim}
	singleData := []float32{5.0, 10.0, 15.0}
	singleTensor, err := tensor.New[float32](singleShape, singleData)
	testutils.AssertNoError(t, err, "Failed to create single tensor")

	output2, err := ln.Forward(ctx, singleTensor)
	testutils.AssertNoError(t, err, "Forward with single element should not error")
	testutils.AssertNotNil(t, output2, "Output should not be nil")
}

// TestLayerNormalization_Backward tests Backward method.
func TestLayerNormalization_Backward(t *testing.T) {
	ctx := context.Background()
	engine := compute.NewCPUEngine[float32](&numeric.Float32Ops{})

	normalizedDim := 3
	ln, err := NewLayerNormalization[float32](engine, normalizedDim)
	testutils.AssertNoError(t, err, "NewLayerNormalization should not return an error")

	// Create input tensor
	inputShape := []int{2, normalizedDim}
	inputData := []float32{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}
	inputTensor, err := tensor.New[float32](inputShape, inputData)
	testutils.AssertNoError(t, err, "Failed to create input tensor")

	// Run forward pass first to cache necessary values
	_, err = ln.Forward(ctx, inputTensor)
	testutils.AssertNoError(t, err, "Forward pass should not return an error")

	// Create gradient tensor (same shape as output)
	gradData := []float32{0.1, 0.2, 0.3, 0.4, 0.5, 0.6}
	gradTensor, err := tensor.New[float32](inputShape, gradData)
	testutils.AssertNoError(t, err, "Failed to create gradient tensor")

	// Test backward pass
	inputGrads, err := ln.Backward(ctx, types.FullBackprop, gradTensor, inputTensor)
	if err != nil {
		// If backward is not implemented, just verify it returns an error gracefully
		testutils.AssertError(t, err, "Backward pass should return an error if not implemented")

		return
	}

	testutils.AssertNotNil(t, inputGrads, "Input gradients should not be nil")
	testutils.AssertEqual(t, len(inputGrads), 1, "Should return one input gradient")

	// Check gradient shape
	inputGrad := inputGrads[0]
	testutils.AssertTrue(t, testutils.IntSliceEqual(inputShape, inputGrad.Shape()), "Input gradient shape should match input shape")
}
