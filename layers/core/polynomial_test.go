package core

import (
	"testing"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
	"github.com/zerfoo/zerfoo/testing/testutils"
)

// TestPolynomialExpansion_Creation tests basic creation of polynomial expansion layers
func TestPolynomialExpansion_Creation(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	// Test valid creation
	poly, err := NewPolynomialExpansion("test_poly", engine, ops, 2, 2, true)
	testutils.AssertNoError(t, err, "expected no error creating polynomial layer")
	testutils.AssertTrue(t, poly != nil, "expected non-nil polynomial layer")
	testutils.AssertTrue(t, poly.GetDegree() == 2, "expected degree 2")
	testutils.AssertTrue(t, poly.GetInputSize() == 2, "expected input size 2")
	testutils.AssertTrue(t, poly.HasBias(), "expected bias term")

	// Test without bias
	polyNoBias, err := NewPolynomialExpansion("test_no_bias", engine, ops, 3, 2, false)
	testutils.AssertNoError(t, err, "expected no error creating polynomial layer without bias")
	testutils.AssertTrue(t, !polyNoBias.HasBias(), "expected no bias term")
}

// TestPolynomialExpansion_ErrorCases tests error handling in creation
func TestPolynomialExpansion_ErrorCases(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	// Test empty name
	_, err := NewPolynomialExpansion("", engine, ops, 2, 2, true)
	testutils.AssertError(t, err, "expected error for empty name")

	// Test invalid input size
	_, err = NewPolynomialExpansion("test", engine, ops, 0, 2, true)
	testutils.AssertError(t, err, "expected error for zero input size")

	_, err = NewPolynomialExpansion("test", engine, ops, -1, 2, true)
	testutils.AssertError(t, err, "expected error for negative input size")

	// Test invalid degree
	_, err = NewPolynomialExpansion("test", engine, ops, 2, 0, true)
	testutils.AssertError(t, err, "expected error for zero degree")

	_, err = NewPolynomialExpansion("test", engine, ops, 2, -1, true)
	testutils.AssertError(t, err, "expected error for negative degree")
}

// TestPolynomialExpansion_TermGeneration tests polynomial term generation
func TestPolynomialExpansion_TermGeneration(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	// Test 2 features, degree 2, with bias
	poly, err := NewPolynomialExpansion("test", engine, ops, 2, 2, true)
	testutils.AssertNoError(t, err, "expected no error creating polynomial layer")

	terms := poly.GetTermIndices()
	// Expected terms: [0,0] (bias), [1,0] (x1), [0,1] (x2), [2,0] (x1^2), [1,1] (x1*x2), [0,2] (x2^2)
	expectedTerms := [][]int{
		{0, 0}, // bias
		{1, 0}, // x1
		{0, 1}, // x2
		{2, 0}, // x1^2
		{1, 1}, // x1*x2
		{0, 2}, // x2^2
	}

	testutils.AssertTrue(t, len(terms) == len(expectedTerms), "expected correct number of terms")
	
	// Check that all expected terms are present (order might vary)
	for _, expectedTerm := range expectedTerms {
		found := false
		for _, actualTerm := range terms {
			if len(actualTerm) == len(expectedTerm) {
				match := true
				for i, power := range expectedTerm {
					if actualTerm[i] != power {
						match = false
						break
					}
				}
				if match {
					found = true
					break
				}
			}
		}
		testutils.AssertTrue(t, found, "expected to find term in generated terms")
	}
}

// TestPolynomialExpansion_ForwardPass tests the forward pass computation
func TestPolynomialExpansion_ForwardPass(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	// Test simple case: 2 features, degree 2, with bias
	poly, err := NewPolynomialExpansion("test", engine, ops, 2, 2, true)
	testutils.AssertNoError(t, err, "expected no error creating polynomial layer")

	// Input: [2.0, 3.0]
	inputData := []float32{2.0, 3.0}
	input, err := tensor.New([]int{1, 2}, inputData)
	testutils.AssertNoError(t, err, "expected no error creating input tensor")

	// Forward pass
	output := poly.Forward(input)
	testutils.AssertTrue(t, output != nil, "expected non-nil output")

	outputShape := output.Shape()
	testutils.AssertTrue(t, len(outputShape) == 2, "expected 2D output")
	testutils.AssertTrue(t, outputShape[0] == 1, "expected batch size 1")
	testutils.AssertTrue(t, outputShape[1] == poly.GetOutputSize(), "expected correct output size")

	outputData := output.Data()
	
	// Expected output for input [2, 3] with terms [bias, x1, x2, x1^2, x1*x2, x2^2]:
	// [1, 2, 3, 4, 6, 9]
	// Note: The actual order depends on how terms are generated, so we check for expected values
	expectedValues := map[float32]bool{
		1.0: false, // bias
		2.0: false, // x1
		3.0: false, // x2
		4.0: false, // x1^2
		6.0: false, // x1*x2
		9.0: false, // x2^2
	}

	// Mark which expected values we found
	for _, value := range outputData {
		if _, exists := expectedValues[value]; exists {
			expectedValues[value] = true
		}
	}

	// Check that all expected values were found
	for value, found := range expectedValues {
		testutils.AssertTrue(t, found, "expected to find value in output")
		_ = value // Use the value to avoid unused variable warning
	}
}

// TestPolynomialExpansion_ForwardPassNoBias tests forward pass without bias term
func TestPolynomialExpansion_ForwardPassNoBias(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	// Test without bias: 2 features, degree 1
	poly, err := NewPolynomialExpansion("test", engine, ops, 2, 1, false)
	testutils.AssertNoError(t, err, "expected no error creating polynomial layer")

	// Input: [2.0, 3.0]
	inputData := []float32{2.0, 3.0}
	input, err := tensor.New([]int{1, 2}, inputData)
	testutils.AssertNoError(t, err, "expected no error creating input tensor")

	// Forward pass
	output := poly.Forward(input)
	outputData := output.Data()

	// Expected output for degree 1 without bias: [x1, x2] = [2, 3]
	testutils.AssertTrue(t, len(outputData) == 2, "expected 2 output features")
	
	// Check that we have the input values (order might vary)
	foundValues := make(map[float32]bool)
	for _, value := range outputData {
		foundValues[value] = true
	}
	
	testutils.AssertTrue(t, foundValues[2.0], "expected to find x1 value")
	testutils.AssertTrue(t, foundValues[3.0], "expected to find x2 value")
}

// TestPolynomialExpansion_BatchProcessing tests processing multiple samples
func TestPolynomialExpansion_BatchProcessing(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	poly, err := NewPolynomialExpansion("test", engine, ops, 2, 1, true)
	testutils.AssertNoError(t, err, "expected no error creating polynomial layer")

	// Batch input: 2 samples, 2 features each
	inputData := []float32{
		1.0, 2.0, // sample 1: [1, 2]
		3.0, 4.0, // sample 2: [3, 4]
	}
	input, err := tensor.New([]int{2, 2}, inputData)
	testutils.AssertNoError(t, err, "expected no error creating batch input tensor")

	// Forward pass
	output := poly.Forward(input)
	outputShape := output.Shape()
	
	testutils.AssertTrue(t, outputShape[0] == 2, "expected batch size 2")
	testutils.AssertTrue(t, outputShape[1] == poly.GetOutputSize(), "expected correct output size")

	outputData := output.Data()
	testutils.AssertTrue(t, len(outputData) == 2*poly.GetOutputSize(), "expected correct output data length")
}

// TestPolynomialExpansion_ForwardErrorCases tests error handling in forward pass
func TestPolynomialExpansion_ForwardErrorCases(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	poly, err := NewPolynomialExpansion("test", engine, ops, 2, 2, true)
	testutils.AssertNoError(t, err, "expected no error creating polynomial layer")

	// Test with no inputs
	testutils.AssertPanics(t, func() {
		poly.Forward()
	}, "expected panic with no inputs")

	// Test with multiple inputs
	input1, _ := tensor.New([]int{1, 2}, []float32{1.0, 2.0})
	input2, _ := tensor.New([]int{1, 2}, []float32{3.0, 4.0})
	testutils.AssertPanics(t, func() {
		poly.Forward(input1, input2)
	}, "expected panic with multiple inputs")

	// Test with wrong input shape (1D)
	input1D, _ := tensor.New([]int{2}, []float32{1.0, 2.0})
	testutils.AssertPanics(t, func() {
		poly.Forward(input1D)
	}, "expected panic with 1D input")

	// Test with wrong input size
	inputWrongSize, _ := tensor.New([]int{1, 3}, []float32{1.0, 2.0, 3.0})
	testutils.AssertPanics(t, func() {
		poly.Forward(inputWrongSize)
	}, "expected panic with wrong input size")
}

// TestPolynomialExpansion_BackwardPass tests the backward pass computation
func TestPolynomialExpansion_BackwardPass(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	poly, err := NewPolynomialExpansion("test", engine, ops, 2, 1, true)
	testutils.AssertNoError(t, err, "expected no error creating polynomial layer")

	// Forward pass first
	inputData := []float32{2.0, 3.0}
	input, err := tensor.New([]int{1, 2}, inputData)
	testutils.AssertNoError(t, err, "expected no error creating input tensor")

	output := poly.Forward(input)
	
	// Create output gradient
	outputGradData := make([]float32, poly.GetOutputSize())
	for i := range outputGradData {
		outputGradData[i] = 1.0 // Gradient of 1 for all outputs
	}
	outputGrad, err := tensor.New(output.Shape(), outputGradData)
	testutils.AssertNoError(t, err, "expected no error creating output gradient")

	// Backward pass
	inputGrads := poly.Backward(outputGrad)
	testutils.AssertTrue(t, len(inputGrads) == 1, "expected 1 input gradient")
	testutils.AssertTrue(t, inputGrads[0] != nil, "expected non-nil input gradient")

	inputGradShape := inputGrads[0].Shape()
	testutils.AssertTrue(t, len(inputGradShape) == 2, "expected 2D input gradient")
	testutils.AssertTrue(t, inputGradShape[0] == 1, "expected batch size 1")
	testutils.AssertTrue(t, inputGradShape[1] == 2, "expected input size 2")
}

// TestPolynomialExpansion_BackwardErrorCases tests error handling in backward pass
func TestPolynomialExpansion_BackwardErrorCases(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	poly, err := NewPolynomialExpansion("test", engine, ops, 2, 2, true)
	testutils.AssertNoError(t, err, "expected no error creating polynomial layer")

	// Test with wrong output gradient size
	wrongSizeGrad, _ := tensor.New([]int{1, 3}, []float32{1.0, 1.0, 1.0})
	testutils.AssertPanics(t, func() {
		poly.Backward(wrongSizeGrad)
	}, "expected panic with wrong gradient size")
}

// TestPolynomialExpansion_Parameters tests parameter retrieval
func TestPolynomialExpansion_Parameters(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	poly, err := NewPolynomialExpansion("test", engine, ops, 2, 2, true)
	testutils.AssertNoError(t, err, "expected no error creating polynomial layer")

	// Polynomial expansion has no trainable parameters
	params := poly.Parameters()
	testutils.AssertTrue(t, params == nil, "expected no parameters")
}

// TestPolynomialExpansion_OutputShape tests output shape computation
func TestPolynomialExpansion_OutputShape(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	poly, err := NewPolynomialExpansion("test", engine, ops, 3, 2, true)
	testutils.AssertNoError(t, err, "expected no error creating polynomial layer")

	shape := poly.OutputShape()
	testutils.AssertTrue(t, len(shape) == 2, "expected 2D output shape")
	testutils.AssertTrue(t, shape[1] == poly.GetOutputSize(), "expected correct output size")
}

// TestPolynomialExpansion_SetName tests name setting
func TestPolynomialExpansion_SetName(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	poly, err := NewPolynomialExpansion("test", engine, ops, 2, 2, true)
	testutils.AssertNoError(t, err, "expected no error creating polynomial layer")

	// SetName should not panic (even though it doesn't store the name)
	poly.SetName("new_name")
}

// TestPolynomialExpansion_HigherDegree tests polynomial expansion with higher degrees
func TestPolynomialExpansion_HigherDegree(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	// Test degree 3 with 2 features
	poly, err := NewPolynomialExpansion("test", engine, ops, 2, 3, false)
	testutils.AssertNoError(t, err, "expected no error creating degree 3 polynomial layer")

	// Input: [2.0, 1.0]
	inputData := []float32{2.0, 1.0}
	input, err := tensor.New([]int{1, 2}, inputData)
	testutils.AssertNoError(t, err, "expected no error creating input tensor")

	// Forward pass
	output := poly.Forward(input)
	testutils.AssertTrue(t, output != nil, "expected non-nil output")

	outputData := output.Data()
	
	// For degree 3 with 2 features, we should have terms like:
	// x1, x2, x1^2, x1*x2, x2^2, x1^3, x1^2*x2, x1*x2^2, x2^3
	// With input [2, 1]: [2, 1, 4, 2, 1, 8, 4, 2, 1]
	
	// Check that we have the expected number of terms
	expectedTermCount := 9 // All combinations up to degree 3, no bias
	testutils.AssertTrue(t, len(outputData) == expectedTermCount, "expected correct number of output terms")
	
	// Check some specific values
	foundValues := make(map[float32]int)
	for _, value := range outputData {
		foundValues[value]++
	}
	
	// We should find these values: 1, 2, 4, 8 (some may appear multiple times)
	testutils.AssertTrue(t, foundValues[1.0] > 0, "expected to find value 1")
	testutils.AssertTrue(t, foundValues[2.0] > 0, "expected to find value 2")
	testutils.AssertTrue(t, foundValues[4.0] > 0, "expected to find value 4")
	testutils.AssertTrue(t, foundValues[8.0] > 0, "expected to find value 8")
}
