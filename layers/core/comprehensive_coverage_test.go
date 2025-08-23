package core

import (
	"context"
	"testing"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
	"github.com/zerfoo/zerfoo/testing/testutils"
)

// TestDense_ComprehensiveCoverage tests all realistic paths in Dense layer creation.
func TestDense_ComprehensiveCoverage(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	// Test 1: Empty name (should fail in NewLinear)
	_, err := NewDense("", engine, ops, 10, 5)
	testutils.AssertError(t, err, "expected error for empty name")

	// Test 2: Valid small sizes (should succeed)
	dense1, err := NewDense("small", engine, ops, 2, 3)
	testutils.AssertNoError(t, err, "expected no error for small sizes")
	testutils.AssertTrue(t, dense1 != nil, "expected non-nil dense layer")

	// Test 3: Valid medium sizes (should succeed)
	dense2, err := NewDense("medium", engine, ops, 10, 5)
	testutils.AssertNoError(t, err, "expected no error for medium sizes")
	testutils.AssertTrue(t, dense2 != nil, "expected non-nil dense layer")

	// Test 4: Single input/output (edge case that should succeed)
	dense3, err := NewDense("single", engine, ops, 1, 1)
	testutils.AssertNoError(t, err, "expected no error for single input/output")
	testutils.AssertTrue(t, dense3 != nil, "expected non-nil dense layer")

	// Test 5: Large sizes (should succeed but test memory allocation paths)
	dense4, err := NewDense("large", engine, ops, 100, 50)
	testutils.AssertNoError(t, err, "expected no error for large sizes")
	testutils.AssertTrue(t, dense4 != nil, "expected non-nil dense layer")

	// Test 6: Very large sizes (might stress test the system)
	dense5, err := NewDense("very_large", engine, ops, 500, 200)
	if err != nil {
		t.Logf("Very large sizes caused error: %v", err)
	} else {
		testutils.AssertTrue(t, dense5 != nil, "expected non-nil dense layer for very large sizes")
		t.Log("Very large sizes succeeded")
	}

	// Test all created layers have correct properties
	testDenseProperties(t, dense1, "small", 2, 3)
	testDenseProperties(t, dense2, "medium", 10, 5)
	testDenseProperties(t, dense3, "single", 1, 1)
	testDenseProperties(t, dense4, "large", 100, 50)

	if dense5 != nil {
		testDenseProperties(t, dense5, "very_large", 500, 200)
	}
}

// TestDense_SpecialCases tests edge cases and special scenarios.
func TestDense_SpecialCases(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	// Test with names that might cause issues
	specialNames := []string{
		"test_with_underscores",
		"test-with-dashes",
		"test123",
		"a", // single character
		"very_long_name_that_might_cause_issues_in_some_systems_or_implementations",
	}

	for _, name := range specialNames {
		dense, err := NewDense(name, engine, ops, 3, 2)
		testutils.AssertNoError(t, err, "expected no error for name: "+name)
		testutils.AssertTrue(t, dense != nil, "expected non-nil dense layer for name: "+name)

		// Test that the layer works
		testDenseBasicOperation(t, dense)
	}
}

// TestDense_ErrorRecovery tests error scenarios and recovery.
func TestDense_ErrorRecovery(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	// Test multiple empty name attempts
	for range 3 {
		_, err := NewDense("", engine, ops, 5, 3)
		testutils.AssertError(t, err, "expected error for empty name attempt")
	}

	// Test that we can still create valid layers after errors
	dense, err := NewDense("recovery_test", engine, ops, 4, 2)
	testutils.AssertNoError(t, err, "expected successful creation after error attempts")
	testutils.AssertTrue(t, dense != nil, "expected non-nil dense layer after recovery")
}

// Helper function to test dense layer properties.
func testDenseProperties(t *testing.T, dense *Dense[float32], expectedName string, _, outputSize int) {
	if dense == nil {
		t.Fatalf("dense layer is nil")
	}

	// Test output shape
	shape := dense.OutputShape()
	testutils.AssertTrue(t, len(shape) == 2, "expected 2D output shape")
	testutils.AssertTrue(t, shape[1] == outputSize, "expected correct output size")

	// Test parameters
	params := dense.Parameters()
	testutils.AssertTrue(t, len(params) == 2, "expected 2 parameters (weights and bias)")

	// Test name setting and getting (if the layer supports it)
	dense.SetName(expectedName + "_updated")
}

// Helper function to test basic dense layer operation.
func testDenseBasicOperation(t *testing.T, dense *Dense[float32]) {
	// Create a simple input
	inputData := []float32{1.0, 0.5, -0.3}
	input, err := tensor.New([]int{1, 3}, inputData)
	testutils.AssertNoError(t, err, "expected no error creating input")

	// Test forward pass
	output, _ := dense.Forward(context.Background(), input)
	testutils.AssertTrue(t, output != nil, "expected non-nil output")

	// Test backward pass
	outputShape := output.Shape()

	gradData := make([]float32, outputShape[0]*outputShape[1])
	for i := range gradData {
		gradData[i] = 1.0
	}

	outputGrad, err := tensor.New(outputShape, gradData)
	testutils.AssertNoError(t, err, "expected no error creating output gradient")

	inputGrads, _ := dense.Backward(context.Background(), outputGrad, input)
	testutils.AssertTrue(t, len(inputGrads) == 1, "expected 1 input gradient")
	testutils.AssertTrue(t, inputGrads[0] != nil, "expected non-nil input gradient")
}
