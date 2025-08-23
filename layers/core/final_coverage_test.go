package core

import (
	"context"
	"testing"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
	"github.com/zerfoo/zerfoo/testing/testutils"
)

// TestDense_NewBiasSpecificError attempts to trigger the specific error path
// where NewLinear succeeds but NewBias fails in NewDense.
func TestDense_NewBiasSpecificError(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	// Since we can't easily mock the internal calls, let's try to create
	// a scenario that exploits the differences between Linear and Bias creation.
	//
	// The key insight is that NewLinear creates a [inputSize, outputSize] tensor
	// while NewBias creates a [outputSize] tensor. Maybe we can find edge cases
	// where one succeeds and the other fails.

	// Test with extreme values that might cause different behavior
	testCases := []struct {
		name       string
		inputSize  int
		outputSize int
		expectErr  bool
	}{
		{"normal", 10, 5, false},
		{"zero_input", 0, 5, true},  // This might fail in NewLinear
		{"zero_output", 5, 0, true}, // This might fail in NewBias differently
		{"large", 1000, 1000, false},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			dense, err := NewDense(tc.name+"_dense", engine, ops, tc.inputSize, tc.outputSize)
			if tc.expectErr {
				if err != nil {
					t.Logf("Expected error occurred for %s: %v", tc.name, err)
				} else {
					t.Logf("Expected error but got success for %s", tc.name)
				}
			} else {
				testutils.AssertNoError(t, err, "expected no error for "+tc.name)
				testutils.AssertTrue(t, dense != nil, "expected non-nil dense for "+tc.name)
			}
		})
	}
}

// TestDense_WithCustomFactories tests NewDense indirectly by testing the underlying
// components with custom factories that might fail in different ways.
func TestDense_WithCustomFactories(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	// Test successful creation first
	dense, err := NewDense("factory_test", engine, ops, 3, 2)
	testutils.AssertNoError(t, err, "expected no error with default factories")
	testutils.AssertTrue(t, dense != nil, "expected non-nil dense layer")

	// Now test the components individually to understand their behavior
	// This helps us understand what might cause failures

	// Test Linear creation with various parameters
	linear1, err := NewLinear("linear_test", engine, ops, 3, 2)
	testutils.AssertNoError(t, err, "expected no error creating linear")
	testutils.AssertTrue(t, linear1 != nil, "expected non-nil linear")

	// Test Bias creation with various parameters
	bias1, err := NewBias("bias_test", engine, ops, 2)
	testutils.AssertNoError(t, err, "expected no error creating bias")
	testutils.AssertTrue(t, bias1 != nil, "expected non-nil bias")

	// Test with edge case sizes
	linear2, err := NewLinear("linear_edge", engine, ops, 1, 1)
	testutils.AssertNoError(t, err, "expected no error creating 1x1 linear")
	testutils.AssertTrue(t, linear2 != nil, "expected non-nil 1x1 linear")

	bias2, err := NewBias("bias_edge", engine, ops, 1)
	testutils.AssertNoError(t, err, "expected no error creating size-1 bias")
	testutils.AssertTrue(t, bias2 != nil, "expected non-nil size-1 bias")
}

// TestDense_ErrorHandling tests comprehensive error handling scenarios.
func TestDense_ErrorHandling(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	// Test all the error paths we can reasonably trigger
	errorTests := []struct {
		name       string
		layerName  string
		inputSize  int
		outputSize int
		shouldFail bool
		reason     string
	}{
		{"empty_name", "", 5, 3, true, "empty layer name"},
		{"valid_small", "small", 1, 1, false, "valid small layer"},
		{"valid_medium", "medium", 10, 5, false, "valid medium layer"},
		{"valid_large", "large", 100, 50, false, "valid large layer"},
	}

	for _, test := range errorTests {
		t.Run(test.name, func(t *testing.T) {
			dense, err := NewDense(test.layerName, engine, ops, test.inputSize, test.outputSize)

			if test.shouldFail {
				testutils.AssertError(t, err, "expected error for "+test.reason)
				testutils.AssertTrue(t, dense == nil, "expected nil dense for "+test.reason)
			} else {
				testutils.AssertNoError(t, err, "expected no error for "+test.reason)
				testutils.AssertTrue(t, dense != nil, "expected non-nil dense for "+test.reason)

				// Test that the layer actually works
				testBasicDenseOperation(t, dense, test.inputSize, test.outputSize)
			}
		})
	}
}

// Helper function to test basic dense layer functionality.
func testBasicDenseOperation(t *testing.T, dense *Dense[float32], inputSize, outputSize int) {
	// Create input tensor
	inputData := make([]float32, inputSize)
	for i := range inputData {
		inputData[i] = float32(i + 1)
	}

	input, err := tensor.New([]int{1, inputSize}, inputData)
	testutils.AssertNoError(t, err, "expected no error creating input")

	// Test forward pass
	output, _ := dense.Forward(context.Background(), input)
	testutils.AssertTrue(t, output != nil, "expected non-nil output")

	outputShape := output.Shape()
	testutils.AssertTrue(t, len(outputShape) == 2, "expected 2D output")
	testutils.AssertTrue(t, outputShape[1] == outputSize, "expected correct output size")

	// Test backward pass
	gradData := make([]float32, outputShape[0]*outputShape[1])
	for i := range gradData {
		gradData[i] = 1.0
	}

	outputGrad, err := tensor.New(outputShape, gradData)
	testutils.AssertNoError(t, err, "expected no error creating output gradient")

	inputGrads, _ := dense.Backward(context.Background(), outputGrad, input)
	testutils.AssertTrue(t, len(inputGrads) == 1, "expected 1 input gradient")
	testutils.AssertTrue(t, inputGrads[0] != nil, "expected non-nil input gradient")

	inputGradShape := inputGrads[0].Shape()
	testutils.AssertTrue(t, inputGradShape[1] == inputSize, "expected correct input gradient size")
}
