package core

import (
	"testing"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
	"github.com/zerfoo/zerfoo/testing/testutils"
)

// TestLinear_Creation tests basic Linear layer creation.
func TestLinear_Creation(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	layer, err := NewLinear("test_linear", engine, ops, 10, 5)
	testutils.AssertNoError(t, err, "expected no error when creating linear layer, got %v")
	testutils.AssertNotNil(t, layer, "expected layer to not be nil")

	expectedShape := []int{1, 5}
	testutils.AssertTrue(t, testutils.IntSliceEqual(expectedShape, layer.OutputShape()), "expected output shape to match")

	params := layer.Parameters()
	testutils.AssertEqual(t, 1, len(params), "expected 1 parameter")
	testutils.AssertEqual(t, "test_linear_weights", params[0].Name, "expected parameter name to match")
}

// TestLinear_WithXavier tests Xavier initialization.
func TestLinear_WithXavier(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	layer, err := NewLinearWithXavier("xavier_test", engine, ops, 10, 5)
	testutils.AssertNoError(t, err, "expected no error when creating linear layer with Xavier, got %v")
	testutils.AssertNotNil(t, layer, "expected layer to not be nil")

	// Check that weights were initialized
	params := layer.Parameters()
	testutils.AssertEqual(t, 1, len(params), "expected 1 parameter")

	weights := params[0].Value
	testutils.AssertNotNil(t, weights, "expected weights tensor to not be nil")

	expectedShape := []int{10, 5}
	testutils.AssertTrue(t, testutils.IntSliceEqual(expectedShape, weights.Shape()), "expected weights shape to match")
}

// TestLinear_WithHe tests He initialization.
func TestLinear_WithHe(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	layer, err := NewLinearWithHe("he_test", engine, ops, 10, 5)
	testutils.AssertNoError(t, err, "expected no error when creating linear layer with He, got %v")
	testutils.AssertNotNil(t, layer, "expected layer to not be nil")

	params := layer.Parameters()
	weights := params[0].Value

	expectedShape := []int{10, 5}
	testutils.AssertTrue(t, testutils.IntSliceEqual(expectedShape, weights.Shape()), "expected weights shape to match")
}

// TestLinear_WithUniform tests uniform initialization.
func TestLinear_WithUniform(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	layer, err := NewLinearWithUniform("uniform_test", engine, ops, 10, 5, 0.1)
	testutils.AssertNoError(t, err, "expected no error when creating linear layer with Uniform, got %v")
	testutils.AssertNotNil(t, layer, "expected layer to not be nil")

	params := layer.Parameters()
	weights := params[0].Value

	expectedShape := []int{10, 5}
	testutils.AssertTrue(t, testutils.IntSliceEqual(expectedShape, weights.Shape()), "expected weights shape to match")
}

// TestLinear_ForwardPass tests the forward pass functionality.
func TestLinear_ForwardPass(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	layer, err := NewLinearWithXavier("forward_test", engine, ops, 3, 2)
	testutils.AssertNoError(t, err, "expected no error when creating linear layer, got %v")

	// Create input tensor (1x3)
	inputData := []float32{1.0, 2.0, 3.0}
	input, err := tensor.New([]int{1, 3}, inputData)
	testutils.AssertNoError(t, err, "expected no error when creating input tensor, got %v")

	output, _ := layer.Forward(input)
	testutils.AssertNotNil(t, output, "expected forward pass output to not be nil")

	expectedShape := []int{1, 2}
	testutils.AssertTrue(t, testutils.IntSliceEqual(expectedShape, output.Shape()), "expected output shape to match")
}

// TestLinear_BackwardPass tests the backward pass functionality.
func TestLinear_BackwardPass(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	layer, err := NewLinearWithXavier("backward_test", engine, ops, 3, 2)
	testutils.AssertNoError(t, err, "expected no error when creating linear layer, got %v")

	// Create input tensor (1x3)
	inputData := []float32{1.0, 2.0, 3.0}
	input, err := tensor.New([]int{1, 3}, inputData)
	testutils.AssertNoError(t, err, "expected no error when creating input tensor, got %v")

	// Forward pass
	_, _ = layer.Forward(input)

	// Create output gradient (1x2)
	outputGradData := []float32{1.0, 1.0}
	outputGrad, err := tensor.New([]int{1, 2}, outputGradData)
	testutils.AssertNoError(t, err, "expected no error when creating output gradient tensor, got %v")

	// Backward pass
	inputGrads, _ := layer.Backward(outputGrad)

	testutils.AssertEqual(t, 1, len(inputGrads), "expected 1 input gradient")

	inputGrad := inputGrads[0]
	expectedShape := []int{1, 3}
	testutils.AssertTrue(t, testutils.IntSliceEqual(expectedShape, inputGrad.Shape()), "expected input gradient shape to match")

	// Check that weight gradient was computed
	params := layer.Parameters()
	weightGrad := params[0].Gradient
	testutils.AssertNotNil(t, weightGrad, "expected weight gradient to not be nil")

	expectedWeightGradShape := []int{3, 2}
	testutils.AssertTrue(t, testutils.IntSliceEqual(expectedWeightGradShape, weightGrad.Shape()), "expected weight gradient shape to match")
}

// TestLinear_CustomInitializer tests using a custom initializer.
func TestLinear_CustomInitializer(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	// Create custom initializer that sets all weights to 0.5
	customInit := &testInitializer[float32]{value: 0.5}

	layer, err := NewLinearWithInitializer("custom_test", engine, ops, 2, 2, customInit)
	testutils.AssertNoError(t, err, "expected no error when creating linear layer with custom initializer, got %v")

	params := layer.Parameters()
	weights := params[0].Value
	weightsData := weights.Data()

	// Check that all weights are 0.5
	for _, weight := range weightsData {
		testutils.AssertEqual(t, float32(0.5), weight, "expected weight to be 0.5")
	}
}

// TestLinear_SetName tests the SetName functionality.
func TestLinear_SetName(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	layer, err := NewLinearWithXavier("original_name", engine, ops, 3, 2)
	testutils.AssertNoError(t, err, "expected no error when creating linear layer, got %v")

	layer.SetName("new_name")

	params := layer.Parameters()
	testutils.AssertEqual(t, "new_name_weights", params[0].Name, "expected parameter name to match")
}

// Helper test initializer.
type testInitializer[T tensor.Numeric] struct {
	value T
}

func (ti *testInitializer[T]) Initialize(inputSize, outputSize int) ([]T, error) {
	weights := make([]T, inputSize*outputSize)
	for i := range weights {
		weights[i] = ti.value
	}

	return weights, nil
}
