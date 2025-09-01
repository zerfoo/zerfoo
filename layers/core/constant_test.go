package core

import (
	"context"
	"math"
	"testing"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
	"github.com/zerfoo/zerfoo/types"
)

func TestNewConstant(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	t.Run("valid_constant", func(t *testing.T) {
		// Create test tensor
		data := []float32{1.0, 2.0, 3.0, 4.0}
		value, err := tensor.New[float32]([]int{2, 2}, data)
		if err != nil {
			t.Fatalf("Failed to create tensor: %v", err)
		}

		layer, err := NewConstant[float32]("test_constant", engine, ops, value)
		if err != nil {
			t.Fatalf("Failed to create Constant layer: %v", err)
		}

		if layer.OpType() != "Constant" {
			t.Errorf("Expected OpType 'Constant', got '%s'", layer.OpType())
		}

		expectedShape := []int{2, 2}
		actualShape := layer.OutputShape()
		if len(actualShape) != len(expectedShape) {
			t.Errorf("Shape length mismatch: expected %v, got %v", expectedShape, actualShape)
		}

		for i, expected := range expectedShape {
			if actualShape[i] != expected {
				t.Errorf("Shape dimension %d: expected %d, got %d", i, expected, actualShape[i])
			}
		}
	})

	t.Run("empty_name", func(t *testing.T) {
		value, _ := tensor.New[float32]([]int{1}, []float32{1.0})
		_, err := NewConstant[float32]("", engine, ops, value)
		if err == nil {
			t.Error("Expected error for empty name, got none")
		}
	})

	t.Run("nil_value", func(t *testing.T) {
		_, err := NewConstant[float32]("test", engine, ops, nil)
		if err == nil {
			t.Error("Expected error for nil value, got none")
		}
	})
}

func TestNewConstantFromData(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	t.Run("valid_data", func(t *testing.T) {
		data := []float32{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}
		shape := []int{2, 3}

		layer, err := NewConstantFromData[float32]("test_constant", engine, ops, shape, data)
		if err != nil {
			t.Fatalf("Failed to create Constant layer from data: %v", err)
		}

		actualShape := layer.OutputShape()
		if len(actualShape) != len(shape) {
			t.Errorf("Shape length mismatch: expected %v, got %v", shape, actualShape)
		}

		for i, expected := range shape {
			if actualShape[i] != expected {
				t.Errorf("Shape dimension %d: expected %d, got %d", i, expected, actualShape[i])
			}
		}

		// Check data integrity
		value := layer.GetValue()
		actualData := value.Data()
		if len(actualData) != len(data) {
			t.Errorf("Data length mismatch: expected %d, got %d", len(data), len(actualData))
		}

		for i, expected := range data {
			if math.Abs(float64(actualData[i]-expected)) > 1e-6 {
				t.Errorf("Data value %d: expected %.6f, got %.6f", i, expected, actualData[i])
			}
		}
	})

	t.Run("invalid_shape_data_mismatch", func(t *testing.T) {
		data := []float32{1.0, 2.0}
		shape := []int{2, 2} // Requires 4 elements, but only 2 provided

		_, err := NewConstantFromData[float32]("test_constant", engine, ops, shape, data)
		if err == nil {
			t.Error("Expected error for shape-data mismatch, got none")
		}
	})
}

func TestConstantForward(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	t.Run("forward_no_inputs", func(t *testing.T) {
		data := []float32{10.0, 20.0, 30.0}
		layer, err := NewConstantFromData[float32]("test_constant", engine, ops, []int{1, 3}, data)
		if err != nil {
			t.Fatalf("Failed to create Constant layer: %v", err)
		}

		ctx := context.Background()
		output, err := layer.Forward(ctx)
		if err != nil {
			t.Fatalf("Forward pass failed: %v", err)
		}

		expectedShape := []int{1, 3}
		actualShape := output.Shape()
		if len(actualShape) != len(expectedShape) {
			t.Errorf("Output shape length mismatch: expected %v, got %v", expectedShape, actualShape)
		}

		outputData := output.Data()
		for i, expected := range data {
			if math.Abs(float64(outputData[i]-expected)) > 1e-6 {
				t.Errorf("Output data %d: expected %.6f, got %.6f", i, expected, outputData[i])
			}
		}
	})

	t.Run("forward_with_inputs", func(t *testing.T) {
		// Constant should ignore inputs and return constant value
		data := []float32{5.0, 15.0}
		layer, err := NewConstantFromData[float32]("test_constant", engine, ops, []int{2}, data)
		if err != nil {
			t.Fatalf("Failed to create Constant layer: %v", err)
		}

		// Create dummy inputs (should be ignored)
		input1, _ := tensor.New[float32]([]int{2}, []float32{100.0, 200.0})
		input2, _ := tensor.New[float32]([]int{1}, []float32{300.0})

		ctx := context.Background()
		output, err := layer.Forward(ctx, input1, input2)
		if err != nil {
			t.Fatalf("Forward pass with inputs failed: %v", err)
		}

		// Output should still be the constant value, ignoring inputs
		outputData := output.Data()
		for i, expected := range data {
			if math.Abs(float64(outputData[i]-expected)) > 1e-6 {
				t.Errorf("Output data %d: expected %.6f, got %.6f", i, expected, outputData[i])
			}
		}
	})
}

func TestConstantBackward(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	t.Run("backward_no_inputs", func(t *testing.T) {
		data := []float32{1.0, 2.0}
		layer, err := NewConstantFromData[float32]("test_constant", engine, ops, []int{2}, data)
		if err != nil {
			t.Fatalf("Failed to create Constant layer: %v", err)
		}

		// Output gradient
		gradOutput, _ := tensor.New[float32]([]int{2}, []float32{1.0, 1.0})

		ctx := context.Background()
		gradInputs, err := layer.Backward(ctx, types.FullBackprop, gradOutput)
		if err != nil {
			t.Fatalf("Backward pass failed: %v", err)
		}

		// Should return empty gradient list for no inputs
		if len(gradInputs) != 0 {
			t.Errorf("Expected 0 input gradients, got %d", len(gradInputs))
		}
	})

	t.Run("backward_with_inputs", func(t *testing.T) {
		data := []float32{10.0}
		layer, err := NewConstantFromData[float32]("test_constant", engine, ops, []int{1}, data)
		if err != nil {
			t.Fatalf("Failed to create Constant layer: %v", err)
		}

		// Create inputs
		input1, _ := tensor.New[float32]([]int{2, 3}, []float32{1, 2, 3, 4, 5, 6})
		input2, _ := tensor.New[float32]([]int{1}, []float32{100.0})

		// Output gradient
		gradOutput, _ := tensor.New[float32]([]int{1}, []float32{5.0})

		ctx := context.Background()
		gradInputs, err := layer.Backward(ctx, types.FullBackprop, gradOutput, input1, input2)
		if err != nil {
			t.Fatalf("Backward pass with inputs failed: %v", err)
		}

		if len(gradInputs) != 2 {
			t.Errorf("Expected 2 input gradients, got %d", len(gradInputs))
		}

		// Check gradient shapes match input shapes
		if !shapeEqual(gradInputs[0].Shape(), input1.Shape()) {
			t.Errorf("Gradient 0 shape mismatch: expected %v, got %v", input1.Shape(), gradInputs[0].Shape())
		}

		if !shapeEqual(gradInputs[1].Shape(), input2.Shape()) {
			t.Errorf("Gradient 1 shape mismatch: expected %v, got %v", input2.Shape(), gradInputs[1].Shape())
		}

		// Check that gradients are zero
		grad1Data := gradInputs[0].Data()
		for i, val := range grad1Data {
			if math.Abs(float64(val)) > 1e-9 {
				t.Errorf("Gradient 0 value %d: expected 0, got %.9f", i, val)
			}
		}

		grad2Data := gradInputs[1].Data()
		for i, val := range grad2Data {
			if math.Abs(float64(val)) > 1e-9 {
				t.Errorf("Gradient 1 value %d: expected 0, got %.9f", i, val)
			}
		}
	})
}

func TestConstantAttributes(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	t.Run("float32_attributes", func(t *testing.T) {
		data := []float32{1.0, 2.0, 3.0, 4.0}
		layer, err := NewConstantFromData[float32]("test_constant", engine, ops, []int{2, 2}, data)
		if err != nil {
			t.Fatalf("Failed to create Constant layer: %v", err)
		}

		attrs := layer.Attributes()
		
		shape, ok := attrs["shape"].([]int)
		if !ok {
			t.Error("Expected shape attribute to be []int")
		} else {
			expectedShape := []int{2, 2}
			if !shapeEqual(shape, expectedShape) {
				t.Errorf("Shape attribute: expected %v, got %v", expectedShape, shape)
			}
		}

		dtype, ok := attrs["dtype"].(string)
		if !ok {
			t.Error("Expected dtype attribute to be string")
		} else if dtype != "float32" {
			t.Errorf("Expected dtype 'float32', got '%s'", dtype)
		}
	})

	t.Run("parameters_empty", func(t *testing.T) {
		data := []float32{1.0}
		layer, err := NewConstantFromData[float32]("test_constant", engine, ops, []int{1}, data)
		if err != nil {
			t.Fatalf("Failed to create Constant layer: %v", err)
		}

		params := layer.Parameters()
		if len(params) != 0 {
			t.Errorf("Expected 0 parameters for constant layer, got %d", len(params))
		}
	})
}

func TestConstantUtilityMethods(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	t.Run("scalar_tensor", func(t *testing.T) {
		data := []float32{42.0}
		layer, err := NewConstantFromData[float32]("scalar", engine, ops, []int{}, data)
		if err != nil {
			t.Fatalf("Failed to create scalar Constant layer: %v", err)
		}

		if !layer.IsScalar() {
			t.Error("Expected scalar tensor to be identified as scalar")
		}

		if layer.IsVector() {
			t.Error("Expected scalar tensor to not be identified as vector")
		}

		if layer.IsMatrix() {
			t.Error("Expected scalar tensor to not be identified as matrix")
		}

		if layer.NumElements() != 1 {
			t.Errorf("Expected 1 element for scalar, got %d", layer.NumElements())
		}
	})

	t.Run("vector_tensor", func(t *testing.T) {
		data := []float32{1.0, 2.0, 3.0}
		layer, err := NewConstantFromData[float32]("vector", engine, ops, []int{3}, data)
		if err != nil {
			t.Fatalf("Failed to create vector Constant layer: %v", err)
		}

		if layer.IsScalar() {
			t.Error("Expected vector tensor to not be identified as scalar")
		}

		if !layer.IsVector() {
			t.Error("Expected vector tensor to be identified as vector")
		}

		if layer.IsMatrix() {
			t.Error("Expected vector tensor to not be identified as matrix")
		}

		if layer.NumElements() != 3 {
			t.Errorf("Expected 3 elements for vector, got %d", layer.NumElements())
		}
	})

	t.Run("matrix_tensor", func(t *testing.T) {
		data := []float32{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}
		layer, err := NewConstantFromData[float32]("matrix", engine, ops, []int{2, 3}, data)
		if err != nil {
			t.Fatalf("Failed to create matrix Constant layer: %v", err)
		}

		if layer.IsScalar() {
			t.Error("Expected matrix tensor to not be identified as scalar")
		}

		if layer.IsVector() {
			t.Error("Expected matrix tensor to not be identified as vector")
		}

		if !layer.IsMatrix() {
			t.Error("Expected matrix tensor to be identified as matrix")
		}

		if layer.NumElements() != 6 {
			t.Errorf("Expected 6 elements for matrix, got %d", layer.NumElements())
		}
	})

	t.Run("name_methods", func(t *testing.T) {
		data := []float32{1.0}
		layer, err := NewConstantFromData[float32]("original_name", engine, ops, []int{1}, data)
		if err != nil {
			t.Fatalf("Failed to create Constant layer: %v", err)
		}

		if layer.Name() != "original_name" {
			t.Errorf("Expected name 'original_name', got '%s'", layer.Name())
		}

		layer.SetName("new_name")
		if layer.Name() != "new_name" {
			t.Errorf("Expected name 'new_name' after SetName, got '%s'", layer.Name())
		}
	})

	t.Run("string_representation", func(t *testing.T) {
		data := []float32{1.0, 2.0, 3.0, 4.0}
		layer, err := NewConstantFromData[float32]("test", engine, ops, []int{2, 2}, data)
		if err != nil {
			t.Fatalf("Failed to create Constant layer: %v", err)
		}

		str := layer.String()
		expected := "Constant([2 2])"
		if str != expected {
			t.Errorf("Expected string '%s', got '%s'", expected, str)
		}
	})
}

func TestConstantUINT8(t *testing.T) {
	ops := numeric.Uint8Ops{}
	engine := compute.NewCPUEngine[uint8](ops)

	t.Run("uint8_constant", func(t *testing.T) {
		data := []uint8{10, 20, 30, 255}
		layer, err := NewConstantFromData[uint8]("uint8_constant", engine, ops, []int{2, 2}, data)
		if err != nil {
			t.Fatalf("Failed to create UINT8 Constant layer: %v", err)
		}

		attrs := layer.Attributes()
		dtype, ok := attrs["dtype"].(string)
		if !ok {
			t.Error("Expected dtype attribute to be string")
		} else if dtype != "uint8" {
			t.Errorf("Expected dtype 'uint8', got '%s'", dtype)
		}

		// Test forward pass
		ctx := context.Background()
		output, err := layer.Forward(ctx)
		if err != nil {
			t.Fatalf("Forward pass failed: %v", err)
		}

		outputData := output.Data()
		for i, expected := range data {
			if outputData[i] != expected {
				t.Errorf("Output data %d: expected %d, got %d", i, expected, outputData[i])
			}
		}
	})
}

// Helper function to compare slices
func shapeEqual(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}