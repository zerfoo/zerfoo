package graph

import (
	"errors"
	"testing"

	"github.com/zerfoo/zerfoo/tensor"
	"github.com/zerfoo/zerfoo/testing/testutils"
)

func TestNewParameter(t *testing.T) {
	value, _ := tensor.New[int]([]int{2, 2}, nil)

	t.Run("successful creation", func(t *testing.T) {
		param, err := NewParameter("test", value, tensor.New[int])
		testutils.AssertNoError(t, err, "expected no error, got %v")
		testutils.AssertNotNil(t, param, "expected parameter to not be nil")
		testutils.AssertEqual(t, "test", param.Name, "expected name %q, got %q")
		testutils.AssertNotNil(t, param.Value, "expected value to not be nil")
		testutils.AssertNotNil(t, param.Gradient, "expected gradient to not be nil")
		testutils.AssertTrue(t, testutils.IntSliceEqual(value.Shape(), param.Gradient.Shape()), "expected gradient shape to match value shape")
	})

	t.Run("nil tensor", func(t *testing.T) {
		_, err := NewParameter[int]("test", nil, tensor.New[int])
		testutils.AssertError(t, err, "expected an error, got nil")
	})

	t.Run("empty name", func(t *testing.T) {
		_, err := NewParameter("", value, tensor.New[int])
		testutils.AssertError(t, err, "expected an error for empty name")
	})

	t.Run("tensor creation fails", func(t *testing.T) {
		mockErr := errors.New("mock error")
		mockNewTensorFn := func(_ []int, _ []int) (*tensor.TensorNumeric[int], error) {
			return nil, mockErr
		}
		_, err := NewParameter("test", value, mockNewTensorFn)
		testutils.AssertError(t, err, "expected an error, got nil")
		testutils.AssertEqual(t, mockErr, err, "expected error %v, got %v")
	})
}

func TestParameter_AddGradient(t *testing.T) {
	value, _ := tensor.New[float32]([]int{2, 2}, []float32{1.0, 2.0, 3.0, 4.0})
	param, err := NewParameter("test", value, tensor.New[float32])
	testutils.AssertNoError(t, err, "Failed to create parameter")

	t.Run("successful gradient addition", func(t *testing.T) {
		// Clear gradient first
		param.ClearGradient()
		
		grad, _ := tensor.New[float32]([]int{2, 2}, []float32{0.1, 0.2, 0.3, 0.4})
		err := param.AddGradient(grad)
		testutils.AssertNoError(t, err, "AddGradient should not error")
		
		// Check that gradient was added
		expectedGrad := []float32{0.1, 0.2, 0.3, 0.4}
		for i, expected := range expectedGrad {
			testutils.AssertFloatEqual(t, expected, param.Gradient.Data()[i], 1e-6, "Gradient should match expected value")
		}
		
		// Add another gradient to test accumulation
		grad2, _ := tensor.New[float32]([]int{2, 2}, []float32{0.1, 0.1, 0.1, 0.1})
		err = param.AddGradient(grad2)
		testutils.AssertNoError(t, err, "Second AddGradient should not error")
		
		// Check accumulated gradient
		expectedAccumulated := []float32{0.2, 0.3, 0.4, 0.5}
		for i, expected := range expectedAccumulated {
			testutils.AssertFloatEqual(t, expected, param.Gradient.Data()[i], 1e-6, "Accumulated gradient should match expected value")
		}
	})

	t.Run("nil gradient tensor", func(t *testing.T) {
		paramNilGrad := &Parameter[float32]{
			Name:     "test",
			Value:    value,
			Gradient: nil,
		}
		grad, _ := tensor.New[float32]([]int{2, 2}, []float32{0.1, 0.2, 0.3, 0.4})
		err := paramNilGrad.AddGradient(grad)
		testutils.AssertError(t, err, "AddGradient should error with nil gradient")
	})

	t.Run("gradient shape mismatch", func(t *testing.T) {
		grad, _ := tensor.New[float32]([]int{3, 2}, []float32{0.1, 0.2, 0.3, 0.4, 0.5, 0.6})
		err := param.AddGradient(grad)
		testutils.AssertError(t, err, "AddGradient should error with shape mismatch")
	})
}

func TestParameter_ClearGradient(t *testing.T) {
	value, _ := tensor.New[float32]([]int{2, 2}, []float32{1.0, 2.0, 3.0, 4.0})
	param, err := NewParameter("test", value, tensor.New[float32])
	testutils.AssertNoError(t, err, "Failed to create parameter")

	// Set some gradient values
	for i := range param.Gradient.Data() {
		param.Gradient.Data()[i] = float32(i + 1)
	}

	// Clear gradient
	param.ClearGradient()

	// Check that all gradient values are zero
	for _, v := range param.Gradient.Data() {
		testutils.AssertFloatEqual(t, 0.0, v, 1e-6, "Gradient should be zero after clearing")
	}
}

func TestParameter_EdgeCases(t *testing.T) {
	t.Run("scalar parameter", func(t *testing.T) {
		value, _ := tensor.New[float32]([]int{}, []float32{5.0})
		param, err := NewParameter("scalar", value, tensor.New[float32])
		testutils.AssertNoError(t, err, "Failed to create scalar parameter")
		
		testutils.AssertEqual(t, "scalar", param.Name, "Parameter name should match")
		testutils.AssertNotNil(t, param.Gradient, "Gradient should not be nil")
		testutils.AssertTrue(t, testutils.IntSliceEqual([]int{}, param.Gradient.Shape()), "Gradient should have scalar shape")
		
		// Test clearing scalar gradient
		param.Gradient.Data()[0] = 1.5
		param.ClearGradient()
		testutils.AssertFloatEqual(t, 0.0, param.Gradient.Data()[0], 1e-6, "Scalar gradient should be zero after clearing")
	})

	t.Run("large parameter", func(t *testing.T) {
		value, _ := tensor.New[int]([]int{10, 10}, nil)
		param, err := NewParameter("large", value, tensor.New[int])
		testutils.AssertNoError(t, err, "Failed to create large parameter")
		
		// Set gradient values
		for i := range param.Gradient.Data() {
			param.Gradient.Data()[i] = i
		}
		
		// Add gradient
		grad, _ := tensor.New[int]([]int{10, 10}, nil)
		for i := range grad.Data() {
			grad.Data()[i] = 1
		}
		
		err = param.AddGradient(grad)
		testutils.AssertNoError(t, err, "AddGradient should not error for large parameter")
		
		// Check some values
		testutils.AssertEqual(t, 1, param.Gradient.Data()[0], "First gradient should be 1")
		testutils.AssertEqual(t, 50, param.Gradient.Data()[49], "50th gradient should be 50")
	})
}
