package graph

import (
	"fmt"
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

	t.Run("tensor creation fails", func(t *testing.T) {
		mockErr := fmt.Errorf("mock error")
		mockNewTensorFn := func(_ []int, _ []int) (*tensor.Tensor[int], error) {
			return nil, mockErr
		}
		_, err := NewParameter("test", value, mockNewTensorFn)
		testutils.AssertError(t, err, "expected an error, got nil")
		testutils.AssertEqual(t, mockErr, err, "expected error %v, got %v")
	})
}
