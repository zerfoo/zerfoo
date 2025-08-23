package tensor

import (
	"reflect"
	"testing"
)

func TestReshape(t *testing.T) {
	tensor, _ := New([]int{2, 6}, []int{
		0, 1, 2, 3, 4, 5,
		6, 7, 8, 9, 10, 11,
	})

	t.Run("ValidReshape", func(t *testing.T) {
		reshaped, err := tensor.Reshape([]int{3, 4})
		if err != nil {
			t.Fatalf("expected no error, got %v", err)
		}

		expectedShape := []int{3, 4}
		if !reflect.DeepEqual(reshaped.Shape(), expectedShape) {
			t.Errorf("expected shape %v, got %v", expectedShape, reshaped.Shape())
		}

		// Check if the view accesses the correct data
		val, _ := reshaped.At(1, 1) // Should be tensor.data[5] -> 5
		if val != 5 {
			t.Errorf("reshaped data is incorrect. Expected 5, got %d", val)
		}
	})

	t.Run("InferredDimension", func(t *testing.T) {
		reshaped, err := tensor.Reshape([]int{4, -1})
		if err != nil {
			t.Fatalf("expected no error, got %v", err)
		}

		expectedShape := []int{4, 3}
		if !reflect.DeepEqual(reshaped.Shape(), expectedShape) {
			t.Errorf("expected shape %v, got %v", expectedShape, reshaped.Shape())
		}
	})

	t.Run("InvalidReshape_SizeMismatch", func(t *testing.T) {
		_, err := tensor.Reshape([]int{3, 5})
		if err == nil {
			t.Fatal("expected an error for size mismatch, got nil")
		}
	})

	t.Run("InvalidReshape_MultipleInferred", func(t *testing.T) {
		_, err := tensor.Reshape([]int{-1, -1})
		if err == nil {
			t.Fatal("expected an error for multiple inferred dimensions, got nil")
		}
	})

	t.Run("InvalidReshape_CannotInfer", func(t *testing.T) {
		tensor, _ := New[int]([]int{2, 5}, nil)

		_, err := tensor.Reshape([]int{3, -1})
		if err == nil {
			t.Fatal("expected an error for cannot infer dimension, got nil")
		}
	})

	t.Run("InvalidReshape_InvalidDimension", func(t *testing.T) {
		_, err := tensor.Reshape([]int{3, 0})
		if err == nil {
			t.Fatal("expected an error for invalid dimension, got nil")
		}
	})
}
