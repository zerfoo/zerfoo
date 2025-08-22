package tensor

import (
	"testing"
)

// Ones creates a slice of the given size filled with ones.
func Ones[T Numeric](size int) []T {
	data := make([]T, size)
	for i := range data {
		data[i] = 1
	}

	return data
}

// Equals checks if two tensors are equal.
func Equals[T Numeric](a, b *TensorNumeric[T]) bool {
	if !a.ShapeEquals(b) {
		return false
	}
	for i := range a.data {
		if a.data[i] != b.data[i] {
			return false
		}
	}

	return true
}

// AssertEquals checks if two tensors are equal and fails the test if they are not.
func AssertEquals[T Numeric](t *testing.T, expected, actual *TensorNumeric[T]) {
	if !Equals(expected, actual) {
		t.Errorf("Expected tensor %v, got %v", expected, actual)
	}
}

// AssertClose checks if two tensors are close enough and fails the test if they are not.
func AssertClose[T Numeric](t *testing.T, expected, actual *TensorNumeric[T], tolerance float64) {
	if !expected.ShapeEquals(actual) {
		t.Errorf("Expected shape %v, got %v", expected.Shape(), actual.Shape())

		return
	}
	for i := range expected.data {
		diff := expected.data[i] - actual.data[i]
		if float64(diff) > tolerance || float64(diff) < -tolerance {
			t.Errorf("Expected tensor %v, got %v", expected, actual)

			return
		}
	}
}
