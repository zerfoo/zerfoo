package tensor

import (
	"testing"
)

// Ones creates a slice of the given size filled with ones.
func Ones[T Numeric](size int) []T {
	data := make([]T, size)
	for i := range data {
		var one T
		switch any(one).(type) {
		case float32:
			one = any(float32(1)).(T)
		case float64:
			one = any(float64(1)).(T)
		case int:
			one = any(int(1)).(T)
		case int8:
			one = any(int8(1)).(T)
		case int16:
			one = any(int16(1)).(T)
		case int32:
			one = any(int32(1)).(T)
		case int64:
			one = any(int64(1)).(T)
		case uint:
			one = any(uint(1)).(T)
		case uint8:
			one = any(uint8(1)).(T)
		case uint32:
			one = any(uint32(1)).(T)
		case uint64:
			one = any(uint64(1)).(T)
		}
		data[i] = one
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
		var diff float64
		switch any(expected.data[i]).(type) {
		case float32:
			diff = float64(any(expected.data[i]).(float32) - any(actual.data[i]).(float32))
		case float64:
			diff = any(expected.data[i]).(float64) - any(actual.data[i]).(float64)
		case int:
			diff = float64(any(expected.data[i]).(int) - any(actual.data[i]).(int))
		case int8:
			diff = float64(any(expected.data[i]).(int8) - any(actual.data[i]).(int8))
		case int16:
			diff = float64(any(expected.data[i]).(int16) - any(actual.data[i]).(int16))
		case int32:
			diff = float64(any(expected.data[i]).(int32) - any(actual.data[i]).(int32))
		case int64:
			diff = float64(any(expected.data[i]).(int64) - any(actual.data[i]).(int64))
		case uint:
			diff = float64(any(expected.data[i]).(uint) - any(actual.data[i]).(uint))
		case uint8:
			diff = float64(any(expected.data[i]).(uint8) - any(actual.data[i]).(uint8))
		case uint32:
			diff = float64(any(expected.data[i]).(uint32) - any(actual.data[i]).(uint32))
		case uint64:
			diff = float64(any(expected.data[i]).(uint64) - any(actual.data[i]).(uint64))
		}
		if diff > tolerance || diff < -tolerance {
			t.Errorf("Expected tensor %v, got %v", expected, actual)

			return
		}
	}
}
