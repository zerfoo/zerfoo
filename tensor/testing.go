package tensor

import (
	"math"
	"testing"
)

// CompareTensorsApprox checks if two tensors are approximately equal element-wise.
func CompareTensorsApprox[T Numeric](t *testing.T, actual, expected *Tensor[T], epsilon T) bool {
	t.Helper()
	if !actual.ShapeEquals(expected) {
		t.Errorf("tensor shapes do not match: actual %v, expected %v", actual.Shape(), expected.Shape())
		return false
	}

	actualData := actual.Data()
	expectedData := expected.Data()

	if len(actualData) != len(expectedData) {
		t.Errorf("tensor data lengths do not match: actual %d, expected %d", len(actualData), len(expectedData))
		return false
	}

	for i := range actualData {
		if math.Abs(float64(actualData[i])-float64(expectedData[i])) > float64(epsilon) {
			t.Errorf("tensor elements at index %d are not approximately equal: actual %v, expected %v, epsilon %v", i, actualData[i], expectedData[i], epsilon)
			return false
		}
	}
	return true
}