package compute

import (
	"context"
	"math"
	"testing"

	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
)

// CompareTensorsApprox checks if two tensors are approximately equal element-wise.
func CompareTensorsApprox[T tensor.Numeric](t *testing.T, actual, expected *tensor.Tensor[T], epsilon T) bool {
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

func TestCPUEngine_BinaryOp_Broadcast(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := NewCPUEngine[float32](ops)
	ctx := context.Background()

	testCases := []struct {
		name     string
		aShape   []int
		bShape   []int
		aData    []float32
		bData    []float32
		expected []float32
		expShape []int
	}{
		{
			name:     "3D_x_1D",
			aShape:   []int{2, 2, 3},
			bShape:   []int{3},
			aData:    []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
			bData:    []float32{10, 20, 30},
			expected: []float32{11, 22, 33, 14, 25, 36, 17, 28, 39, 20, 31, 42},
			expShape: []int{2, 2, 3},
		},
		{
			name:     "1D_x_3D",
			aShape:   []int{3},
			bShape:   []int{2, 2, 3},
			aData:    []float32{10, 20, 30},
			bData:    []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
			expected: []float32{11, 22, 33, 14, 25, 36, 17, 28, 39, 20, 31, 42},
			expShape: []int{2, 2, 3},
		},
		{
			name:     "2D_x_1D",
			aShape:   []int{2, 3},
			bShape:   []int{3},
			aData:    []float32{1, 2, 3, 4, 5, 6},
			bData:    []float32{10, 20, 30},
			expected: []float32{11, 22, 33, 14, 25, 36},
			expShape: []int{2, 3},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			a, _ := tensor.New[float32](tc.aShape, tc.aData)
			b, _ := tensor.New[float32](tc.bShape, tc.bData)

			result, err := engine.binaryOp(ctx, a, b, ops.Add)
			if err != nil {
				t.Fatalf("binaryOp failed: %v", err)
			}

			expectedTensor, _ := tensor.New[float32](tc.expShape, tc.expected)
			if !CompareTensorsApprox(t, result, expectedTensor, 1e-6) {
				t.Errorf("Result tensor does not match expected. Got %v, want %v", result.Data(), expectedTensor.Data())
			}
		})
	}
}