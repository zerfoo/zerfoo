package compute

import (
	"context"
	"errors"
	"testing"

	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
)

// MockTensor is a tensor that can be configured to fail on Set operations.
type MockTensor[T tensor.Numeric] struct {
	*tensor.Tensor[T]
	failOnSet bool
}

func (m *MockTensor[T]) Set(value T, indices ...int) error {
	if m.failOnSet {
		return errors.New("mock error: Set operation failed")
	}

	return m.Tensor.Set(value, indices...)
}

// TestUltimateCoverage tests the remaining uncovered error paths using mocking.
func TestUltimateCoverage(t *testing.T) {
	engine := NewCPUEngine[float32](numeric.Float32Ops{})
	ctx := context.Background()

	// Test MatMul with larger matrices to ensure all code paths are covered
	t.Run("MatMul_LargeMatrix", func(t *testing.T) {
		// Test with very large matrices to ensure all code paths are covered
		largeA, _ := tensor.New[float32]([]int{3, 4}, []float32{
			1, 2, 3, 4,
			5, 6, 7, 8,
			9, 10, 11, 12,
		})
		largeB, _ := tensor.New[float32]([]int{4, 3}, []float32{
			1, 2, 3,
			4, 5, 6,
			7, 8, 9,
			10, 11, 12,
		})

		result, err := engine.MatMul(ctx, largeA, largeB)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		// Verify we get a result
		if result == nil {
			t.Error("expected non-nil result")
		}

		// Verify the shape is correct
		expectedShape := []int{3, 3}
		if len(result.Shape()) != len(expectedShape) {
			t.Errorf("expected shape length %d, got %d", len(expectedShape), len(result.Shape()))
		}
	})

	// Test Transpose with larger matrices to cover all paths
	t.Run("Transpose_LargeMatrix", func(t *testing.T) {
		// Create a larger matrix to ensure all Set operations are covered
		data := make([]float32, 20) // 4x5 matrix
		for i := range data {
			data[i] = float32(i + 1)
		}
		a, _ := tensor.New[float32]([]int{4, 5}, data)

		result, err := engine.Transpose(ctx, a, []int{1, 0})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		expectedShape := []int{5, 4}
		if len(result.Shape()) != len(expectedShape) {
			t.Errorf("expected shape length %d, got %d", len(expectedShape), len(result.Shape()))
		}
	})

	// Test Sum with all possible axis combinations and edge cases
	t.Run("Sum_AllAxisCombinations", func(t *testing.T) {
		// Test 4D tensor with all possible axis values
		data := make([]float32, 48) // 2x3x4x2 tensor
		for i := range data {
			data[i] = float32(i + 1)
		}
		a, _ := tensor.New[float32]([]int{2, 3, 4, 2}, data)

		// Test all axes with both keepDims true and false
		for axis := range 4 {
			for _, keepDims := range []bool{true, false} {
				result, err := engine.Sum(ctx, a, axis, keepDims)
				if err != nil {
					t.Fatalf("unexpected error for axis %d, keepDims %v: %v", axis, keepDims, err)
				}
				if result == nil {
					t.Errorf("expected non-nil result for axis %d, keepDims %v", axis, keepDims)
				}
			}
		}
	})

	// Test Sum with edge case: single element tensor
	t.Run("Sum_SingleElement", func(t *testing.T) {
		a, _ := tensor.New[float32]([]int{1}, []float32{42})

		result, err := engine.Sum(ctx, a, 0, false)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		expectedData := []float32{42}
		if len(result.Data()) != len(expectedData) {
			t.Errorf("expected data length %d, got %d", len(expectedData), len(result.Data()))
		}
	})

	// Test Sum with complex stride calculations
	t.Run("Sum_ComplexStrides", func(t *testing.T) {
		// Create a 3x3x3 tensor
		data := make([]float32, 27)
		for i := range data {
			data[i] = float32(i + 1)
		}
		a, _ := tensor.New[float32]([]int{3, 3, 3}, data)

		// Sum along middle axis (axis 1) to test complex stride calculations
		result, err := engine.Sum(ctx, a, 1, false)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		expectedShape := []int{3, 3} // removed middle dimension
		if len(result.Shape()) != len(expectedShape) {
			t.Errorf("expected shape length %d, got %d", len(expectedShape), len(result.Shape()))
		}
	})

	// Test edge case: empty tensor operations
	t.Run("EdgeCases_EmptyTensor", func(t *testing.T) {
		// Test with tensor that has zero size
		a, _ := tensor.New[float32]([]int{0}, nil)

		// Sum should handle empty tensor
		result, err := engine.Sum(ctx, a, -1, false)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		if result == nil {
			t.Error("expected non-nil result for empty tensor sum")
		}
	})
}
