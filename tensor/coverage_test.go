package tensor

import (
	"testing"
)

// TestCoverageSpecific tests specific uncovered code paths to achieve 100% coverage
func TestCoverageSpecific(t *testing.T) {
	// Test case 1: Create a 0-dimensional view tensor manually
	// Since Slice() cannot create 0-dimensional views, we need to create one manually
	// to test the Data() method's 0-dimensional view branch

	// Create a regular tensor first
	baseTensor, _ := New[int]([]int{1}, []int{42})

	// Manually create a view that simulates a 0-dimensional view
	// This is the only way to test the specific branch in Data() for 0-dimensional views
	viewTensor := &Tensor[int]{
		shape:   []int{}, // 0-dimensional
		strides: []int{}, // 0-dimensional
		data:    baseTensor.data,
		isView:  true, // This is the key - it's a view
	}

	// Test the Data() method on this 0-dimensional view
	// This should trigger the specific branch: if t.isView && t.Dims() == 0
	data := viewTensor.Data()
	expected := []int{42}
	if len(data) != 1 || data[0] != 42 {
		t.Errorf("expected data %v, got %v", expected, data)
	}

	// Test case 2: Test eachRecursive with a tensor that has 0 dimensions
	// Create a tensor with empty shape (0-dimensional)
	zeroD, _ := New[int]([]int{}, []int{99})

	// Call Each which internally calls eachRecursive
	// This should test the eachRecursive branch: if t.Dims() == 0
	var callCount int
	var receivedValue int
	zeroD.Each(func(val int) {
		callCount++
		receivedValue = val
	})

	// For a 0-dimensional tensor, Each should call the function once
	if callCount != 1 {
		t.Errorf("expected Each to be called once for 0-dimensional tensor, got %d", callCount)
	}
	if receivedValue != 99 {
		t.Errorf("expected value 99, got %d", receivedValue)
	}

	// Test case 3: Test eachRecursive with a tensor that has size 0 but dimensions > 0
	// This tests a different branch in eachRecursive
	zeroSizeTensor, _ := New[int]([]int{2, 0, 3}, nil)

	var zeroSizeCallCount int
	zeroSizeTensor.Each(func(val int) {
		zeroSizeCallCount++
	})

	// For a tensor with size 0, Each should not call the function at all
	if zeroSizeCallCount != 0 {
		t.Errorf("expected Each to not be called for zero-size tensor, got %d calls", zeroSizeCallCount)
	}

	// Test case 4: Test eachRecursive directly with a manually created 0-dimensional tensor
	// to trigger the specific uncovered branch in eachRecursive
	// Since Each() handles 0-dimensional tensors directly, we need to call eachRecursive manually
	zeroDTensor := &Tensor[int]{
		shape:   []int{}, // 0-dimensional
		strides: []int{}, // 0-dimensional
		data:    []int{123},
		isView:  false,
	}

	// Call eachRecursive directly to test the "if t.Dims() == 0" branch
	var directCallCount int
	zeroDTensor.eachRecursive([]int{}, 0, func(val int) {
		directCallCount++
	})

	// For a 0-dimensional tensor, eachRecursive should return early and not call the function
	if directCallCount != 0 {
		t.Errorf("expected eachRecursive to not call function for 0-dimensional tensor, got %d calls", directCallCount)
	}
}
