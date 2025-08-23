package compute

import (
	"context"
	"testing"

	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
)

// TestFailableTensorSetError tests the previously unreachable MatMul error path.
func TestFailableTensorSetError(t *testing.T) {
	engine := NewTestableEngine[float32](numeric.Float32Ops{})
	ctx := context.Background()

	// Test MatMul Set error path
	t.Run("MatMul_SetError", func(t *testing.T) {
		a, _ := tensor.New[float32]([]int{2, 2}, []float32{1, 2, 3, 4})
		b, _ := tensor.New[float32]([]int{2, 2}, []float32{5, 6, 7, 8})
		result, _ := tensor.New[float32]([]int{2, 2}, []float32{0, 0, 0, 0})

		// Wrap result in FailableTensor and configure it to fail on Set
		failableResult := NewFailableTensor(result)
		failableResult.SetFailOnSet(true)

		// This should now trigger the error path in MatMul
		err := engine.TestableMatMul(ctx, a, b, failableResult)
		if err == nil {
			t.Error("expected error from Set operation, got nil")
		}

		if err.Error() != "controlled failure: Set operation failed" {
			t.Errorf("expected controlled failure error, got: %v", err)
		}
	})

	// Test MatMul Set error after specific number of calls
	t.Run("MatMul_SetErrorAfterCalls", func(t *testing.T) {
		a, _ := tensor.New[float32]([]int{2, 2}, []float32{1, 2, 3, 4})
		b, _ := tensor.New[float32]([]int{2, 2}, []float32{5, 6, 7, 8})
		result, _ := tensor.New[float32]([]int{2, 2}, []float32{0, 0, 0, 0})

		// Configure to fail after 2 Set calls
		failableResult := NewFailableTensor(result)
		failableResult.SetFailOnSetAfter(2)

		// This should trigger the error path after 2 successful Set operations
		err := engine.TestableMatMul(ctx, a, b, failableResult)
		if err == nil {
			t.Error("expected error from Set operation after 2 calls, got nil")
		}
	})

	// Test Transpose Set error path
	t.Run("Transpose_SetError", func(t *testing.T) {
		a, _ := tensor.New[float32]([]int{2, 3}, []float32{1, 2, 3, 4, 5, 6})
		result, _ := tensor.New[float32]([]int{3, 2}, []float32{0, 0, 0, 0, 0, 0})

		// Configure to fail on Set
		failableResult := NewFailableTensor(result)
		failableResult.SetFailOnSet(true)

		// This should trigger the error path in Transpose
		err := engine.TestableTranspose(ctx, a, failableResult)
		if err == nil {
			t.Error("expected error from Set operation, got nil")
		}

		if err.Error() != "controlled failure: Set operation failed" {
			t.Errorf("expected controlled failure error, got: %v", err)
		}
	})

	// Test Transpose Set error after specific calls
	t.Run("Transpose_SetErrorAfterCalls", func(t *testing.T) {
		a, _ := tensor.New[float32]([]int{2, 3}, []float32{1, 2, 3, 4, 5, 6})
		result, _ := tensor.New[float32]([]int{3, 2}, []float32{0, 0, 0, 0, 0, 0})

		// Configure to fail after 3 Set calls
		failableResult := NewFailableTensor(result)
		failableResult.SetFailOnSetAfter(3)

		// This should trigger the error path after 3 successful Set operations
		err := engine.TestableTranspose(ctx, a, failableResult)
		if err == nil {
			t.Error("expected error from Set operation after 3 calls, got nil")
		}
	})
}

// TestFailableZeroerError tests the previously unreachable Sum Zero error path.
func TestFailableZeroerError(t *testing.T) {
	engine := NewTestableEngine[float32](numeric.Float32Ops{})
	ctx := context.Background()

	// Test Sum Zero error path
	t.Run("Sum_ZeroError", func(t *testing.T) {
		a, _ := tensor.New[float32]([]int{2, 3}, []float32{1, 2, 3, 4, 5, 6})
		result, _ := tensor.New[float32]([]int{1}, []float32{0})

		// Create a FailableZeroer and configure it to fail
		zeroer := NewFailableZeroer(engine)
		zeroer.SetFailOnZero(true)

		// This should trigger the error path in Sum
		err := engine.TestableSum(ctx, a, -1, false, zeroer, result)
		if err == nil {
			t.Error("expected error from Zero operation, got nil")
		}

		if err.Error() != "controlled failure: Zero operation failed" {
			t.Errorf("expected controlled failure error, got: %v", err)
		}
	})

	// Test successful Sum operation with working zeroer
	t.Run("Sum_ZeroSuccess", func(t *testing.T) {
		a, _ := tensor.New[float32]([]int{2, 3}, []float32{1, 2, 3, 4, 5, 6})
		result, _ := tensor.New[float32]([]int{1}, []float32{0})

		// Create a FailableZeroer but don't configure it to fail
		zeroer := NewFailableZeroer(engine)
		zeroer.SetFailOnZero(false)

		// This should succeed
		err := engine.TestableSum(ctx, a, -1, false, zeroer, result)
		if err != nil {
			t.Errorf("unexpected error: %v", err)
		}
	})
}

// TestTestableEngineErrorHandling tests various error conditions.
func TestTestableEngineErrorHandling(t *testing.T) {
	engine := NewTestableEngine[float32](numeric.Float32Ops{})
	ctx := context.Background()

	// Test MatMul with nil inputs
	t.Run("TestableMatMul_NilInputs", func(t *testing.T) {
		result, _ := tensor.New[float32]([]int{2, 2}, []float32{0, 0, 0, 0})
		failableResult := NewFailableTensor(result)

		err := engine.TestableMatMul(ctx, nil, nil, failableResult)
		if err == nil {
			t.Error("expected error for nil inputs")
		}
	})

	// Test MatMul with wrong result shape
	t.Run("TestableMatMul_WrongResultShape", func(t *testing.T) {
		a, _ := tensor.New[float32]([]int{2, 3}, []float32{1, 2, 3, 4, 5, 6})
		b, _ := tensor.New[float32]([]int{3, 2}, []float32{1, 2, 3, 4, 5, 6})
		wrongResult, _ := tensor.New[float32]([]int{3, 3}, make([]float32, 9))
		failableResult := NewFailableTensor(wrongResult)

		err := engine.TestableMatMul(ctx, a, b, failableResult)
		if err == nil {
			t.Error("expected error for wrong result shape")
		}
	})

	// Test Transpose with nil input
	t.Run("TestableTranspose_NilInput", func(t *testing.T) {
		result, _ := tensor.New[float32]([]int{3, 2}, make([]float32, 6))
		failableResult := NewFailableTensor(result)

		err := engine.TestableTranspose(ctx, nil, failableResult)
		if err == nil {
			t.Error("expected error for nil input")
		}
	})

	// Test Transpose with wrong result shape
	t.Run("TestableTranspose_WrongResultShape", func(t *testing.T) {
		a, _ := tensor.New[float32]([]int{2, 3}, []float32{1, 2, 3, 4, 5, 6})
		wrongResult, _ := tensor.New[float32]([]int{2, 2}, make([]float32, 4))
		failableResult := NewFailableTensor(wrongResult)

		err := engine.TestableTranspose(ctx, a, failableResult)
		if err == nil {
			t.Error("expected error for wrong result shape")
		}
	})

	// Test Sum with nil inputs
	t.Run("TestableSum_NilInputs", func(t *testing.T) {
		result, _ := tensor.New[float32]([]int{1}, []float32{0})
		zeroer := NewFailableZeroer(engine)

		err := engine.TestableSum(ctx, nil, -1, false, zeroer, result)
		if err == nil {
			t.Error("expected error for nil input tensor")
		}

		a, _ := tensor.New[float32]([]int{2, 3}, []float32{1, 2, 3, 4, 5, 6})

		err = engine.TestableSum(ctx, a, -1, false, nil, result)
		if err == nil {
			t.Error("expected error for nil zeroer")
		}

		err = engine.TestableSum(ctx, a, -1, false, zeroer, nil)
		if err == nil {
			t.Error("expected error for nil result tensor")
		}
	})
}

// TestFailableTensorNormalOperation tests that FailableTensor works normally when not configured to fail.
func TestFailableTensorNormalOperation(t *testing.T) {
	engine := NewTestableEngine[float32](numeric.Float32Ops{})
	ctx := context.Background()

	// Test normal MatMul operation
	t.Run("MatMul_NormalOperation", func(t *testing.T) {
		a, _ := tensor.New[float32]([]int{2, 2}, []float32{1, 2, 3, 4})
		b, _ := tensor.New[float32]([]int{2, 2}, []float32{5, 6, 7, 8})
		result, _ := tensor.New[float32]([]int{2, 2}, []float32{0, 0, 0, 0})

		// Use FailableTensor but don't configure it to fail
		failableResult := NewFailableTensor(result)

		err := engine.TestableMatMul(ctx, a, b, failableResult)
		if err != nil {
			t.Errorf("unexpected error: %v", err)
		}

		// Verify the result is correct
		expectedData := []float32{19, 22, 43, 50}

		resultData := failableResult.Data()
		for i, expected := range expectedData {
			if resultData[i] != expected {
				t.Errorf("expected data[%d] = %f, got %f", i, expected, resultData[i])
			}
		}
	})

	// Test normal Transpose operation
	t.Run("Transpose_NormalOperation", func(t *testing.T) {
		a, _ := tensor.New[float32]([]int{2, 3}, []float32{1, 2, 3, 4, 5, 6})
		result, _ := tensor.New[float32]([]int{3, 2}, []float32{0, 0, 0, 0, 0, 0})

		// Use FailableTensor but don't configure it to fail
		failableResult := NewFailableTensor(result)

		err := engine.TestableTranspose(ctx, a, failableResult)
		if err != nil {
			t.Errorf("unexpected error: %v", err)
		}

		// Verify the result is correct
		expectedData := []float32{1, 4, 2, 5, 3, 6}

		resultData := failableResult.Data()
		for i, expected := range expectedData {
			if resultData[i] != expected {
				t.Errorf("expected data[%d] = %f, got %f", i, expected, resultData[i])
			}
		}
	})
}
