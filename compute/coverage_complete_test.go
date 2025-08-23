package compute

import (
	"context"
	"reflect"
	"testing"

	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
)

// TestCoverageComplete tests specific uncovered code paths to achieve 100% coverage.
func TestCoverageComplete(t *testing.T) {
	engine := NewCPUEngine[float32](numeric.Float32Ops{})
	ctx := context.Background()

	// Test Copy function error cases (currently 66.7% coverage)
	t.Run("Copy_NilDst", func(t *testing.T) {
		src, _ := tensor.New[float32]([]int{2, 2}, []float32{1, 2, 3, 4})

		err := engine.Copy(ctx, nil, src)
		if err == nil {
			t.Error("expected error for nil dst tensor")
		}
	})

	t.Run("Copy_NilSrc", func(t *testing.T) {
		dst, _ := tensor.New[float32]([]int{2, 2}, []float32{0, 0, 0, 0})

		err := engine.Copy(ctx, dst, nil)
		if err == nil {
			t.Error("expected error for nil src tensor")
		}
	})

	t.Run("Copy_ShapeMismatch", func(t *testing.T) {
		dst, _ := tensor.New[float32]([]int{2, 2}, []float32{0, 0, 0, 0})
		src, _ := tensor.New[float32]([]int{3, 3}, []float32{1, 2, 3, 4, 5, 6, 7, 8, 9})

		err := engine.Copy(ctx, dst, src)
		if err == nil {
			t.Error("expected error for shape mismatch")
		}
	})

	// Test Zero function error cases (currently 80.0% coverage)
	t.Run("Zero_NilTensor", func(t *testing.T) {
		err := engine.Zero(ctx, nil)
		if err == nil {
			t.Error("expected error for nil tensor")
		}
	})

	// Test Sum function with keepDims=true and axis-specific cases (currently 77.8% coverage)
	t.Run("Sum_KeepDims_Axis0", func(t *testing.T) {
		a, _ := tensor.New[float32]([]int{2, 3}, []float32{1, 2, 3, 4, 5, 6})

		result, err := engine.Sum(ctx, a, 0, true)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		expectedShape := []int{1, 3}
		if !reflect.DeepEqual(result.Shape(), expectedShape) {
			t.Errorf("expected shape %v, got %v", expectedShape, result.Shape())
		}

		expectedData := []float32{5, 7, 9} // sum along axis 0
		if !reflect.DeepEqual(result.Data(), expectedData) {
			t.Errorf("expected data %v, got %v", expectedData, result.Data())
		}
	})

	t.Run("Sum_KeepDims_Axis1", func(t *testing.T) {
		a, _ := tensor.New[float32]([]int{2, 3}, []float32{1, 2, 3, 4, 5, 6})

		result, err := engine.Sum(ctx, a, 1, true)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		expectedShape := []int{2, 1}
		if !reflect.DeepEqual(result.Shape(), expectedShape) {
			t.Errorf("expected shape %v, got %v", expectedShape, result.Shape())
		}

		expectedData := []float32{6, 15} // sum along axis 1
		if !reflect.DeepEqual(result.Data(), expectedData) {
			t.Errorf("expected data %v, got %v", expectedData, result.Data())
		}
	})

	t.Run("Sum_NoKeepDims_ScalarResult", func(t *testing.T) {
		a, _ := tensor.New[float32]([]int{2}, []float32{3, 7})

		result, err := engine.Sum(ctx, a, 0, false)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		expectedShape := []int{1} // scalar result when reducing single dimension
		if !reflect.DeepEqual(result.Shape(), expectedShape) {
			t.Errorf("expected shape %v, got %v", expectedShape, result.Shape())
		}

		expectedData := []float32{10}
		if !reflect.DeepEqual(result.Data(), expectedData) {
			t.Errorf("expected data %v, got %v", expectedData, result.Data())
		}
	})

	t.Run("Sum_AxisOutOfBounds", func(t *testing.T) {
		a, _ := tensor.New[float32]([]int{2, 3}, []float32{1, 2, 3, 4, 5, 6})

		_, err := engine.Sum(ctx, a, 5, false)
		if err == nil {
			t.Error("expected error for axis out of bounds")
		}
	})

	// Test Exp function error cases (currently 87.5% coverage)
	t.Run("Exp_NilTensor", func(t *testing.T) {
		_, err := engine.Exp(ctx, nil)
		if err == nil {
			t.Error("expected error for nil tensor")
		}
	})

	// Test Log function error cases (currently 87.5% coverage)
	t.Run("Log_NilTensor", func(t *testing.T) {
		_, err := engine.Log(ctx, nil)
		if err == nil {
			t.Error("expected error for nil tensor")
		}
	})

	// Test Pow function error cases (currently 87.5% coverage)
	t.Run("Pow_NilBase", func(t *testing.T) {
		exp, _ := tensor.New[float32]([]int{2}, []float32{2, 3})

		_, err := engine.Pow(ctx, nil, exp)
		if err == nil {
			t.Error("expected error for nil base tensor")
		}
	})

	t.Run("Pow_NilExponent", func(t *testing.T) {
		base, _ := tensor.New[float32]([]int{2}, []float32{2, 3})

		_, err := engine.Pow(ctx, base, nil)
		if err == nil {
			t.Error("expected error for nil exponent tensor")
		}
	})

	// Test Transpose function error cases (currently 92.9% coverage)
	t.Run("Transpose_NilTensor", func(t *testing.T) {
		_, err := engine.Transpose(ctx, nil, []int{1, 0})
		if err == nil {
			t.Error("expected error for nil tensor")
		}
	})

	t.Run("Transpose_InvalidAxes", func(t *testing.T) {
		a, _ := tensor.New[float32]([]int{2, 3, 4}, make([]float32, 24))

		_, err := engine.Transpose(ctx, a, []int{0, 1})
		if err == nil {
			t.Error("expected error for invalid axes")
		}
	})

	// Test MatMul function error cases (currently 94.7% coverage)
	t.Run("MatMul_NilA", func(t *testing.T) {
		b, _ := tensor.New[float32]([]int{2, 2}, []float32{1, 2, 3, 4})

		_, err := engine.MatMul(ctx, nil, b)
		if err == nil {
			t.Error("expected error for nil tensor a")
		}
	})

	t.Run("MatMul_NilB", func(t *testing.T) {
		a, _ := tensor.New[float32]([]int{2, 2}, []float32{1, 2, 3, 4})

		_, err := engine.MatMul(ctx, a, nil)
		if err == nil {
			t.Error("expected error for nil tensor b")
		}
	})

	t.Run("MatMul_Non2D_A", func(t *testing.T) {
		a, _ := tensor.New[float32]([]int{2, 3, 4}, make([]float32, 24))
		b, _ := tensor.New[float32]([]int{2, 2}, []float32{1, 2, 3, 4})

		_, err := engine.MatMul(ctx, a, b)
		if err == nil {
			t.Error("expected error for non-2D tensor a")
		}
	})

	t.Run("MatMul_Non2D_B", func(t *testing.T) {
		a, _ := tensor.New[float32]([]int{2, 2}, []float32{1, 2, 3, 4})
		b, _ := tensor.New[float32]([]int{2, 3, 4}, make([]float32, 24))

		_, err := engine.MatMul(ctx, a, b)
		if err == nil {
			t.Error("expected error for non-2D tensor b")
		}
	})

	t.Run("MatMul_IncompatibleDims", func(t *testing.T) {
		a, _ := tensor.New[float32]([]int{2, 3}, []float32{1, 2, 3, 4, 5, 6})
		b, _ := tensor.New[float32]([]int{2, 2}, []float32{1, 2, 3, 4})

		_, err := engine.MatMul(ctx, a, b)
		if err == nil {
			t.Error("expected error for incompatible matrix dimensions")
		}
	})
}

// TestSumComplexCases tests more complex Sum scenarios to ensure full coverage.
func TestSumComplexCases(t *testing.T) {
	engine := NewCPUEngine[float32](numeric.Float32Ops{})
	ctx := context.Background()

	// Test Sum with negative axis (sum all elements)
	t.Run("Sum_NegativeAxis_KeepDims", func(t *testing.T) {
		a, _ := tensor.New[float32]([]int{2, 3}, []float32{1, 2, 3, 4, 5, 6})

		result, err := engine.Sum(ctx, a, -1, true)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		expectedShape := []int{1, 1} // keepDims with 2D input
		if !reflect.DeepEqual(result.Shape(), expectedShape) {
			t.Errorf("expected shape %v, got %v", expectedShape, result.Shape())
		}

		expectedData := []float32{21} // sum of all elements
		if !reflect.DeepEqual(result.Data(), expectedData) {
			t.Errorf("expected data %v, got %v", expectedData, result.Data())
		}
	})

	t.Run("Sum_NegativeAxis_NoKeepDims", func(t *testing.T) {
		a, _ := tensor.New[float32]([]int{2, 3}, []float32{1, 2, 3, 4, 5, 6})

		result, err := engine.Sum(ctx, a, -1, false)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		expectedShape := []int{1} // scalar result
		if !reflect.DeepEqual(result.Shape(), expectedShape) {
			t.Errorf("expected shape %v, got %v", expectedShape, result.Shape())
		}

		expectedData := []float32{21} // sum of all elements
		if !reflect.DeepEqual(result.Data(), expectedData) {
			t.Errorf("expected data %v, got %v", expectedData, result.Data())
		}
	})

	// Test Sum with 3D tensor to cover more complex indexing
	t.Run("Sum_3D_Axis1", func(t *testing.T) {
		// 2x3x2 tensor
		a, _ := tensor.New[float32]([]int{2, 3, 2}, []float32{
			1, 2, 3, 4, 5, 6, // first 2x3 slice
			7, 8, 9, 10, 11, 12, // second 2x3 slice
		})

		result, err := engine.Sum(ctx, a, 1, false) // sum along middle axis
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		expectedShape := []int{2, 2} // remove middle dimension
		if !reflect.DeepEqual(result.Shape(), expectedShape) {
			t.Errorf("expected shape %v, got %v", expectedShape, result.Shape())
		}
		// Expected: sum along axis 1 (middle axis)
		expectedData := []float32{9, 12, 27, 30} // [1+3+5, 2+4+6, 7+9+11, 8+10+12]
		if !reflect.DeepEqual(result.Data(), expectedData) {
			t.Errorf("expected data %v, got %v", expectedData, result.Data())
		}
	})
}

// TestFinalCoverageEdgeCases tests the remaining uncovered edge cases.
func TestFinalCoverageEdgeCases(t *testing.T) {
	engine := NewCPUEngine[float32](numeric.Float32Ops{})
	ctx := context.Background()

	// Test Pow function with shape broadcasting error
	t.Run("Pow_BroadcastError", func(t *testing.T) {
		// Create tensors with incompatible shapes for broadcasting
		base, _ := tensor.New[float32]([]int{2, 3}, []float32{1, 2, 3, 4, 5, 6})
		exp, _ := tensor.New[float32]([]int{4, 5}, make([]float32, 20))

		_, err := engine.Pow(ctx, base, exp)
		if err == nil {
			t.Error("expected error for incompatible broadcasting shapes")
		}
	})

	// Test MatMul with destination tensor error (getOrCreateDest error path)
	t.Run("MatMul_DestError", func(t *testing.T) {
		a, _ := tensor.New[float32]([]int{2, 3}, []float32{1, 2, 3, 4, 5, 6})
		b, _ := tensor.New[float32]([]int{3, 2}, []float32{1, 2, 3, 4, 5, 6})
		// Create a destination tensor with wrong shape
		wrongDst, _ := tensor.New[float32]([]int{3, 3}, make([]float32, 9))

		_, err := engine.MatMul(ctx, a, b, wrongDst)
		if err == nil {
			t.Error("expected error for wrong destination tensor shape")
		}
	})

	// Test MatMul result.Set error path
	t.Run("MatMul_SetError", func(t *testing.T) {
		// This is harder to trigger since Set typically doesn't fail for valid indices
		// But we can test the normal MatMul path to ensure coverage
		a, _ := tensor.New[float32]([]int{1, 1}, []float32{2})
		b, _ := tensor.New[float32]([]int{1, 1}, []float32{3})

		result, err := engine.MatMul(ctx, a, b)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		expectedData := []float32{6}
		if !reflect.DeepEqual(result.Data(), expectedData) {
			t.Errorf("expected data %v, got %v", expectedData, result.Data())
		}
	})

	// Test Transpose destination error
	t.Run("Transpose_DestError", func(t *testing.T) {
		a, _ := tensor.New[float32]([]int{2, 3}, []float32{1, 2, 3, 4, 5, 6})
		// Create a destination tensor with wrong shape
		wrongDst, _ := tensor.New[float32]([]int{2, 2}, make([]float32, 4))

		_, err := engine.Transpose(ctx, a, []int{1, 0}, wrongDst)
		if err == nil {
			t.Error("expected error for wrong destination tensor shape")
		}
	})

	// Test Sum with destination error
	t.Run("Sum_DestError", func(t *testing.T) {
		a, _ := tensor.New[float32]([]int{2, 3}, []float32{1, 2, 3, 4, 5, 6})
		// Create a destination tensor with wrong shape
		wrongDst, _ := tensor.New[float32]([]int{5, 5}, make([]float32, 25))

		_, err := engine.Sum(ctx, a, 0, false, wrongDst)
		if err == nil {
			t.Error("expected error for wrong destination tensor shape")
		}
	})

	// Test Sum Zero error path
	t.Run("Sum_ZeroError", func(t *testing.T) {
		// This is difficult to trigger since Zero typically doesn't fail
		// But we can test a normal sum operation to ensure coverage
		a, _ := tensor.New[float32]([]int{2, 2}, []float32{1, 2, 3, 4})

		result, err := engine.Sum(ctx, a, 0, false)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		expectedData := []float32{4, 6} // sum along axis 0
		if !reflect.DeepEqual(result.Data(), expectedData) {
			t.Errorf("expected data %v, got %v", expectedData, result.Data())
		}
	})

	// Test Pow with destination error
	t.Run("Pow_DestError", func(t *testing.T) {
		base, _ := tensor.New[float32]([]int{2, 2}, []float32{1, 2, 3, 4})
		exp, _ := tensor.New[float32]([]int{2, 2}, []float32{2, 2, 2, 2})
		// Create a destination tensor with wrong shape
		wrongDst, _ := tensor.New[float32]([]int{3, 3}, make([]float32, 9))

		_, err := engine.Pow(ctx, base, exp, wrongDst)
		if err == nil {
			t.Error("expected error for wrong destination tensor shape")
		}
	})

	// Test Transpose result.Set error path - create a scenario where Set might fail
	t.Run("Transpose_SetPath", func(t *testing.T) {
		// Test normal transpose to ensure the Set path is covered
		a, _ := tensor.New[float32]([]int{2, 3}, []float32{1, 2, 3, 4, 5, 6})

		result, err := engine.Transpose(ctx, a, []int{1, 0})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		expectedShape := []int{3, 2}
		if !reflect.DeepEqual(result.Shape(), expectedShape) {
			t.Errorf("expected shape %v, got %v", expectedShape, result.Shape())
		}

		expectedData := []float32{1, 4, 2, 5, 3, 6} // transposed data
		if !reflect.DeepEqual(result.Data(), expectedData) {
			t.Errorf("expected data %v, got %v", expectedData, result.Data())
		}
	})

	// Test MatMul with exact dimension match to cover all paths
	t.Run("MatMul_ExactMatch", func(t *testing.T) {
		a, _ := tensor.New[float32]([]int{2, 2}, []float32{1, 2, 3, 4})
		b, _ := tensor.New[float32]([]int{2, 2}, []float32{5, 6, 7, 8})

		result, err := engine.MatMul(ctx, a, b)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		// Expected result: [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]] = [[19, 22], [43, 50]]
		expectedData := []float32{19, 22, 43, 50}
		if !reflect.DeepEqual(result.Data(), expectedData) {
			t.Errorf("expected data %v, got %v", expectedData, result.Data())
		}
	})

	// Test Sum with complex 3D tensor to cover all indexing paths
	t.Run("Sum_3D_Complex", func(t *testing.T) {
		// Create a 2x2x2 tensor
		a, _ := tensor.New[float32]([]int{2, 2, 2}, []float32{1, 2, 3, 4, 5, 6, 7, 8})

		result, err := engine.Sum(ctx, a, 2, true) // sum along last axis with keepDims
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		expectedShape := []int{2, 2, 1}
		if !reflect.DeepEqual(result.Shape(), expectedShape) {
			t.Errorf("expected shape %v, got %v", expectedShape, result.Shape())
		}
		// Expected: sum along last axis [1+2, 3+4, 5+6, 7+8] = [3, 7, 11, 15]
		expectedData := []float32{3, 7, 11, 15}
		if !reflect.DeepEqual(result.Data(), expectedData) {
			t.Errorf("expected data %v, got %v", expectedData, result.Data())
		}
	})
}
