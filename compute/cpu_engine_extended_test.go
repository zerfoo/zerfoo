package compute

import (
	"context"
	"testing"

	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
)

// TestCPUEngine_UncoveredFunctions tests functions with 0% coverage
func TestCPUEngine_UncoveredFunctions(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := NewCPUEngine[float32](ops)
	ctx := context.Background()

	t.Run("Tanh", func(t *testing.T) {
		input, err := tensor.New[float32]([]int{2, 2}, []float32{0.5, -0.5, 1.0, -1.0})
		if err != nil {
			t.Fatalf("Failed to create input tensor: %v", err)
		}

		result, err := engine.Tanh(ctx, input, nil)
		if err != nil {
			t.Fatalf("Tanh failed: %v", err)
		}

		// Verify result shape
		if !equalSlices(result.Shape(), []int{2, 2}) {
			t.Errorf("Expected shape [2, 2], got %v", result.Shape())
		}

		// Verify result values (approximately)
		data := result.Data()
		if len(data) != 4 {
			t.Errorf("Expected 4 values, got %d", len(data))
		}

		// tanh(0.5) ≈ 0.462, tanh(-0.5) ≈ -0.462, tanh(1.0) ≈ 0.762, tanh(-1.0) ≈ -0.762
		expected := []float32{0.462, -0.462, 0.762, -0.762}
		for i, v := range data {
			if abs(v-expected[i]) > 0.01 { // Allow small tolerance
				t.Errorf("Expected tanh value ~%f, got %f at index %d", expected[i], v, i)
			}
		}
	})

	t.Run("TanhPrime", func(t *testing.T) {
		input, err := tensor.New[float32]([]int{2, 2}, []float32{0.0, 0.5, 1.0, 2.0})
		if err != nil {
			t.Fatalf("Failed to create input tensor: %v", err)
		}

		upstream, err := tensor.New[float32]([]int{2, 2}, []float32{1.0, 1.0, 1.0, 1.0})
		if err != nil {
			t.Fatalf("Failed to create upstream tensor: %v", err)
		}

		result, err := engine.TanhPrime(ctx, input, upstream, nil)
		if err != nil {
			t.Fatalf("TanhPrime failed: %v", err)
		}

		// Verify result shape
		if !equalSlices(result.Shape(), []int{2, 2}) {
			t.Errorf("Expected shape [2, 2], got %v", result.Shape())
		}

		// TanhPrime should give the gradient of tanh
		data := result.Data()
		if len(data) != 4 {
			t.Errorf("Expected 4 values, got %d", len(data))
		}
	})

	t.Run("Split", func(t *testing.T) {
		// Test splitting a tensor along an axis
		input, err := tensor.New[float32]([]int{4, 2}, []float32{1, 2, 3, 4, 5, 6, 7, 8})
		if err != nil {
			t.Fatalf("Failed to create input tensor: %v", err)
		}

		// Split along axis 0 into 2 parts
		results, err := engine.Split(ctx, input, 2, 0)
		if err != nil {
			t.Fatalf("Split failed: %v", err)
		}

		if len(results) != 2 {
			t.Errorf("Expected 2 splits, got %d", len(results))
		}

		// Each split should have shape [2, 2]
		for i, result := range results {
			if !equalSlices(result.Shape(), []int{2, 2}) {
				t.Errorf("Split %d: expected shape [2, 2], got %v", i, result.Shape())
			}
		}

		// Verify data in splits
		split0Data := results[0].Data()
		split1Data := results[1].Data()

		expectedSplit0 := []float32{1, 2, 3, 4}
		expectedSplit1 := []float32{5, 6, 7, 8}

		for i, v := range split0Data {
			if v != expectedSplit0[i] {
				t.Errorf("Split 0 data[%d]: expected %f, got %f", i, expectedSplit0[i], v)
			}
		}

		for i, v := range split1Data {
			if v != expectedSplit1[i] {
				t.Errorf("Split 1 data[%d]: expected %f, got %f", i, expectedSplit1[i], v)
			}
		}
	})

	t.Run("ReduceSum", func(t *testing.T) {
		input, err := tensor.New[float32]([]int{2, 3}, []float32{1, 2, 3, 4, 5, 6})
		if err != nil {
			t.Fatalf("Failed to create input tensor: %v", err)
		}

		// Reduce sum along axis 1
		result, err := engine.ReduceSum(ctx, input, 1, false)
		if err != nil {
			t.Fatalf("ReduceSum failed: %v", err)
		}

		// Result should have shape [2] with sums [6, 15] 
		if !equalSlices(result.Shape(), []int{2}) {
			t.Errorf("Expected shape [2], got %v", result.Shape())
		}

		data := result.Data()
		expected := []float32{6.0, 15.0} // [1+2+3, 4+5+6]
		for i, v := range data {
			if v != expected[i] {
				t.Errorf("Expected %f, got %f at index %d", expected[i], v, i)
			}
		}
	})

	t.Run("AddScalar", func(t *testing.T) {
		input, err := tensor.New[float32]([]int{2, 2}, []float32{1, 2, 3, 4})
		if err != nil {
			t.Fatalf("Failed to create input tensor: %v", err)
		}

		result, err := engine.AddScalar(ctx, input, 10.0, nil)
		if err != nil {
			t.Fatalf("AddScalar failed: %v", err)
		}

		// Verify result shape
		if !equalSlices(result.Shape(), []int{2, 2}) {
			t.Errorf("Expected shape [2, 2], got %v", result.Shape())
		}

		// Verify result values
		data := result.Data()
		expected := []float32{11, 12, 13, 14}
		for i, v := range data {
			if v != expected[i] {
				t.Errorf("Expected %f, got %f at index %d", expected[i], v, i)
			}
		}
	})

	t.Run("MulScalar", func(t *testing.T) {
		input, err := tensor.New[float32]([]int{2, 2}, []float32{1, 2, 3, 4})
		if err != nil {
			t.Fatalf("Failed to create input tensor: %v", err)
		}

		result, err := engine.MulScalar(ctx, input, 3.0, nil)
		if err != nil {
			t.Fatalf("MulScalar failed: %v", err)
		}

		// Verify result shape
		if !equalSlices(result.Shape(), []int{2, 2}) {
			t.Errorf("Expected shape [2, 2], got %v", result.Shape())
		}

		// Verify result values
		data := result.Data()
		expected := []float32{3, 6, 9, 12}
		for i, v := range data {
			if v != expected[i] {
				t.Errorf("Expected %f, got %f at index %d", expected[i], v, i)
			}
		}
	})

	t.Run("DivScalar", func(t *testing.T) {
		input, err := tensor.New[float32]([]int{2, 2}, []float32{10, 20, 30, 40})
		if err != nil {
			t.Fatalf("Failed to create input tensor: %v", err)
		}

		result, err := engine.DivScalar(ctx, input, 10.0, nil)
		if err != nil {
			t.Fatalf("DivScalar failed: %v", err)
		}

		// Verify result shape
		if !equalSlices(result.Shape(), []int{2, 2}) {
			t.Errorf("Expected shape [2, 2], got %v", result.Shape())
		}

		// Verify result values
		data := result.Data()
		expected := []float32{1, 2, 3, 4}
		for i, v := range data {
			if v != expected[i] {
				t.Errorf("Expected %f, got %f at index %d", expected[i], v, i)
			}
		}
	})

	t.Run("Zeros", func(t *testing.T) {
		// Create a tensor and zero it out with new shape
		input, err := tensor.New[float32]([]int{2, 2}, []float32{1, 2, 3, 4})
		if err != nil {
			t.Fatalf("Failed to create input tensor: %v", err)
		}

		err = engine.Zeros(ctx, input, []int{3, 2})
		if err != nil {
			t.Fatalf("Zeros failed: %v", err)
		}

		// Verify result shape
		if !equalSlices(input.Shape(), []int{3, 2}) {
			t.Errorf("Expected shape [3, 2], got %v", input.Shape())
		}

		// Verify all values are zero
		data := input.Data()
		for i, v := range data {
			if v != 0.0 {
				t.Errorf("Expected 0, got %f at index %d", v, i)
			}
		}
	})

	t.Run("RandomUniform", func(t *testing.T) {
		// Create a tensor to fill with random values
		result, err := tensor.New[float32]([]int{10, 10}, nil)
		if err != nil {
			t.Fatalf("Failed to create result tensor: %v", err)
		}

		err = engine.RandomUniform(ctx, result, 0.0, 1.0)
		if err != nil {
			t.Fatalf("RandomUniform failed: %v", err)
		}

		// Verify result shape
		if !equalSlices(result.Shape(), []int{10, 10}) {
			t.Errorf("Expected shape [10, 10], got %v", result.Shape())
		}

		// Verify all values are in range [0, 1)
		data := result.Data()
		for i, v := range data {
			if v < 0.0 || v >= 1.0 {
				t.Errorf("Expected value in [0, 1), got %f at index %d", v, i)
			}
		}
	})

	t.Run("Fill", func(t *testing.T) {
		input, err := tensor.New[float32]([]int{2, 3}, []float32{1, 2, 3, 4, 5, 6})
		if err != nil {
			t.Fatalf("Failed to create input tensor: %v", err)
		}

		err = engine.Fill(ctx, input, 42.0)
		if err != nil {
			t.Fatalf("Fill failed: %v", err)
		}

		// Verify all values are 42.0
		data := input.Data()
		for i, v := range data {
			if v != 42.0 {
				t.Errorf("Expected 42.0, got %f at index %d", v, i)
			}
		}
	})

	t.Run("Ops", func(t *testing.T) {
		// Test that Ops returns the correct operations instance
		result := engine.Ops()
		if result != ops {
			t.Errorf("Expected ops instance to match")
		}
	})
}

func TestCPUEngine_AdvancedFunctions(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := NewCPUEngine[float32](ops)
	ctx := context.Background()

	t.Run("Concat", func(t *testing.T) {
		// Test concatenation along axis 0
		tensor1, err := tensor.New[float32]([]int{2, 2}, []float32{1, 2, 3, 4})
		if err != nil {
			t.Fatalf("Failed to create tensor1: %v", err)
		}

		tensor2, err := tensor.New[float32]([]int{2, 2}, []float32{5, 6, 7, 8})
		if err != nil {
			t.Fatalf("Failed to create tensor2: %v", err)
		}

		result, err := engine.Concat(ctx, []*tensor.TensorNumeric[float32]{tensor1, tensor2}, 0, nil)
		if err != nil {
			t.Fatalf("Concat failed: %v", err)
		}

		// Result should have shape [4, 2]
		if !equalSlices(result.Shape(), []int{4, 2}) {
			t.Errorf("Expected shape [4, 2], got %v", result.Shape())
		}

		// Verify concatenated data
		data := result.Data()
		expected := []float32{1, 2, 3, 4, 5, 6, 7, 8}
		for i, v := range data {
			if v != expected[i] {
				t.Errorf("Expected %f, got %f at index %d", expected[i], v, i)
			}
		}
	})

	t.Run("OneHot", func(t *testing.T) {
		// Test one-hot encoding
		indices, err := tensor.New[int]([]int{3}, []int{0, 2, 1})
		if err != nil {
			t.Fatalf("Failed to create indices tensor: %v", err)
		}

		result, err := engine.OneHot(ctx, indices, 3, nil)
		if err != nil {
			t.Fatalf("OneHot failed: %v", err)
		}

		// Result should have shape [3, 3]
		if !equalSlices(result.Shape(), []int{3, 3}) {
			t.Errorf("Expected shape [3, 3], got %v", result.Shape())
		}

		// Verify one-hot encoding (using default 1.0/0.0 values)
		data := result.Data()
		expected := []float32{
			1, 0, 0, // index 0
			0, 0, 1, // index 2
			0, 1, 0, // index 1
		}
		for i, v := range data {
			if v != expected[i] {
				t.Errorf("Expected %f, got %f at index %d", expected[i], v, i)
			}
		}
	})

	t.Run("Reshape", func(t *testing.T) {
		input, err := tensor.New[float32]([]int{2, 3}, []float32{1, 2, 3, 4, 5, 6})
		if err != nil {
			t.Fatalf("Failed to create input tensor: %v", err)
		}

		result, err := engine.Reshape(ctx, input, []int{3, 2}, nil)
		if err != nil {
			t.Fatalf("Reshape failed: %v", err)
		}

		// Result should have shape [3, 2]
		if !equalSlices(result.Shape(), []int{3, 2}) {
			t.Errorf("Expected shape [3, 2], got %v", result.Shape())
		}

		// Data should remain the same
		data := result.Data()
		expected := []float32{1, 2, 3, 4, 5, 6}
		for i, v := range data {
			if v != expected[i] {
				t.Errorf("Expected %f, got %f at index %d", expected[i], v, i)
			}
		}
	})

	t.Run("Repeat", func(t *testing.T) {
		input, err := tensor.New[float32]([]int{2, 2}, []float32{1, 2, 3, 4})
		if err != nil {
			t.Fatalf("Failed to create input tensor: %v", err)
		}

		// Repeat 2 times along axis 0
		result, err := engine.Repeat(ctx, input, 0, 2, nil)
		if err != nil {
			t.Fatalf("Repeat failed: %v", err)
		}

		// Result should have shape [4, 2]
		if !equalSlices(result.Shape(), []int{4, 2}) {
			t.Errorf("Expected shape [4, 2], got %v", result.Shape())
		}

		// Verify repeated data - repeating along axis 0 means the entire tensor is repeated
		data := result.Data()
		// Input was [1,2] [3,4], repeating along axis 0 twice gives: [1,2] [3,4] [1,2] [3,4]
		expected := []float32{1, 2, 3, 4, 1, 2, 3, 4}
		
		if len(data) != len(expected) {
			t.Errorf("Expected %d values, got %d. Data: %v", len(expected), len(data), data)
		}
		
		if !equalFloat32Slices(data, expected) {
			t.Errorf("Repeat failed: expected %v, got %v", expected, data)
		}
	})

	t.Run("ReduceMean", func(t *testing.T) {
		input, err := tensor.New[float32]([]int{2, 3}, []float32{1, 2, 3, 4, 5, 6})
		if err != nil {
			t.Fatalf("Failed to create input tensor: %v", err)
		}

		// Reduce mean along axis 1
		result, err := engine.ReduceMean(ctx, input, 1, false, nil)
		if err != nil {
			t.Fatalf("ReduceMean failed: %v", err)
		}

		// Result should have shape [2] (mean of [1,2,3] = 2, mean of [4,5,6] = 5)
		if !equalSlices(result.Shape(), []int{2}) {
			t.Errorf("Expected shape [2], got %v", result.Shape())
		}

		data := result.Data()
		expected := []float32{2.0, 5.0}
		for i, v := range data {
			if abs(v-expected[i]) > 0.001 {
				t.Errorf("Expected %f, got %f at index %d", expected[i], v, i)
			}
		}
	})

	t.Run("Rsqrt", func(t *testing.T) {
		input, err := tensor.New[float32]([]int{2, 2}, []float32{1, 4, 9, 16})
		if err != nil {
			t.Fatalf("Failed to create input tensor: %v", err)
		}

		result, err := engine.Rsqrt(ctx, input, nil)
		if err != nil {
			t.Fatalf("Rsqrt failed: %v", err)
		}

		// Verify result shape
		if !equalSlices(result.Shape(), []int{2, 2}) {
			t.Errorf("Expected shape [2, 2], got %v", result.Shape())
		}

		// Rsqrt(x) = 1/sqrt(x): 1/1=1, 1/2=0.5, 1/3≈0.333, 1/4=0.25
		data := result.Data()
		expected := []float32{1.0, 0.5, 0.333, 0.25}
		for i, v := range data {
			if abs(v-expected[i]) > 0.01 {
				t.Errorf("Expected ~%f, got %f at index %d", expected[i], v, i)
			}
		}
	})

	t.Run("Sqrt", func(t *testing.T) {
		input, err := tensor.New[float32]([]int{2, 2}, []float32{1, 4, 9, 16})
		if err != nil {
			t.Fatalf("Failed to create input tensor: %v", err)
		}

		result, err := engine.Sqrt(ctx, input, nil)
		if err != nil {
			t.Fatalf("Sqrt failed: %v", err)
		}

		// Verify result shape
		if !equalSlices(result.Shape(), []int{2, 2}) {
			t.Errorf("Expected shape [2, 2], got %v", result.Shape())
		}

		// Sqrt values: sqrt(1)=1, sqrt(4)=2, sqrt(9)=3, sqrt(16)=4
		data := result.Data()
		expected := []float32{1.0, 2.0, 3.0, 4.0}
		for i, v := range data {
			if v != expected[i] {
				t.Errorf("Expected %f, got %f at index %d", expected[i], v, i)
			}
		}
	})
}

// TestCPUEngine_LowCoverage tests functions with low coverage
func TestCPUEngine_LowCoverage(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := NewCPUEngine[float32](ops)
	ctx := context.Background()

	t.Run("MatMul_EdgeCases", func(t *testing.T) {
		// Test different matrix sizes to improve MatMul coverage
		
		// Test 1x1 matrices
		a, err := tensor.New[float32]([]int{1, 1}, []float32{2.0})
		if err != nil {
			t.Fatalf("Failed to create tensor a: %v", err)
		}
		
		b, err := tensor.New[float32]([]int{1, 1}, []float32{3.0})
		if err != nil {
			t.Fatalf("Failed to create tensor b: %v", err)
		}
		
		result, err := engine.MatMul(ctx, a, b, nil)
		if err != nil {
			t.Fatalf("MatMul failed: %v", err)
		}
		
		if !equalSlices(result.Shape(), []int{1, 1}) {
			t.Errorf("Expected shape [1, 1], got %v", result.Shape())
		}
		
		data := result.Data()
		if len(data) != 1 || data[0] != 6.0 {
			t.Errorf("Expected [6.0], got %v", data)
		}
		
		// Test different dimension combinations
		c, err := tensor.New[float32]([]int{2, 3}, []float32{1, 2, 3, 4, 5, 6})
		if err != nil {
			t.Fatalf("Failed to create tensor c: %v", err)
		}
		
		d, err := tensor.New[float32]([]int{3, 2}, []float32{1, 2, 3, 4, 5, 6})
		if err != nil {
			t.Fatalf("Failed to create tensor d: %v", err)
		}
		
		result2, err := engine.MatMul(ctx, c, d, nil)
		if err != nil {
			t.Fatalf("MatMul 2x3 * 3x2 failed: %v", err)
		}
		
		if !equalSlices(result2.Shape(), []int{2, 2}) {
			t.Errorf("Expected shape [2, 2], got %v", result2.Shape())
		}
	})

	t.Run("Reshape_EdgeCases", func(t *testing.T) {
		// Test reshape with different dimension configurations
		input, err := tensor.New[float32]([]int{2, 3}, []float32{1, 2, 3, 4, 5, 6})
		if err != nil {
			t.Fatalf("Failed to create input tensor: %v", err)
		}
		
		// Test reshape to 1D
		result1, err := engine.Reshape(ctx, input, []int{6}, nil)
		if err != nil {
			t.Fatalf("Reshape to 1D failed: %v", err)
		}
		
		if !equalSlices(result1.Shape(), []int{6}) {
			t.Errorf("Expected shape [6], got %v", result1.Shape())
		}
		
		// Test reshape to 3D
		result2, err := engine.Reshape(ctx, input, []int{1, 2, 3}, nil)
		if err != nil {
			t.Fatalf("Reshape to 3D failed: %v", err)
		}
		
		if !equalSlices(result2.Shape(), []int{1, 2, 3}) {
			t.Errorf("Expected shape [1, 2, 3], got %v", result2.Shape())
		}
		
		// Test invalid reshape (should fail)
		_, err = engine.Reshape(ctx, input, []int{5}, nil)
		if err == nil {
			t.Error("Expected error for invalid reshape, got nil")
		}
	})
	
	t.Run("Gather_EdgeCases", func(t *testing.T) {
		// Test Gather function to improve its coverage
		// Gather expects 2D params [vocab, dim] and 1D/2D indices
		params, err := tensor.New[float32]([]int{3, 2}, []float32{1, 2, 3, 4, 5, 6})
		if err != nil {
			t.Fatalf("Failed to create params tensor: %v", err)
		}
		
		indices, err := tensor.New[int]([]int{2}, []int{0, 2})
		if err != nil {
			t.Fatalf("Failed to create indices tensor: %v", err)
		}
		
		output, err := tensor.New[float32]([]int{2, 2}, make([]float32, 4))
		if err != nil {
			t.Fatalf("Failed to create output tensor: %v", err)
		}
		
		err = engine.Gather(ctx, params, indices, output)
		if err != nil {
			t.Fatalf("Gather failed: %v", err)
		}
		
		// Verify output shape
		if !equalSlices(output.Shape(), []int{2, 2}) {
			t.Errorf("Expected shape [2, 2], got %v", output.Shape())
		}
		
		// Verify values: should gather rows 0 and 2 from params
		data := output.Data()
		expected := []float32{1, 2, 5, 6} // row 0: [1,2], row 2: [5,6]
		if !equalFloat32Slices(data, expected) {
			t.Errorf("Expected %v, got %v", expected, data)
		}
	})

	t.Run("ScatterAdd_EdgeCases", func(t *testing.T) {
		// Test ScatterAdd function to improve its coverage
		// ScatterAdd expects dEmbeddingTable [vocab, dim], indices [N], dOut [N, dim]
		dEmbeddingTable, err := tensor.New[float32]([]int{3, 2}, []float32{1, 2, 3, 4, 5, 6})
		if err != nil {
			t.Fatalf("Failed to create dEmbeddingTable tensor: %v", err)
		}
		
		indices, err := tensor.New[int]([]int{2}, []int{0, 2})
		if err != nil {
			t.Fatalf("Failed to create indices tensor: %v", err)
		}
		
		dOut, err := tensor.New[float32]([]int{2, 2}, []float32{10, 20, 30, 40})
		if err != nil {
			t.Fatalf("Failed to create dOut tensor: %v", err)
		}
		
		err = engine.ScatterAdd(ctx, dEmbeddingTable, indices, dOut)
		if err != nil {
			t.Fatalf("ScatterAdd failed: %v", err)
		}
		
		// Verify that the values were scattered correctly
		// Original [1,2,3,4,5,6], scatter add [10,20] to row 0, [30,40] to row 2
		// Should be [11,22,3,4,35,46]
		data := dEmbeddingTable.Data()
		expected := []float32{11, 22, 3, 4, 35, 46}
		if !equalFloat32Slices(data, expected) {
			t.Errorf("Expected %v, got %v", expected, data)
		}
	})

	t.Run("Softmax_EdgeCases", func(t *testing.T) {
		// Test Softmax with different tensor shapes and values
		
		// Test 1D case
		input1D, err := tensor.New[float32]([]int{3}, []float32{1.0, 2.0, 3.0})
		if err != nil {
			t.Fatalf("Failed to create 1D tensor: %v", err)
		}
		
		result1D, err := engine.Softmax(ctx, input1D, -1, nil)
		if err != nil {
			t.Fatalf("Softmax 1D failed: %v", err)
		}
		
		if !equalSlices(result1D.Shape(), []int{3}) {
			t.Errorf("Expected shape [3], got %v", result1D.Shape())
		}
		
		// Test 2D case with different axes
		input2D, err := tensor.New[float32]([]int{2, 3}, []float32{1, 2, 3, 4, 5, 6})
		if err != nil {
			t.Fatalf("Failed to create 2D tensor: %v", err)
		}
		
		// Test axis = -1 (last axis)
		result2D, err := engine.Softmax(ctx, input2D, -1, nil)
		if err != nil {
			t.Fatalf("Softmax 2D axis=-1 failed: %v", err)
		}
		
		if !equalSlices(result2D.Shape(), []int{2, 3}) {
			t.Errorf("Expected shape [2, 3], got %v", result2D.Shape())
		}
		
		// Test axis = 0
		result2D_axis0, err := engine.Softmax(ctx, input2D, 0, nil)
		if err != nil {
			t.Fatalf("Softmax 2D axis=0 failed: %v", err)
		}
		
		if !equalSlices(result2D_axis0.Shape(), []int{2, 3}) {
			t.Errorf("Expected shape [2, 3], got %v", result2D_axis0.Shape())
		}
	})

	t.Run("DivScalar_EdgeCases", func(t *testing.T) {
		// Test DivScalar with different values and shapes
		input, err := tensor.New[float32]([]int{2, 2}, []float32{4.0, 8.0, 12.0, 16.0})
		if err != nil {
			t.Fatalf("Failed to create input tensor: %v", err)
		}
		
		result, err := engine.DivScalar(ctx, input, 2.0, nil)
		if err != nil {
			t.Fatalf("DivScalar failed: %v", err)
		}
		
		if !equalSlices(result.Shape(), []int{2, 2}) {
			t.Errorf("Expected shape [2, 2], got %v", result.Shape())
		}
		
		data := result.Data()
		expected := []float32{2.0, 4.0, 6.0, 8.0}
		if !equalFloat32Slices(data, expected) {
			t.Errorf("Expected %v, got %v", expected, data)
		}
		
		// Test with edge case values
		input2, err := tensor.New[float32]([]int{2}, []float32{0.0, 1.0})
		if err != nil {
			t.Fatalf("Failed to create input2 tensor: %v", err)
		}
		
		result2, err := engine.DivScalar(ctx, input2, 1.0, nil)
		if err != nil {
			t.Fatalf("DivScalar edge case failed: %v", err)
		}
		
		data2 := result2.Data()
		expected2 := []float32{0.0, 1.0}
		if !equalFloat32Slices(data2, expected2) {
			t.Errorf("Expected %v, got %v", expected2, data2)
		}
	})

	t.Run("Zeros_EdgeCases", func(t *testing.T) {
		// Test Zeros with different shapes
		
		// Test reshaping to 1D
		tensor1D, err := tensor.New[float32]([]int{3}, []float32{1, 2, 3})
		if err != nil {
			t.Fatalf("Failed to create tensor: %v", err)
		}
		
		err = engine.Zeros(ctx, tensor1D, []int{5})
		if err != nil {
			t.Fatalf("Zeros 1D failed: %v", err)
		}
		
		if !equalSlices(tensor1D.Shape(), []int{5}) {
			t.Errorf("Expected shape [5], got %v", tensor1D.Shape())
		}
		
		data1D := tensor1D.Data()
		for i, v := range data1D {
			if v != 0.0 {
				t.Errorf("Expected 0.0 at index %d, got %f", i, v)
			}
		}
		
		// Test reshaping to 3D
		tensor3D, err := tensor.New[float32]([]int{2}, []float32{5, 6})
		if err != nil {
			t.Fatalf("Failed to create tensor: %v", err)
		}
		
		err = engine.Zeros(ctx, tensor3D, []int{2, 2, 2})
		if err != nil {
			t.Fatalf("Zeros 3D failed: %v", err)
		}
		
		if !equalSlices(tensor3D.Shape(), []int{2, 2, 2}) {
			t.Errorf("Expected shape [2, 2, 2], got %v", tensor3D.Shape())
		}
		
		data3D := tensor3D.Data()
		for i, v := range data3D {
			if v != 0.0 {
				t.Errorf("Expected 0.0 at index %d, got %f", i, v)
			}
		}
		
		// Test zeroing without reshaping
		tensorNoReshape, err := tensor.New[float32]([]int{3}, []float32{7, 8, 9})
		if err != nil {
			t.Fatalf("Failed to create tensor: %v", err)
		}
		
		err = engine.Zeros(ctx, tensorNoReshape, nil)
		if err != nil {
			t.Fatalf("Zeros without reshaping failed: %v", err)
		}
		
		dataNoReshape := tensorNoReshape.Data()
		for i, v := range dataNoReshape {
			if v != 0.0 {
				t.Errorf("Expected 0.0 at index %d, got %f", i, v)
			}
		}
	})

	t.Run("ParallelProcessing_LargeTensors", func(t *testing.T) {
		// Test with large tensors to trigger parallel processing paths
		
		// Large matrix multiplication to trigger parallelFor
		largeA, err := tensor.New[float32]([]int{100, 50}, make([]float32, 5000))
		if err != nil {
			t.Fatalf("Failed to create large tensor A: %v", err)
		}
		
		// Fill with some data
		dataA := largeA.Data()
		for i := range dataA {
			dataA[i] = float32(i % 10)
		}
		
		largeB, err := tensor.New[float32]([]int{50, 80}, make([]float32, 4000))
		if err != nil {
			t.Fatalf("Failed to create large tensor B: %v", err)
		}
		
		// Fill with some data
		dataB := largeB.Data()
		for i := range dataB {
			dataB[i] = float32((i % 5) + 1)
		}
		
		result, err := engine.MatMul(ctx, largeA, largeB, nil)
		if err != nil {
			t.Fatalf("Large MatMul failed: %v", err)
		}
		
		if !equalSlices(result.Shape(), []int{100, 80}) {
			t.Errorf("Expected shape [100, 80], got %v", result.Shape())
		}
		
		// Test large tensor operations that might use parallelFor
		largeUnary, err := tensor.New[float32]([]int{1000, 100}, make([]float32, 100000))
		if err != nil {
			t.Fatalf("Failed to create large tensor for unary ops: %v", err)
		}
		
		// Fill with data
		dataUnary := largeUnary.Data()
		for i := range dataUnary {
			dataUnary[i] = float32(i % 100)
		}
		
		// Test Exp on large tensor (might trigger parallel processing)
		_, err = engine.Exp(ctx, largeUnary, nil)
		if err != nil {
			t.Fatalf("Large Exp failed: %v", err)
		}
		
		// Test Sum on large tensor with different axes
		_, err = engine.Sum(ctx, largeUnary, 0, false, nil)
		if err != nil {
			t.Fatalf("Large Sum axis 0 failed: %v", err)
		}
		
		_, err = engine.Sum(ctx, largeUnary, 1, false, nil)
		if err != nil {
			t.Fatalf("Large Sum axis 1 failed: %v", err)
		}
	})
}

// Helper function for floating point comparison
func abs(x float32) float32 {
	if x < 0 {
		return -x
	}
	return x
}


// Helper function to compare float32 slices
func equalFloat32Slices(a, b []float32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}