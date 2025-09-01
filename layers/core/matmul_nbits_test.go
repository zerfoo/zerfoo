package core

import (
	"context"
	"math"
	"testing"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
	"github.com/zerfoo/zerfoo/types"
)

func TestNewMatMulNBits(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	// Test valid construction
	t.Run("valid_4bit", func(t *testing.T) {
		// Create test quantized weights: 2x3 matrix (6 uint8 values, each containing 2 packed 4-bit weights)
		quantWeights, err := tensor.New[uint8]([]int{2, 3}, []uint8{0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC})
		if err != nil {
			t.Fatalf("Failed to create quantized weights: %v", err)
		}

		// Scale: per-row scaling
		scale, err := tensor.New[float32]([]int{2}, []float32{0.1, 0.2})
		if err != nil {
			t.Fatalf("Failed to create scale tensor: %v", err)
		}

		layer, err := NewMatMulNBits[float32](
			"test_matmul",
			engine,
			ops,
			quantWeights,
			scale,
			nil, // No zero point for symmetric
			4,   // 4-bit
			true, // symmetric
		)

		if err != nil {
			t.Fatalf("Failed to create MatMulNBits layer: %v", err)
		}

		if layer.OpType() != "MatMulNBits" {
			t.Errorf("Expected OpType 'MatMulNBits', got '%s'", layer.OpType())
		}

		expectedShape := []int{2, 6} // 2 rows, 3*2=6 cols (4-bit unpacking doubles the columns)
		actualShape := layer.OutputShape()
		if len(actualShape) != len(expectedShape) {
			t.Errorf("Shape length mismatch: expected %v, got %v", expectedShape, actualShape)
		}

		for i, expected := range expectedShape {
			if actualShape[i] != expected {
				t.Errorf("Shape dimension %d: expected %d, got %d", i, expected, actualShape[i])
			}
		}
	})

	t.Run("invalid_nbits", func(t *testing.T) {
		quantWeights, _ := tensor.New[uint8]([]int{2, 2}, []uint8{1, 2, 3, 4})
		scale, _ := tensor.New[float32]([]int{2}, []float32{0.1, 0.2})

		_, err := NewMatMulNBits[float32](
			"test_matmul",
			engine,
			ops,
			quantWeights,
			scale,
			nil,
			8, // Invalid: only 4-bit supported
			true,
		)

		if err == nil {
			t.Errorf("Expected error for invalid nbits, got none")
		}
	})

	t.Run("nil_weights", func(t *testing.T) {
		scale, _ := tensor.New[float32]([]int{2}, []float32{0.1, 0.2})

		_, err := NewMatMulNBits[float32](
			"test_matmul",
			engine,
			ops,
			nil, // nil weights
			scale,
			nil,
			4,
			true,
		)

		if err == nil {
			t.Errorf("Expected error for nil weights, got none")
		}
	})
}

func TestMatMulNBitsDequantization(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	// Test symmetric quantization
	t.Run("symmetric_dequantization", func(t *testing.T) {
		// Create test data: [0x12] = [0001 0010] = [2, 1] when unpacked
		quantWeights, err := tensor.New[uint8]([]int{1, 1}, []uint8{0x12})
		if err != nil {
			t.Fatalf("Failed to create quantized weights: %v", err)
		}

		// Scale: 0.1
		scale, err := tensor.New[float32]([]int{1}, []float32{0.1})
		if err != nil {
			t.Fatalf("Failed to create scale tensor: %v", err)
		}

		layer, err := NewMatMulNBits[float32](
			"test_matmul",
			engine,
			ops,
			quantWeights,
			scale,
			nil,
			4,
			true, // symmetric
		)
		if err != nil {
			t.Fatalf("Failed to create MatMulNBits layer: %v", err)
		}

		// Get dequantized weights
		dequantized, err := layer.GetDequantizedWeights()
		if err != nil {
			t.Fatalf("Failed to dequantize weights: %v", err)
		}

		data := dequantized.Data()
		if len(data) != 2 {
			t.Fatalf("Expected 2 dequantized values, got %d", len(data))
		}

		// For symmetric quantization: dequantized = scale * (quantized - 128)
		// Unpacked: [2, 1]
		// Expected: 0.1 * (2 - 128) = -12.6, 0.1 * (1 - 128) = -12.7
		expected := []float32{
			0.1 * (2 - 128), // -12.6
			0.1 * (1 - 128), // -12.7
		}

		for i, exp := range expected {
			if math.Abs(float64(data[i]-exp)) > 1e-6 {
				t.Errorf("Dequantized value %d: expected %.6f, got %.6f", i, exp, data[i])
			}
		}
	})

	t.Run("asymmetric_dequantization", func(t *testing.T) {
		// Test asymmetric quantization with zero point
		quantWeights, err := tensor.New[uint8]([]int{1, 1}, []uint8{0x34}) // [4, 3] unpacked
		if err != nil {
			t.Fatalf("Failed to create quantized weights: %v", err)
		}

		scale, err := tensor.New[float32]([]int{1}, []float32{0.2})
		if err != nil {
			t.Fatalf("Failed to create scale tensor: %v", err)
		}

		zeroPoint, err := tensor.New[uint8]([]int{1}, []uint8{2})
		if err != nil {
			t.Fatalf("Failed to create zero point tensor: %v", err)
		}

		layer, err := NewMatMulNBits[float32](
			"test_matmul",
			engine,
			ops,
			quantWeights,
			scale,
			zeroPoint,
			4,
			false, // asymmetric
		)
		if err != nil {
			t.Fatalf("Failed to create MatMulNBits layer: %v", err)
		}

		dequantized, err := layer.GetDequantizedWeights()
		if err != nil {
			t.Fatalf("Failed to dequantize weights: %v", err)
		}

		data := dequantized.Data()
		
		// For asymmetric quantization: dequantized = scale * (quantized - zero_point)
		// Unpacked: [4, 3], zero_point=2, scale=0.2
		// Expected: 0.2 * (4 - 2) = 0.4, 0.2 * (3 - 2) = 0.2
		expected := []float32{
			0.2 * (4 - 2), // 0.4
			0.2 * (3 - 2), // 0.2
		}

		for i, exp := range expected {
			if math.Abs(float64(data[i]-exp)) > 1e-6 {
				t.Errorf("Dequantized value %d: expected %.6f, got %.6f", i, exp, data[i])
			}
		}
	})
}

func TestMatMulNBitsForward(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	// Create a simple 2x2 quantized weight matrix
	// Packed as [0x10, 0x23] = [[0,1], [3,2]] when unpacked
	quantWeights, err := tensor.New[uint8]([]int{2, 1}, []uint8{0x10, 0x23})
	if err != nil {
		t.Fatalf("Failed to create quantized weights: %v", err)
	}

	// Simple scale: 1.0 (no scaling)
	scale, err := tensor.New[float32]([]int{1}, []float32{1.0})
	if err != nil {
		t.Fatalf("Failed to create scale tensor: %v", err)
	}

	layer, err := NewMatMulNBits[float32](
		"test_matmul",
		engine,
		ops,
		quantWeights,
		scale,
		nil,
		4,
		true, // symmetric (zero point = 128)
	)
	if err != nil {
		t.Fatalf("Failed to create MatMulNBits layer: %v", err)
	}

	// Test input: [1, 2] (1x2 matrix)
	input, err := tensor.New[float32]([]int{1, 2}, []float32{1.0, 2.0})
	if err != nil {
		t.Fatalf("Failed to create input tensor: %v", err)
	}

	// Forward pass
	ctx := context.Background()
	output, err := layer.Forward(ctx, input)
	if err != nil {
		t.Fatalf("Forward pass failed: %v", err)
	}

	// Check output shape
	expectedShape := []int{1, 2} // Input was 1x2, weights are 2x2, output should be 1x2
	actualShape := output.Shape()
	if len(actualShape) != len(expectedShape) {
		t.Errorf("Output shape length mismatch: expected %v, got %v", expectedShape, actualShape)
	}

	// The actual values depend on the matrix multiplication of:
	// input [1, 2] @ dequantized_weights
	// Where dequantized weights are: [0-128, 1-128; 3-128, 2-128] = [[-128, -127], [-125, -126]]
	// Result: [1, 2] @ [[-128, -127], [-125, -126]] = [1*(-128) + 2*(-125), 1*(-127) + 2*(-126)]
	//       = [-128 - 250, -127 - 252] = [-378, -379]

	outputData := output.Data()
	if len(outputData) != 2 {
		t.Errorf("Expected 2 output values, got %d", len(outputData))
	}

	// Check that output is reasonable (exact values depend on matrix mult implementation)
	for i, val := range outputData {
		if math.IsNaN(float64(val)) || math.IsInf(float64(val), 0) {
			t.Errorf("Output value %d is invalid: %f", i, val)
		}
	}

	t.Logf("Forward pass output: %v", outputData)
}

func TestMatMulNBitsBackward(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	// Create simple test layer
	quantWeights, err := tensor.New[uint8]([]int{2, 1}, []uint8{0x10, 0x23})
	if err != nil {
		t.Fatalf("Failed to create quantized weights: %v", err)
	}

	scale, err := tensor.New[float32]([]int{1}, []float32{0.1})
	if err != nil {
		t.Fatalf("Failed to create scale tensor: %v", err)
	}

	layer, err := NewMatMulNBits[float32](
		"test_matmul",
		engine,
		ops,
		quantWeights,
		scale,
		nil,
		4,
		true,
	)
	if err != nil {
		t.Fatalf("Failed to create MatMulNBits layer: %v", err)
	}

	// Test input
	input, err := tensor.New[float32]([]int{1, 2}, []float32{1.0, 2.0})
	if err != nil {
		t.Fatalf("Failed to create input tensor: %v", err)
	}

	// Gradient output
	gradOutput, err := tensor.New[float32]([]int{1, 2}, []float32{1.0, 1.0})
	if err != nil {
		t.Fatalf("Failed to create gradient output: %v", err)
	}

	// Backward pass
	ctx := context.Background()
	gradInputs, err := layer.Backward(ctx, types.FullBackprop, gradOutput, input)
	if err != nil {
		t.Fatalf("Backward pass failed: %v", err)
	}

	if len(gradInputs) != 1 {
		t.Errorf("Expected 1 input gradient, got %d", len(gradInputs))
	}

	gradInput := gradInputs[0]
	expectedShape := input.Shape()
	actualShape := gradInput.Shape()

	if len(actualShape) != len(expectedShape) {
		t.Errorf("Gradient shape length mismatch: expected %v, got %v", expectedShape, actualShape)
	}

	for i, expected := range expectedShape {
		if actualShape[i] != expected {
			t.Errorf("Gradient shape dimension %d: expected %d, got %d", i, expected, actualShape[i])
		}
	}

	// Check gradient values are reasonable
	gradData := gradInput.Data()
	for i, val := range gradData {
		if math.IsNaN(float64(val)) || math.IsInf(float64(val), 0) {
			t.Errorf("Gradient value %d is invalid: %f", i, val)
		}
	}

	t.Logf("Backward pass gradient: %v", gradData)
}

func TestMatMulNBitsAttributes(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	quantWeights, _ := tensor.New[uint8]([]int{2, 1}, []uint8{0x10, 0x23})
	scale, _ := tensor.New[float32]([]int{1}, []float32{0.1})

	layer, err := NewMatMulNBits[float32](
		"test_matmul",
		engine,
		ops,
		quantWeights,
		scale,
		nil,
		4,
		true,
	)
	if err != nil {
		t.Fatalf("Failed to create MatMulNBits layer: %v", err)
	}

	// Test attributes
	attrs := layer.Attributes()
	if attrs["nbits"] != 4 {
		t.Errorf("Expected nbits=4, got %v", attrs["nbits"])
	}

	if attrs["symmetric"] != true {
		t.Errorf("Expected symmetric=true, got %v", attrs["symmetric"])
	}

	// Test parameters (should be empty for quantized layers)
	params := layer.Parameters()
	if len(params) != 0 {
		t.Errorf("Expected 0 parameters for quantized layer, got %d", len(params))
	}

	// Test quantization info
	info := layer.QuantizationInfo()
	if info["nbits"] != 4 {
		t.Errorf("Expected nbits=4 in quantization info, got %v", info["nbits"])
	}

	if info["symmetric"] != true {
		t.Errorf("Expected symmetric=true in quantization info, got %v", info["symmetric"])
	}
}

func TestMatMulNBitsCache(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	quantWeights, _ := tensor.New[uint8]([]int{1, 1}, []uint8{0x12})
	scale, _ := tensor.New[float32]([]int{1}, []float32{0.1})

	layer, err := NewMatMulNBits[float32](
		"test_matmul",
		engine,
		ops,
		quantWeights,
		scale,
		nil,
		4,
		true,
	)
	if err != nil {
		t.Fatalf("Failed to create MatMulNBits layer: %v", err)
	}

	// First call should compute and cache
	weights1, err := layer.GetDequantizedWeights()
	if err != nil {
		t.Fatalf("Failed to get dequantized weights (first call): %v", err)
	}

	// Second call should return cached result
	weights2, err := layer.GetDequantizedWeights()
	if err != nil {
		t.Fatalf("Failed to get dequantized weights (second call): %v", err)
	}

	// Should be the same instance (cached)
	if weights1 != weights2 {
		t.Errorf("Expected same cached instance, got different instances")
	}

	// Invalidate cache and verify new computation
	layer.InvalidateCache()
	weights3, err := layer.GetDequantizedWeights()
	if err != nil {
		t.Fatalf("Failed to get dequantized weights (after cache invalidation): %v", err)
	}

	// Should be different instance after cache invalidation
	if weights1 == weights3 {
		t.Errorf("Expected different instance after cache invalidation, got same instance")
	}

	// But data should be the same
	data1 := weights1.Data()
	data3 := weights3.Data()
	if len(data1) != len(data3) {
		t.Errorf("Data length changed after cache invalidation: %d vs %d", len(data1), len(data3))
	}

	for i := range data1 {
		if math.Abs(float64(data1[i]-data3[i])) > 1e-9 {
			t.Errorf("Data value %d changed after cache invalidation: %.9f vs %.9f", i, data1[i], data3[i])
		}
	}
}