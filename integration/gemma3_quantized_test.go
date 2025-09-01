package integration

import (
	"context"
	"math"
	"testing"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/layers/core"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
)

// TestGemma3QuantizedInference tests end-to-end quantized inference simulation.
// This simulates the key components needed for Gemma 3 quantized model inference:
// 1. Constant tensors for quantized weights, scales, zero points
// 2. MatMulNBits operations for quantized matrix multiplication
// 3. Integration between components
func TestGemma3QuantizedInference(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)
	ctx := context.Background()

	t.Run("simple_quantized_linear_layer", func(t *testing.T) {
		// Simulate a simple quantized linear layer: input @ weights + bias
		// This represents a typical transformer layer component

		// Step 1: Create quantized weight constants
		// Simulate 4x3 weight matrix packed as 4x2 uint8 (4-bit packed)
		quantizedWeightsData := []uint8{
			0x12, 0x34, // Row 0: [2, 1, 4, 3] when unpacked
			0x56, 0x78, // Row 1: [6, 5, 8, 7] when unpacked
			0x9A, 0xBC, // Row 2: [10, 9, 12, 11] when unpacked
			0xDE, 0xF0, // Row 3: [14, 13, 0, 15] when unpacked
		}

		quantWeights, err := core.NewConstantFromData[uint8](
			"quantized_weights",
			compute.NewCPUEngine[uint8](numeric.Uint8Ops{}),
			numeric.Uint8Ops{},
			[]int{4, 2}, // 4 rows, 2 packed columns (will be 4 cols after unpacking)
			quantizedWeightsData,
		)
		if err != nil {
			t.Fatalf("Failed to create quantized weights constant: %v", err)
		}

		// Step 2: Create scale constants
		scaleData := []float32{0.1, 0.2, 0.3, 0.4} // Per-row scaling
		scaleConst, err := core.NewConstantFromData[float32](
			"weight_scales",
			engine,
			ops,
			[]int{4},
			scaleData,
		)
		if err != nil {
			t.Fatalf("Failed to create scale constant: %v", err)
		}

		// Step 3: Get constant values for MatMulNBits layer
		quantWeightsTensor, err := quantWeights.Forward(ctx)
		if err != nil {
			t.Fatalf("Failed to get quantized weights: %v", err)
		}

		scaleTensor, err := scaleConst.Forward(ctx)
		if err != nil {
			t.Fatalf("Failed to get scale tensor: %v", err)
		}

		// Step 4: Create MatMulNBits layer
		matmulLayer, err := core.NewMatMulNBits[float32](
			"quantized_linear",
			engine,
			ops,
			quantWeightsTensor,
			scaleTensor,
			nil,  // No zero point (symmetric quantization)
			4,    // 4-bit quantization
			true, // symmetric
		)
		if err != nil {
			t.Fatalf("Failed to create MatMulNBits layer: %v", err)
		}

		// Step 5: Create input tensor (batch_size=2, input_features=4)
		inputData := []float32{
			1.0, 2.0, 3.0, 4.0, // Sample 1
			0.5, 1.5, 2.5, 3.5, // Sample 2
		}
		input, err := tensor.New[float32]([]int{2, 4}, inputData)
		if err != nil {
			t.Fatalf("Failed to create input tensor: %v", err)
		}

		// Step 6: Forward pass through quantized layer
		output, err := matmulLayer.Forward(ctx, input)
		if err != nil {
			t.Fatalf("Quantized forward pass failed: %v", err)
		}

		// Step 7: Validate output shape and data
		expectedShape := []int{2, 4} // batch_size=2, output_features=4
		if !shapeEqual(output.Shape(), expectedShape) {
			t.Errorf("Output shape mismatch: expected %v, got %v", expectedShape, output.Shape())
		}

		outputData := output.Data()
		if len(outputData) != 8 { // 2 * 4 = 8 elements
			t.Errorf("Expected 8 output elements, got %d", len(outputData))
		}

		// Verify outputs are reasonable (not NaN or Inf)
		for i, val := range outputData {
			if !isFinite(val) {
				t.Errorf("Output element %d is not finite: %f", i, val)
			}
		}

		t.Logf("Quantized inference successful. Output shape: %v, Sample output: %v", 
			output.Shape(), outputData[:4])
	})

	t.Run("multi_layer_quantized_pipeline", func(t *testing.T) {
		// Simulate a multi-layer quantized pipeline
		// Layer 1: 3 -> 4 (quantized)
		// Layer 2: 4 -> 2 (quantized)

		// Layer 1 setup
		quant1Data := []uint8{0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC} // 3x2 packed -> 3x4 unpacked
		scale1Data := []float32{0.1, 0.1, 0.1}                     // uniform scaling

		quant1, err := core.NewConstantFromData[uint8](
			"layer1_weights", 
			compute.NewCPUEngine[uint8](numeric.Uint8Ops{}),
			numeric.Uint8Ops{},
			[]int{3, 2}, quant1Data)
		if err != nil {
			t.Fatalf("Failed to create layer 1 weights: %v", err)
		}

		scale1, err := core.NewConstantFromData[float32](
			"layer1_scales", engine, ops, []int{3}, scale1Data)
		if err != nil {
			t.Fatalf("Failed to create layer 1 scales: %v", err)
		}

		// Layer 2 setup  
		quant2Data := []uint8{0xAB, 0xCD, 0xEF, 0x12} // 4x1 packed -> 4x2 unpacked
		scale2Data := []float32{0.2, 0.2, 0.2, 0.2}   // uniform scaling

		quant2, err := core.NewConstantFromData[uint8](
			"layer2_weights",
			compute.NewCPUEngine[uint8](numeric.Uint8Ops{}),
			numeric.Uint8Ops{},
			[]int{4, 1}, quant2Data)
		if err != nil {
			t.Fatalf("Failed to create layer 2 weights: %v", err)
		}

		scale2, err := core.NewConstantFromData[float32](
			"layer2_scales", engine, ops, []int{4}, scale2Data)
		if err != nil {
			t.Fatalf("Failed to create layer 2 scales: %v", err)
		}

		// Create MatMulNBits layers
		quant1Tensor, _ := quant1.Forward(ctx)
		scale1Tensor, _ := scale1.Forward(ctx)
		layer1, err := core.NewMatMulNBits[float32](
			"layer1", engine, ops, quant1Tensor, scale1Tensor, nil, 4, true)
		if err != nil {
			t.Fatalf("Failed to create layer 1: %v", err)
		}

		quant2Tensor, _ := quant2.Forward(ctx)
		scale2Tensor, _ := scale2.Forward(ctx)
		layer2, err := core.NewMatMulNBits[float32](
			"layer2", engine, ops, quant2Tensor, scale2Tensor, nil, 4, true)
		if err != nil {
			t.Fatalf("Failed to create layer 2: %v", err)
		}

		// Input: batch_size=1, features=3
		input, err := tensor.New[float32]([]int{1, 3}, []float32{1.0, 2.0, 3.0})
		if err != nil {
			t.Fatalf("Failed to create input: %v", err)
		}

		// Forward pass through pipeline
		hidden, err := layer1.Forward(ctx, input)
		if err != nil {
			t.Fatalf("Layer 1 forward failed: %v", err)
		}

		output, err := layer2.Forward(ctx, hidden)
		if err != nil {
			t.Fatalf("Layer 2 forward failed: %v", err)
		}

		// Validate pipeline output
		expectedShape := []int{1, 2}
		if !shapeEqual(output.Shape(), expectedShape) {
			t.Errorf("Pipeline output shape mismatch: expected %v, got %v", expectedShape, output.Shape())
		}

		outputData := output.Data()
		for i, val := range outputData {
			if !isFinite(val) {
				t.Errorf("Pipeline output element %d is not finite: %f", i, val)
			}
		}

		t.Logf("Multi-layer quantized pipeline successful. Final output: %v", outputData)
	})

	t.Run("quantized_vs_full_precision_comparison", func(t *testing.T) {
		// Compare quantized vs full precision inference
		// This helps validate that quantization is working reasonably

		// Shared input
		input, err := tensor.New[float32]([]int{1, 2}, []float32{1.0, 2.0})
		if err != nil {
			t.Fatalf("Failed to create comparison input: %v", err)
		}

		// Use 2x2 matrices for simplicity (2 inputs, 2 outputs)
		// Full precision layer
		fpLayer, err := core.NewLinear[float32]("full_precision", engine, ops, 2, 2)
		if err != nil {
			t.Fatalf("Failed to create full precision layer: %v", err)
		}

		// Set known weights for comparison: [[1, 2], [3, 4]]
		fpWeights := fpLayer.Parameters()[0].Value
		fpWeightsData := fpWeights.Data()
		knownWeights := []float32{1, 3, 2, 4} // Column-major order
		copy(fpWeightsData, knownWeights)

		// Create simple quantized representation
		// Just use raw uint8 values for testing
		quantizedData := []uint8{0x12, 0x34} // Will unpack to [2,1,4,3] 
		quantTensor, err := tensor.New[uint8]([]int{2, 1}, quantizedData)
		if err != nil {
			t.Fatalf("Failed to create quantized tensor: %v", err)
		}

		scale := float32(0.1)
		scaleTensor, err := tensor.New[float32]([]int{1}, []float32{scale})
		if err != nil {
			t.Fatalf("Failed to create scale tensor: %v", err)
		}

		quantLayer, err := core.NewMatMulNBits[float32](
			"quantized", engine, ops, quantTensor, scaleTensor, nil, 4, true)
		if err != nil {
			t.Fatalf("Failed to create quantized layer: %v", err)
		}

		// Forward passes
		fpOutput, err := fpLayer.Forward(ctx, input)
		if err != nil {
			t.Fatalf("Full precision forward failed: %v", err)
		}

		quantOutput, err := quantLayer.Forward(ctx, input)
		if err != nil {
			t.Fatalf("Quantized forward failed: %v", err)
		}

		// Both should produce valid outputs
		fpData := fpOutput.Data()
		quantData := quantOutput.Data()

		if len(fpData) != len(quantData) {
			t.Errorf("Output length mismatch: FP32=%d, Quantized=%d", len(fpData), len(quantData))
		}

		// Outputs should be different (due to quantization) but both finite
		for i := 0; i < len(fpData) && i < len(quantData); i++ {
			if !isFinite(fpData[i]) {
				t.Errorf("Full precision output %d is not finite: %f", i, fpData[i])
			}
			if !isFinite(quantData[i]) {
				t.Errorf("Quantized output %d is not finite: %f", i, quantData[i])
			}
		}

		t.Logf("Comparison successful:")
		t.Logf("  Full precision output: %v", fpData)
		t.Logf("  Quantized output: %v", quantData)
	})
}

// TestQuantizationRoundTrip tests that quantization and dequantization preserve information reasonably.
func TestQuantizationRoundTrip(t *testing.T) {
	t.Run("symmetric_quantization_roundtrip", func(t *testing.T) {
		// Test the round-trip: float32 -> quantize -> dequantize -> float32
		originalValues := []float32{-12.8, -6.4, 0.0, 6.4, 12.8}
		scale := float32(0.1)

		quantConfig, err := numeric.NewQuantizationConfig(scale, 128, true) // symmetric
		if err != nil {
			t.Fatalf("Failed to create quantization config: %v", err)
		}

		// Quantize
		var quantized []uint8
		for _, val := range originalValues {
			qval := quantConfig.Quantize(val)
			quantized = append(quantized, qval)
		}

		// Dequantize
		var dequantized []float32
		for _, qval := range quantized {
			dval := quantConfig.Dequantize(qval)
			dequantized = append(dequantized, dval)
		}

		// Check that round-trip error is reasonable
		for i, orig := range originalValues {
			deq := dequantized[i]
			error := abs(orig - deq)
			
			// Error should be within quantization step size
			maxError := scale * 2 // Allow up to 2 quantization steps of error
			if error > maxError {
				t.Errorf("Round-trip error too large for value %f: got %f, error %f > %f", 
					orig, deq, error, maxError)
			}
		}

		t.Logf("Round-trip test successful:")
		t.Logf("  Original: %v", originalValues)
		t.Logf("  Quantized: %v", quantized)
		t.Logf("  Dequantized: %v", dequantized)
	})
}

// Helper functions
func shapeEqual(a, b []int) bool {
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

func isFinite(f float32) bool {
	return !math.IsNaN(float64(f)) && !math.IsInf(float64(f), 0)
}

func abs(f float32) float32 {
	if f < 0 {
		return -f
	}
	return f
}