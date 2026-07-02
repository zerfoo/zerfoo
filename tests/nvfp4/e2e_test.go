//go:build integration

package nvfp4_test

import (
	"math"
	"math/rand/v2"
	"testing"

	"github.com/zerfoo/zerfoo/training/fp8"
)

// TestNVFP4GemmaE2E exercises the full NVFP4 quantize → dequantize roundtrip
// on a tensor shaped like a Gemma 3 1B embedding row (dim=1152) filled with
// mock weights. It asserts that the output shape is preserved and that the
// mean absolute quantization error stays below a tight threshold.
func TestNVFP4GemmaE2E(t *testing.T) {
	const (
		rows      = 4    // small batch for test speed
		cols      = 1152 // Gemma 3 1B hidden dim
		maxMaeErr = 0.1  // acceptance: mean absolute error < 0.1
	)

	// Generate mock weights in [-1, 1] (typical post-LayerNorm range).
	rng := rand.New(rand.NewPCG(42, 0))
	weights := make([]float32, rows*cols)
	for i := range weights {
		weights[i] = rng.Float32()*2 - 1 // uniform [-1, 1)
	}

	// Quantize and dequantize each block.
	reconstructed := make([]float32, len(weights))
	blockSize := fp8.NVFP4BlockSize

	for start := 0; start < len(weights); start += blockSize {
		end := start + blockSize
		if end > len(weights) {
			end = len(weights)
		}
		block := weights[start:end]

		qBlock, scale := fp8.QuantizeBlockNVFP4(block)
		dBlock := fp8.DequantizeBlockNVFP4(qBlock, scale)

		if len(dBlock) != len(block) {
			t.Fatalf("block at offset %d: shape mismatch: got %d, want %d",
				start, len(dBlock), len(block))
		}
		copy(reconstructed[start:end], dBlock)
	}

	// Verify output tensor shape.
	if len(reconstructed) != rows*cols {
		t.Fatalf("output shape mismatch: got %d elements, want %d", len(reconstructed), rows*cols)
	}

	// Compute mean absolute error.
	var sumAbsErr float64
	for i, orig := range weights {
		sumAbsErr += math.Abs(float64(orig - reconstructed[i]))
	}
	mae := sumAbsErr / float64(len(weights))

	t.Logf("NVFP4 roundtrip MAE: %.6f (threshold: %.1f)", mae, maxMaeErr)
	if mae >= maxMaeErr {
		t.Errorf("mean absolute error %.6f exceeds threshold %.1f", mae, maxMaeErr)
	}
}

// TestNVFP4QuantizeDequantizeExact verifies that specific known values
// roundtrip correctly through NVFP4 quantization.
func TestNVFP4QuantizeDequantizeExact(t *testing.T) {
	tests := []struct {
		name  string
		input float32
		want  float32
	}{
		{"zero", 0, 0},
		{"half", 0.5, 0.5},
		{"one", 1.0, 1.0},
		{"one_point_five", 1.5, 1.5},
		{"two", 2.0, 2.0},
		{"three", 3.0, 3.0},
		{"four", 4.0, 4.0},
		{"six", 6.0, 6.0},
		{"neg_one", -1.0, -1.0},
		{"neg_six", -6.0, -6.0},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			q := fp8.QuantizeToNVFP4(tc.input)
			got := fp8.DequantizeNVFP4(q)
			if got != tc.want {
				t.Errorf("QuantizeToNVFP4(%g) → DequantizeNVFP4 = %g, want %g",
					tc.input, got, tc.want)
			}
		})
	}
}

// TestNVFP4Saturation verifies that values beyond the representable range
// are clamped to ±6.0.
func TestNVFP4Saturation(t *testing.T) {
	tests := []struct {
		name  string
		input float32
		want  float32
	}{
		{"large_positive", 100.0, 6.0},
		{"large_negative", -100.0, -6.0},
		{"pos_inf", float32(math.Inf(1)), 6.0},
		{"neg_inf", float32(math.Inf(-1)), -6.0},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			q := fp8.QuantizeToNVFP4(tc.input)
			got := fp8.DequantizeNVFP4(q)
			if got != tc.want {
				t.Errorf("QuantizeToNVFP4(%g) → DequantizeNVFP4 = %g, want %g",
					tc.input, got, tc.want)
			}
		})
	}
}
