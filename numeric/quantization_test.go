package numeric

import (
	"math"
	"testing"
)

func TestNewQuantizationConfig(t *testing.T) {
	tests := []struct {
		name      string
		scale     float32
		zeroPoint int64
		symmetric bool
		expectErr bool
	}{
		{
			name:      "valid_asymmetric",
			scale:     0.5,
			zeroPoint: 128,
			symmetric: false,
			expectErr: false,
		},
		{
			name:      "valid_symmetric",
			scale:     1.0,
			zeroPoint: 0,
			symmetric: true,
			expectErr: false,
		},
		{
			name:      "invalid_scale_zero",
			scale:     0.0,
			zeroPoint: 128,
			symmetric: false,
			expectErr: true,
		},
		{
			name:      "invalid_scale_negative",
			scale:     -0.5,
			zeroPoint: 128,
			symmetric: false,
			expectErr: true,
		},
		{
			name:      "invalid_zero_point_negative",
			scale:     1.0,
			zeroPoint: -1,
			symmetric: false,
			expectErr: true,
		},
		{
			name:      "invalid_zero_point_too_large",
			scale:     1.0,
			zeroPoint: 256,
			symmetric: false,
			expectErr: true,
		},
		{
			name:      "symmetric_ignores_zero_point",
			scale:     1.0,
			zeroPoint: 1000, // Should be ignored for symmetric
			symmetric: true,
			expectErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			config, err := NewQuantizationConfig(tt.scale, tt.zeroPoint, tt.symmetric)

			if tt.expectErr {
				if err == nil {
					t.Errorf("Expected error but got none")
				}
				return
			}

			if err != nil {
				t.Errorf("Unexpected error: %v", err)
				return
			}

			if config.Scale != tt.scale {
				t.Errorf("Scale mismatch: expected %f, got %f", tt.scale, config.Scale)
			}

			if config.ZeroPoint != tt.zeroPoint {
				t.Errorf("ZeroPoint mismatch: expected %d, got %d", tt.zeroPoint, config.ZeroPoint)
			}

			if config.Symmetric != tt.symmetric {
				t.Errorf("Symmetric mismatch: expected %t, got %t", tt.symmetric, config.Symmetric)
			}
		})
	}
}

func TestQuantizeDequantize(t *testing.T) {
	tests := []struct {
		name         string
		scale        float32
		zeroPoint    int64
		symmetric    bool
		testValues   []float32
		tolerance    float64
	}{
		{
			name:      "basic_asymmetric",
			scale:     0.1,
			zeroPoint: 128,
			symmetric: false,
			testValues: []float32{-12.8, -6.4, 0.0, 6.4, 12.7},
			tolerance:  0.1,
		},
		{
			name:      "basic_symmetric",
			scale:     0.2,
			zeroPoint: 0,
			symmetric: true,
			testValues: []float32{-25.0, -12.5, 0.0, 12.5, 25.0},
			tolerance:  0.2,
		},
		{
			name:      "edge_values",
			scale:     1.0,
			zeroPoint: 0,
			symmetric: true,
			testValues: []float32{-127.0, -1.0, 0.0, 1.0, 127.0},
			tolerance:  1.0,
		},
		{
			name:      "small_scale",
			scale:     0.01,
			zeroPoint: 100,
			symmetric: false,
			testValues: []float32{-1.0, -0.5, 0.0, 0.5, 1.0},
			tolerance:  0.01,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			config, err := NewQuantizationConfig(tt.scale, tt.zeroPoint, tt.symmetric)
			if err != nil {
				t.Fatalf("Failed to create config: %v", err)
			}

			for _, value := range tt.testValues {
				// Test individual quantize/dequantize
				quantized := config.Quantize(value)
				dequantized := config.Dequantize(quantized)

				diff := math.Abs(float64(value - dequantized))
				if diff > tt.tolerance {
					t.Errorf("Round-trip failed for value %f: quantized=%d, dequantized=%f, error=%f > tolerance=%f",
						value, quantized, dequantized, diff, tt.tolerance)
				}
			}

			// Test slice operations
			quantizedSlice := config.QuantizeSlice(tt.testValues)
			dequantizedSlice := config.DequantizeSlice(quantizedSlice)

			if len(dequantizedSlice) != len(tt.testValues) {
				t.Errorf("Slice length mismatch: expected %d, got %d", len(tt.testValues), len(dequantizedSlice))
			}

			for i, orig := range tt.testValues {
				diff := math.Abs(float64(orig - dequantizedSlice[i]))
				if diff > tt.tolerance {
					t.Errorf("Slice round-trip failed at index %d: original=%f, dequantized=%f, error=%f",
						i, orig, dequantizedSlice[i], diff)
				}
			}
		})
	}
}

func TestPack4BitWeights(t *testing.T) {
	tests := []struct {
		name      string
		low4      uint8
		high4     uint8
		expected  uint8
		expectErr bool
	}{
		{
			name:     "valid_zero",
			low4:     0,
			high4:    0,
			expected: 0x00,
		},
		{
			name:     "valid_max",
			low4:     15,
			high4:    15,
			expected: 0xFF,
		},
		{
			name:     "mixed_values",
			low4:     5,
			high4:    10,
			expected: 0xA5, // 1010 0101
		},
		{
			name:     "asymmetric",
			low4:     3,
			high4:    12,
			expected: 0xC3, // 1100 0011
		},
		{
			name:      "invalid_low",
			low4:      16,
			high4:     5,
			expectErr: true,
		},
		{
			name:      "invalid_high",
			low4:      5,
			high4:     16,
			expectErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := Pack4BitWeights(tt.low4, tt.high4)

			if tt.expectErr {
				if err == nil {
					t.Errorf("Expected error but got none")
				}
				return
			}

			if err != nil {
				t.Errorf("Unexpected error: %v", err)
				return
			}

			if result != tt.expected {
				t.Errorf("Pack result mismatch: expected 0x%02X, got 0x%02X", tt.expected, result)
			}

			// Test round-trip with unpack
			unpackedLow, unpackedHigh := Unpack4BitWeights(result)
			if unpackedLow != tt.low4 {
				t.Errorf("Round-trip low mismatch: expected %d, got %d", tt.low4, unpackedLow)
			}
			if unpackedHigh != tt.high4 {
				t.Errorf("Round-trip high mismatch: expected %d, got %d", tt.high4, unpackedHigh)
			}
		})
	}
}

func TestUnpack4BitWeights(t *testing.T) {
	tests := []struct {
		name         string
		packed       uint8
		expectedLow  uint8
		expectedHigh uint8
	}{
		{
			name:         "zero",
			packed:       0x00,
			expectedLow:  0,
			expectedHigh: 0,
		},
		{
			name:         "max",
			packed:       0xFF,
			expectedLow:  15,
			expectedHigh: 15,
		},
		{
			name:         "mixed",
			packed:       0xA5, // 1010 0101
			expectedLow:  5,    // 0101
			expectedHigh: 10,   // 1010
		},
		{
			name:         "asymmetric",
			packed:       0x3C, // 0011 1100
			expectedLow:  12,   // 1100
			expectedHigh: 3,    // 0011
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			low, high := Unpack4BitWeights(tt.packed)

			if low != tt.expectedLow {
				t.Errorf("Low 4-bit mismatch: expected %d, got %d", tt.expectedLow, low)
			}

			if high != tt.expectedHigh {
				t.Errorf("High 4-bit mismatch: expected %d, got %d", tt.expectedHigh, high)
			}
		})
	}
}

func TestPack4BitSlice(t *testing.T) {
	tests := []struct {
		name      string
		input     []uint8
		expected  []uint8
		expectErr bool
	}{
		{
			name:     "empty",
			input:    []uint8{},
			expected: []uint8{},
		},
		{
			name:     "single_pair",
			input:    []uint8{3, 12},
			expected: []uint8{0xC3}, // high=12, low=3
		},
		{
			name:     "multiple_pairs",
			input:    []uint8{1, 2, 3, 4, 5, 6},
			expected: []uint8{0x21, 0x43, 0x65}, // [2,1], [4,3], [6,5]
		},
		{
			name:      "odd_length",
			input:     []uint8{1, 2, 3},
			expectErr: true,
		},
		{
			name:      "invalid_value",
			input:     []uint8{16, 5},
			expectErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := Pack4BitSlice(tt.input)

			if tt.expectErr {
				if err == nil {
					t.Errorf("Expected error but got none")
				}
				return
			}

			if err != nil {
				t.Errorf("Unexpected error: %v", err)
				return
			}

			if len(result) != len(tt.expected) {
				t.Errorf("Result length mismatch: expected %d, got %d", len(tt.expected), len(result))
			}

			for i, expected := range tt.expected {
				if result[i] != expected {
					t.Errorf("Result mismatch at index %d: expected 0x%02X, got 0x%02X", i, expected, result[i])
				}
			}

			// Test round-trip if no error expected
			if !tt.expectErr && len(tt.input) > 0 {
				unpacked := Unpack4BitSlice(result)
				if len(unpacked) != len(tt.input) {
					t.Errorf("Round-trip length mismatch: expected %d, got %d", len(tt.input), len(unpacked))
				}

				for i, expected := range tt.input {
					if unpacked[i] != expected {
						t.Errorf("Round-trip mismatch at index %d: expected %d, got %d", i, expected, unpacked[i])
					}
				}
			}
		})
	}
}

func TestDequantize4BitWeights(t *testing.T) {
	// Test the combined operation of unpacking 4-bit weights and dequantizing
	config, err := NewQuantizationConfig(0.5, 8, false) // scale=0.5, zeroPoint=8
	if err != nil {
		t.Fatalf("Failed to create config: %v", err)
	}

	// Pack some test values: [2, 3, 10, 15] -> two uint8 values
	packed, err := Pack4BitSlice([]uint8{2, 3, 10, 15})
	if err != nil {
		t.Fatalf("Failed to pack test values: %v", err)
	}

	// Dequantize using the combined function
	dequantized := config.Dequantize4BitWeights(packed)

	// Expected results: scale * (value - zeroPoint)
	expected := []float32{
		0.5 * (2 - 8),  // -3.0
		0.5 * (3 - 8),  // -2.5
		0.5 * (10 - 8), // 1.0
		0.5 * (15 - 8), // 3.5
	}

	if len(dequantized) != len(expected) {
		t.Errorf("Length mismatch: expected %d, got %d", len(expected), len(dequantized))
	}

	for i, exp := range expected {
		if math.Abs(float64(dequantized[i]-exp)) > 1e-6 {
			t.Errorf("Dequantized value mismatch at index %d: expected %f, got %f", i, exp, dequantized[i])
		}
	}
}

func TestComputeQuantizationParams(t *testing.T) {
	tests := []struct {
		name      string
		minVal    float32
		maxVal    float32
		symmetric bool
		expectErr bool
	}{
		{
			name:      "valid_asymmetric",
			minVal:    -10.0,
			maxVal:    20.0,
			symmetric: false,
		},
		{
			name:      "valid_symmetric",
			minVal:    -15.0,
			maxVal:    15.0,
			symmetric: true,
		},
		{
			name:      "single_value",
			minVal:    5.0,
			maxVal:    5.0,
			symmetric: false,
		},
		{
			name:      "invalid_range",
			minVal:    10.0,
			maxVal:    5.0,
			expectErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			config, err := ComputeQuantizationParams(tt.minVal, tt.maxVal, tt.symmetric)

			if tt.expectErr {
				if err == nil {
					t.Errorf("Expected error but got none")
				}
				return
			}

			if err != nil {
				t.Errorf("Unexpected error: %v", err)
				return
			}

			// Verify that the configuration can handle the input range
			if tt.minVal != tt.maxVal {
				quantizedMin := config.Quantize(tt.minVal)
				quantizedMax := config.Quantize(tt.maxVal)

				dequantizedMin := config.Dequantize(quantizedMin)
				dequantizedMax := config.Dequantize(quantizedMax)

				// Check that the range is preserved reasonably well
				minError := math.Abs(float64(tt.minVal - dequantizedMin))
				maxError := math.Abs(float64(tt.maxVal - dequantizedMax))

				tolerance := float64(config.Scale * 2) // Allow up to 2 quantization steps error
				if minError > tolerance {
					t.Errorf("Min value error too large: %f > %f", minError, tolerance)
				}
				if maxError > tolerance {
					t.Errorf("Max value error too large: %f > %f", maxError, tolerance)
				}
			}
		})
	}
}

func TestQuantizationError(t *testing.T) {
	config, err := NewQuantizationConfig(0.1, 128, false)
	if err != nil {
		t.Fatalf("Failed to create config: %v", err)
	}

	tests := []struct {
		name     string
		values   []float32
		maxError float64
	}{
		{
			name:     "zero_values",
			values:   []float32{0, 0, 0},
			maxError: 0.1,
		},
		{
			name:     "small_values",
			values:   []float32{0.05, 0.15, 0.25},
			maxError: 0.1,
		},
		{
			name:     "mixed_values",
			values:   []float32{-5.0, 0.0, 5.0, 10.0},
			maxError: 0.2,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if len(tt.values) == 0 {
				return
			}

			error := config.QuantizationError(tt.values)
			if error > tt.maxError {
				t.Errorf("Quantization error too large: %f > %f", error, tt.maxError)
			}

			// Error should be non-negative
			if error < 0 {
				t.Errorf("Quantization error should be non-negative, got %f", error)
			}
		})
	}

	// Test empty slice
	error := config.QuantizationError([]float32{})
	if error != 0.0 {
		t.Errorf("Empty slice should have zero error, got %f", error)
	}
}

func TestValidateQuantizationRoundTrip(t *testing.T) {
	config, err := NewQuantizationConfig(0.05, 100, false)
	if err != nil {
		t.Fatalf("Failed to create config: %v", err)
	}

	tests := []struct {
		name      string
		values    []float32
		tolerance float64
		expectErr bool
	}{
		{
			name:      "values_within_tolerance",
			values:    []float32{1.0, 2.0, 3.0},
			tolerance: 0.1,
			expectErr: false,
		},
		{
			name:      "values_exceed_tolerance",
			values:    []float32{0.001, 0.002, 0.003}, // Very small values, large relative error
			tolerance: 1e-6,                          // Very tight tolerance
			expectErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := config.ValidateQuantizationRoundTrip(tt.values, tt.tolerance)

			if tt.expectErr {
				if err == nil {
					t.Errorf("Expected validation error but got none")
				}
			} else {
				if err != nil {
					t.Errorf("Unexpected validation error: %v", err)
				}
			}
		})
	}
}