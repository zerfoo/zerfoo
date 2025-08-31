package numeric

import (
	"fmt"
	"math"
)

// QuantizationConfig holds parameters for quantization operations
type QuantizationConfig struct {
	Scale     float32
	ZeroPoint int64
	Symmetric bool // If true, zero_point is ignored and assumed to be 0
}

// NewQuantizationConfig creates a quantization configuration with validation
func NewQuantizationConfig(scale float32, zeroPoint int64, symmetric bool) (*QuantizationConfig, error) {
	if scale <= 0 {
		return nil, fmt.Errorf("quantization scale must be positive, got %f", scale)
	}
	
	if !symmetric && (zeroPoint < 0 || zeroPoint > 255) {
		return nil, fmt.Errorf("zero point must be in range [0, 255] for asymmetric quantization, got %d", zeroPoint)
	}
	
	return &QuantizationConfig{
		Scale:     scale,
		ZeroPoint: zeroPoint,
		Symmetric: symmetric,
	}, nil
}

// Quantize converts float32 values to uint8 using linear quantization
// Formula: quantized = round(value / scale + zero_point)
func (qc *QuantizationConfig) Quantize(value float32) uint8 {
	var zeroPoint float32
	if qc.Symmetric {
		// For symmetric quantization, zero point is at 128 (middle of uint8 range)
		zeroPoint = 128.0
	} else {
		zeroPoint = float32(qc.ZeroPoint)
	}
	
	quantized := value/qc.Scale + zeroPoint
	
	// Round and clamp to uint8 range
	rounded := float32(math.Round(float64(quantized)))
	if rounded < 0 {
		return 0
	}
	if rounded > 255 {
		return 255
	}
	return uint8(rounded)
}

// QuantizeSlice quantizes a slice of float32 values to uint8
func (qc *QuantizationConfig) QuantizeSlice(values []float32) []uint8 {
	result := make([]uint8, len(values))
	for i, v := range values {
		result[i] = qc.Quantize(v)
	}
	return result
}

// Dequantize converts uint8 values back to float32 using linear dequantization
// Formula: dequantized = scale * (quantized - zero_point)
func (qc *QuantizationConfig) Dequantize(quantized uint8) float32 {
	var zeroPoint float32
	if qc.Symmetric {
		// For symmetric quantization, zero point is at 128 (middle of uint8 range)
		zeroPoint = 128.0
	} else {
		zeroPoint = float32(qc.ZeroPoint)
	}
	
	return qc.Scale * (float32(quantized) - zeroPoint)
}

// DequantizeSlice dequantizes a slice of uint8 values to float32
func (qc *QuantizationConfig) DequantizeSlice(quantized []uint8) []float32 {
	result := make([]float32, len(quantized))
	for i, q := range quantized {
		result[i] = qc.Dequantize(q)
	}
	return result
}

// Pack4BitWeights packs two 4-bit values into a single uint8
// This is used for MatMulNBits where weights are stored as 4-bit values
func Pack4BitWeights(low4, high4 uint8) (uint8, error) {
	if low4 > 15 {
		return 0, fmt.Errorf("low 4-bit value must be in range [0, 15], got %d", low4)
	}
	if high4 > 15 {
		return 0, fmt.Errorf("high 4-bit value must be in range [0, 15], got %d", high4)
	}
	
	return (high4 << 4) | low4, nil
}

// Unpack4BitWeights extracts two 4-bit values from a single uint8
// Returns (low4, high4) where low4 is bits [0:3] and high4 is bits [4:7]
func Unpack4BitWeights(packed uint8) (uint8, uint8) {
	low4 := packed & 0x0F  // Extract lower 4 bits
	high4 := packed >> 4   // Extract upper 4 bits
	return low4, high4
}

// Unpack4BitSlice unpacks a slice of uint8 values into 4-bit weights
// Each uint8 contains two 4-bit values, so output length is 2x input length
func Unpack4BitSlice(packed []uint8) []uint8 {
	result := make([]uint8, len(packed)*2)
	for i, p := range packed {
		low, high := Unpack4BitWeights(p)
		result[i*2] = low
		result[i*2+1] = high
	}
	return result
}

// Pack4BitSlice packs a slice of 4-bit values into uint8 array
// Input length must be even. Each pair of values is packed into one uint8.
func Pack4BitSlice(values []uint8) ([]uint8, error) {
	if len(values)%2 != 0 {
		return nil, fmt.Errorf("input length must be even for 4-bit packing, got %d", len(values))
	}
	
	result := make([]uint8, len(values)/2)
	for i := 0; i < len(values); i += 2 {
		packed, err := Pack4BitWeights(values[i], values[i+1])
		if err != nil {
			return nil, fmt.Errorf("failed to pack values at indices %d,%d: %w", i, i+1, err)
		}
		result[i/2] = packed
	}
	return result, nil
}

// Dequantize4BitWeights combines 4-bit unpacking with dequantization
// This is the typical operation for MatMulNBits: unpack 4-bit -> dequantize -> float32
func (qc *QuantizationConfig) Dequantize4BitWeights(packed []uint8) []float32 {
	// First unpack 4-bit values to uint8
	unpacked := Unpack4BitSlice(packed)
	
	// Then dequantize to float32
	return qc.DequantizeSlice(unpacked)
}

// ComputeQuantizationParams computes optimal scale and zero_point for a given data range
// This is useful for dynamic quantization where parameters aren't predetermined
func ComputeQuantizationParams(minVal, maxVal float32, symmetric bool) (*QuantizationConfig, error) {
	if minVal > maxVal {
		return nil, fmt.Errorf("min value (%f) cannot be greater than max value (%f)", minVal, maxVal)
	}
	
	if minVal == maxVal {
		// Handle edge case where all values are the same
		return NewQuantizationConfig(1.0, 128, symmetric)
	}
	
	if symmetric {
		// Symmetric quantization: zero_point at 128, scale covers [-absMax, absMax]
		// Map [-absMax, absMax] to [1, 255] with 128 as zero
		absMax := float32(math.Max(math.Abs(float64(minVal)), math.Abs(float64(maxVal))))
		scale := absMax / 127.0 // Map [-absMax, absMax] to [-127, 127], centered at 128
		return NewQuantizationConfig(scale, 0, true)
	}
	
	// Asymmetric quantization: map [minVal, maxVal] to [0, 255]
	scale := (maxVal - minVal) / 255.0
	zeroPoint := int64(math.Round(float64(-minVal / scale)))
	
	// Clamp zero_point to valid uint8 range
	if zeroPoint < 0 {
		zeroPoint = 0
	}
	if zeroPoint > 255 {
		zeroPoint = 255
	}
	
	return NewQuantizationConfig(scale, zeroPoint, false)
}

// QuantizationError computes the quantization error (RMS) between original and quantized values
func (qc *QuantizationConfig) QuantizationError(original []float32) float64 {
	if len(original) == 0 {
		return 0.0
	}
	
	// Quantize and dequantize
	quantized := qc.QuantizeSlice(original)
	dequantized := qc.DequantizeSlice(quantized)
	
	// Compute RMS error
	sumSquaredError := 0.0
	for i, orig := range original {
		diff := float64(orig - dequantized[i])
		sumSquaredError += diff * diff
	}
	
	return math.Sqrt(sumSquaredError / float64(len(original)))
}

// ValidateQuantizationRoundTrip checks that quantize->dequantize preserves values within tolerance
func (qc *QuantizationConfig) ValidateQuantizationRoundTrip(values []float32, tolerance float64) error {
	quantized := qc.QuantizeSlice(values)
	dequantized := qc.DequantizeSlice(quantized)
	
	for i, orig := range values {
		diff := math.Abs(float64(orig - dequantized[i]))
		if diff > tolerance {
			return fmt.Errorf("quantization round-trip failed at index %d: original=%.6f, dequantized=%.6f, error=%.6f > tolerance=%.6f",
				i, orig, dequantized[i], diff, tolerance)
		}
	}
	return nil
}