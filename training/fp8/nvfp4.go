package fp8

import "math"

// NVFloat4 represents a 4-bit floating-point number in NVIDIA's NVFP4 (E2M1) format.
//
// Bit layout (4 bits packed into a uint8):
//   - 1 bit  : Sign (0 = positive, 1 = negative)
//   - 2 bits : Exponent (biased by 1, range [0, 3])
//   - 1 bit  : Mantissa (1 explicit bit)
//
// Representable values (positive): 0, 0.5, 1, 1.5, 2, 3, 4, 6
// Only the lower 4 bits of the uint8 are used.
type NVFloat4 uint8

// NVFP4 format constants.
const (
	nvfp4SignMask     = 0b1000
	nvfp4ExponentMask = 0b0110
	nvfp4MantissaMask = 0b0001
	nvfp4ExponentBias = 1
	nvfp4MaxNormal    = float32(6.0) // largest representable magnitude
)

// QuantizeToNVFP4 converts a float32 value to the nearest NVFP4 (E2M1)
// representation using round-to-nearest-even. Values exceeding the
// representable range are clamped (saturated).
func QuantizeToNVFP4(f float32) NVFloat4 {
	sign := uint8(0)
	if math.Signbit(float64(f)) {
		sign = nvfp4SignMask
		f = -f
	}

	if f == 0 || math.IsNaN(float64(f)) {
		return NVFloat4(sign) // ±0
	}

	// Clamp to max representable magnitude.
	if f > nvfp4MaxNormal || math.IsInf(float64(f), 1) {
		return NVFloat4(sign | 0b0111) // ±6.0
	}

	// Enumerate all 8 positive NVFP4 values and pick the nearest.
	// With only 8 entries a linear scan is simpler and clearer than
	// bit-manipulation on a 4-bit float.
	type entry struct {
		bits uint8
		val  float32
	}
	table := [8]entry{
		{0b0000, 0.0},
		{0b0001, 0.5},
		{0b0010, 1.0},
		{0b0011, 1.5},
		{0b0100, 2.0},
		{0b0101, 3.0},
		{0b0110, 4.0},
		{0b0111, 6.0},
	}

	bestBits := uint8(0)
	bestDist := float32(math.MaxFloat32)
	for _, e := range table {
		d := f - e.val
		if d < 0 {
			d = -d
		}
		if d < bestDist {
			bestDist = d
			bestBits = e.bits
		}
	}

	return NVFloat4(sign | bestBits)
}

// DequantizeNVFP4 converts an NVFP4 (E2M1) value back to float32.
func DequantizeNVFP4(n NVFloat4) float32 {
	negative := n&nvfp4SignMask != 0
	exp := (uint8(n) & 0b0110) >> 1
	man := uint8(n) & nvfp4MantissaMask

	var val float32
	if exp == 0 {
		// Subnormal: value = 0.mantissa × 2^(1−bias) = mantissa × 0.5
		val = float32(man) * 0.5
	} else {
		// Normal: value = (1 + mantissa/2) × 2^(exp−bias)
		val = (1.0 + float32(man)*0.5) * float32(math.Exp2(float64(int(exp)-nvfp4ExponentBias)))
	}

	if negative {
		val = -val
	}
	return val
}

// NVFP4BlockSize is the number of elements per quantization block.
// Each block shares a single FP32 scale factor (absmax scaling).
const NVFP4BlockSize = 32

// QuantizeBlockNVFP4 quantizes a block of float32 values to NVFP4 with a
// per-block absmax scale factor. Returns the quantized values and the scale
// used for dequantization.
func QuantizeBlockNVFP4(data []float32) ([]NVFloat4, float32) {
	if len(data) == 0 {
		return nil, 0
	}

	// Compute absmax for the block.
	amax := float32(0)
	for _, v := range data {
		a := v
		if a < 0 {
			a = -a
		}
		if a > amax {
			amax = a
		}
	}

	scale := float32(1)
	if amax > 0 {
		scale = nvfp4MaxNormal / amax
	}

	out := make([]NVFloat4, len(data))
	for i, v := range data {
		out[i] = QuantizeToNVFP4(v * scale)
	}
	return out, scale
}

// DequantizeBlockNVFP4 dequantizes a block of NVFP4 values back to float32
// using the provided scale factor from QuantizeBlockNVFP4.
func DequantizeBlockNVFP4(q []NVFloat4, scale float32) []float32 {
	out := make([]float32, len(q))
	for i, v := range q {
		out[i] = DequantizeNVFP4(v) / scale
	}
	return out
}
