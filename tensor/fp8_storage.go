package tensor

import (
	"math"
	"unsafe"

	"github.com/zerfoo/zerfoo/device"
)

// FP8E4M3Storage holds FP8 E4M3 quantized tensor data on CPU.
// Uses per-tensor absmax scaling: fp8_value = float32_value / scale.
type FP8E4M3Storage struct {
	data  []byte
	scale float32
	len   int

	// GPU pointer cache (like BFloat16Storage).
	gpuPtr      unsafe.Pointer
	gpuByteSize int
	gpuDeviceID int

	// GPU pointer for the per-tensor scale factor (single float32 on device).
	scaleGPUPtr unsafe.Pointer
}

// NewFP8E4M3Storage quantizes float32 data into FP8 E4M3 format with absmax scaling.
func NewFP8E4M3Storage(src []float32) *FP8E4M3Storage {
	n := len(src)
	if n == 0 {
		return &FP8E4M3Storage{len: 0}
	}

	// Compute absmax scale. E4M3 max representable value is 448.
	const maxFP8 float32 = 448.0
	var absMax float32
	for _, v := range src {
		if av := float32(math.Abs(float64(v))); av > absMax {
			absMax = av
		}
	}

	var scale float32
	if absMax > 0 {
		scale = absMax / maxFP8
	}

	data := make([]byte, n)
	if scale > 0 {
		invScale := 1.0 / scale
		for i, v := range src {
			scaled := v * invScale
			data[i] = encodeE4M3(scaled)
		}
	}

	return &FP8E4M3Storage{data: data, scale: scale, len: n}
}

// Len returns the number of logical float32 elements.
func (s *FP8E4M3Storage) Len() int { return s.len }

// Slice decodes FP8 E4M3 data to float32 by multiplying by the scale factor.
func (s *FP8E4M3Storage) Slice() []float32 {
	dst := make([]float32, s.len)
	for i, b := range s.data {
		dst[i] = decodeE4M3(b) * s.scale
	}
	return dst
}

// Set encodes float32 data into FP8 E4M3 format.
func (s *FP8E4M3Storage) Set(data []float32) {
	*s = *NewFP8E4M3Storage(data)
}

// DeviceType returns device.CPU.
func (s *FP8E4M3Storage) DeviceType() device.Type { return device.CPU }

// Scale returns the per-tensor scale factor.
func (s *FP8E4M3Storage) Scale() float32 { return s.scale }

// RawBytes returns the raw FP8 data as a byte slice (1 byte per element).
func (s *FP8E4M3Storage) RawBytes() []byte { return s.data }

// SetGPUPtr stores a pre-uploaded GPU device pointer for the raw FP8 bytes.
func (s *FP8E4M3Storage) SetGPUPtr(ptr unsafe.Pointer, byteSize, deviceID int) {
	s.gpuPtr = ptr
	s.gpuByteSize = byteSize
	s.gpuDeviceID = deviceID
}

// GPUPtr returns the cached GPU device pointer, byte size, and device ID.
// Returns nil if no GPU copy exists.
func (s *FP8E4M3Storage) GPUPtr() (unsafe.Pointer, int, int) {
	return s.gpuPtr, s.gpuByteSize, s.gpuDeviceID
}

// SetScaleGPUPtr stores the GPU device pointer for the per-tensor scale factor.
func (s *FP8E4M3Storage) SetScaleGPUPtr(ptr unsafe.Pointer) {
	s.scaleGPUPtr = ptr
}

// ScaleGPUPtr returns the GPU device pointer for the per-tensor scale factor.
func (s *FP8E4M3Storage) ScaleGPUPtr() unsafe.Pointer {
	return s.scaleGPUPtr
}

// Ensure FP8E4M3Storage implements Storage[float32].
var _ Storage[float32] = (*FP8E4M3Storage)(nil)

// FP8E5M2Storage holds FP8 E5M2 quantized tensor data on CPU.
// Uses per-tensor absmax scaling: fp8_value = float32_value / scale.
type FP8E5M2Storage struct {
	data  []byte
	scale float32
	len   int
}

// NewFP8E5M2Storage quantizes float32 data into FP8 E5M2 format with absmax scaling.
func NewFP8E5M2Storage(src []float32) *FP8E5M2Storage {
	n := len(src)
	if n == 0 {
		return &FP8E5M2Storage{len: 0}
	}

	// Compute absmax scale. E5M2 max representable value is 57344.
	const maxFP8 float32 = 57344.0
	var absMax float32
	for _, v := range src {
		if av := float32(math.Abs(float64(v))); av > absMax {
			absMax = av
		}
	}

	var scale float32
	if absMax > 0 {
		scale = absMax / maxFP8
	}

	data := make([]byte, n)
	if scale > 0 {
		invScale := 1.0 / scale
		for i, v := range src {
			scaled := v * invScale
			data[i] = encodeE5M2(scaled)
		}
	}

	return &FP8E5M2Storage{data: data, scale: scale, len: n}
}

// Len returns the number of logical float32 elements.
func (s *FP8E5M2Storage) Len() int { return s.len }

// Slice decodes FP8 E5M2 data to float32 by multiplying by the scale factor.
func (s *FP8E5M2Storage) Slice() []float32 {
	dst := make([]float32, s.len)
	for i, b := range s.data {
		dst[i] = decodeE5M2(b) * s.scale
	}
	return dst
}

// Set encodes float32 data into FP8 E5M2 format.
func (s *FP8E5M2Storage) Set(data []float32) {
	*s = *NewFP8E5M2Storage(data)
}

// DeviceType returns device.CPU.
func (s *FP8E5M2Storage) DeviceType() device.Type { return device.CPU }

// Scale returns the per-tensor scale factor.
func (s *FP8E5M2Storage) Scale() float32 { return s.scale }

// Ensure FP8E5M2Storage implements Storage[float32].
var _ Storage[float32] = (*FP8E5M2Storage)(nil)

// encodeE4M3 converts a float32 to an E4M3 byte.
// E4M3: 1 sign, 4 exponent (bias=7), 3 mantissa bits.
// Max finite value: 448, no infinity/NaN — all 256 bit patterns are finite.
func encodeE4M3(f float32) byte {
	if f == 0 {
		return 0x00
	}

	bits := math.Float32bits(f)
	sign := byte((bits >> 31) & 1)
	exp32 := int((bits>>23)&0xFF) - 127
	mant32 := bits & 0x7FFFFF

	const (
		e4m3Bias   = 7
		e4m3MantBits = 3
	)

	biasedExp := exp32 + e4m3Bias

	// E4M3 has a 4-bit exponent field; valid biased exponents are 0-15.
	// biasedExp > 15 means overflow.
	if biasedExp > 15 {
		// Overflow -> max finite value (0x7E = 0_1111_110 = 448)
		return (sign << 7) | 0x7E
	}
	if biasedExp <= 0 {
		// Subnormal or underflow -> zero
		return sign << 7
	}

	// Round mantissa to 3 bits with round-to-nearest-even.
	mant3 := mant32 >> (23 - e4m3MantBits)
	remainder := mant32 & ((1 << (23 - e4m3MantBits)) - 1)
	halfway := uint32(1 << (23 - e4m3MantBits - 1))
	if remainder > halfway || (remainder == halfway && mant3&1 != 0) {
		mant3++
		if mant3 >= (1 << e4m3MantBits) {
			mant3 = 0
			biasedExp++
			if biasedExp > 15 {
				return (sign << 7) | 0x7E
			}
		}
	}

	return (sign << 7) | byte(biasedExp<<e4m3MantBits) | byte(mant3&0x07)
}

// decodeE4M3 converts an E4M3 byte to float32.
// All bit patterns are treated as finite numbers (no infinity/NaN).
func decodeE4M3(b byte) float32 {
	sign := uint32(b>>7) & 1
	exp := int((b >> 3) & 0x0F)
	mant := uint32(b & 0x07)

	if exp == 0 && mant == 0 {
		return math.Float32frombits(sign << 31) // +-0
	}
	if exp == 0 {
		// Subnormal: value = (-1)^sign * 2^(1-bias) * (0.mant / 8)
		val := float32(mant) / 8.0 * float32(math.Pow(2, float64(1-7)))
		if sign != 0 {
			val = -val
		}
		return val
	}

	// Normal: convert to float32 components.
	exp32 := uint32(exp - 7 + 127)
	mant32 := mant << (23 - 3)
	return math.Float32frombits((sign << 31) | (exp32 << 23) | mant32)
}

// encodeE5M2 converts a float32 to an E5M2 byte.
// E5M2: 1 sign, 5 exponent (bias=15), 2 mantissa bits.
// Max finite value: 57344, min subnormal: 2^-16.
func encodeE5M2(f float32) byte {
	if f == 0 {
		if math.Signbit(float64(f)) {
			return 0x80
		}
		return 0x00
	}

	bits := math.Float32bits(f)
	sign := byte((bits >> 31) & 1)
	exp32 := int((bits>>23)&0xFF) - 127 // unbias float32 exponent
	mant32 := bits & 0x7FFFFF

	// Clamp to E5M2 range.
	const (
		e5m2Bias   = 15
		e5m2ExpMax = 15 // max biased exponent for finite values (30 is max biased, but 31 = inf/nan)
	)

	biasedExp := exp32 + e5m2Bias

	if biasedExp >= 31 {
		// Overflow -> max finite value (0x7B = 0_11110_11)
		return (sign << 7) | 0x7B
	}
	if biasedExp <= 0 {
		// Subnormal or underflow -> zero
		return sign << 7
	}

	// Round mantissa to 2 bits with round-to-nearest-even.
	mant2 := mant32 >> (23 - 2)
	remainder := mant32 & ((1 << (23 - 2)) - 1)
	halfway := uint32(1 << (23 - 2 - 1))
	if remainder > halfway || (remainder == halfway && mant2&1 != 0) {
		mant2++
		if mant2 >= (1 << 2) {
			mant2 = 0
			biasedExp++
			if biasedExp >= 31 {
				return (sign << 7) | 0x7B
			}
		}
	}

	return (sign << 7) | byte(biasedExp<<2) | byte(mant2&0x03)
}

// decodeE5M2 converts an E5M2 byte to float32.
func decodeE5M2(b byte) float32 {
	sign := uint32(b>>7) & 1
	exp := int((b >> 2) & 0x1F)
	mant := uint32(b & 0x03)

	if exp == 0 && mant == 0 {
		return math.Float32frombits(sign << 31) // +-0
	}
	if exp == 31 {
		if mant == 0 {
			return math.Float32frombits((sign << 31) | (0xFF << 23)) // +-inf
		}
		return float32(math.NaN())
	}
	if exp == 0 {
		// Subnormal: value = (-1)^sign * 2^(1-bias) * (0.mant / 4)
		val := float32(mant) / 4.0 * float32(math.Pow(2, float64(1-15)))
		if sign != 0 {
			val = -val
		}
		return val
	}

	// Normal: convert to float32 components.
	exp32 := uint32(exp - 15 + 127)
	mant32 := mant << (23 - 2)
	return math.Float32frombits((sign << 31) | (exp32 << 23) | mant32)
}
