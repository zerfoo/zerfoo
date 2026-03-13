package tensor

import (
	"encoding/binary"
	"fmt"

	"github.com/zerfoo/float16"
	"github.com/zerfoo/zerfoo/device"
)

// BFloat16Storage stores tensor data in BFloat16 format (2 bytes per element)
// and implements Storage[float32]. Data is stored as raw little-endian bytes
// and decoded to float32 on Slice().
type BFloat16Storage struct {
	raw         []byte // little-endian BFloat16 values
	len         int    // number of elements
	cachedSlice []float32
}

// NewBFloat16StorageFromRaw creates a BFloat16Storage from raw bytes.
// Each element is 2 bytes in little-endian BFloat16 format.
func NewBFloat16StorageFromRaw(raw []byte, numElements int) (*BFloat16Storage, error) {
	if numElements <= 0 {
		return nil, fmt.Errorf("numElements must be positive, got %d", numElements)
	}
	need := numElements * 2
	if len(raw) < need {
		return nil, fmt.Errorf("BF16 raw data too short: need %d bytes for %d elements, got %d", need, numElements, len(raw))
	}
	// Copy raw bytes so caller can reuse buffer.
	data := make([]byte, need)
	copy(data, raw[:need])
	return &BFloat16Storage{raw: data, len: numElements}, nil
}

// NewBFloat16StorageFromFloat32 creates a BFloat16Storage by converting float32 data.
func NewBFloat16StorageFromFloat32(src []float32) *BFloat16Storage {
	raw := make([]byte, len(src)*2)
	for i, v := range src {
		bits := float16.BFloat16FromFloat32(v).Bits()
		binary.LittleEndian.PutUint16(raw[i*2:i*2+2], bits)
	}
	return &BFloat16Storage{raw: raw, len: len(src)}
}

// Len returns the number of elements.
func (s *BFloat16Storage) Len() int { return s.len }

// Slice returns a float32 view by decoding BFloat16 values.
// The result is cached after the first call.
func (s *BFloat16Storage) Slice() []float32 {
	if s.cachedSlice != nil {
		return s.cachedSlice
	}
	dst := make([]float32, s.len)
	for i := range s.len {
		bits := binary.LittleEndian.Uint16(s.raw[i*2 : i*2+2])
		dst[i] = float16.BFloat16FromBits(bits).ToFloat32()
	}
	s.cachedSlice = dst
	return dst
}

// Set encodes float32 values to BFloat16 and replaces the storage contents.
func (s *BFloat16Storage) Set(data []float32) {
	s.raw = make([]byte, len(data)*2)
	for i, v := range data {
		bits := float16.BFloat16FromFloat32(v).Bits()
		binary.LittleEndian.PutUint16(s.raw[i*2:i*2+2], bits)
	}
	s.len = len(data)
	s.cachedSlice = nil
}

// DeviceType returns device.CPU.
func (s *BFloat16Storage) DeviceType() device.Type { return device.CPU }

// ByteSize returns the raw byte size of the BFloat16 data.
func (s *BFloat16Storage) ByteSize() int { return s.len * 2 }

// Ensure BFloat16Storage implements Storage[float32].
var _ Storage[float32] = (*BFloat16Storage)(nil)
