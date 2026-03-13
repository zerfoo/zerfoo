package tensor

import (
	"unsafe"

	"github.com/zerfoo/float16"
	"github.com/zerfoo/zerfoo/device"
)

// BFloat16Storage holds float32 tensor data in BFloat16 format on CPU.
// It implements Storage[float32] so that models can use BF16 weights
// with FP32 activations (mixed-precision inference). The raw BF16 bytes
// can be uploaded to GPU for use with cublasGemmEx.
type BFloat16Storage struct {
	data []uint16 // BF16 values stored as raw uint16
	len  int

	// GPU pointer cache (like Q4Storage / Q8Storage).
	gpuPtr      unsafe.Pointer
	gpuByteSize int
	gpuDeviceID int
}

// NewBFloat16Storage converts float32 data to BFloat16 format.
func NewBFloat16Storage(src []float32) *BFloat16Storage {
	n := len(src)
	data := make([]uint16, n)
	for i, v := range src {
		data[i] = uint16(float16.BFloat16FromFloat32(v))
	}
	return &BFloat16Storage{data: data, len: n}
}

// NewBFloat16StorageFromRaw creates a BFloat16Storage from pre-encoded uint16 values.
func NewBFloat16StorageFromRaw(data []uint16) *BFloat16Storage {
	return &BFloat16Storage{data: data, len: len(data)}
}

// Len returns the number of logical float32 elements.
func (s *BFloat16Storage) Len() int { return s.len }

// Slice decodes BFloat16 data to float32.
func (s *BFloat16Storage) Slice() []float32 {
	dst := make([]float32, s.len)
	for i, v := range s.data {
		dst[i] = float16.BFloat16(v).ToFloat32()
	}
	return dst
}

// Set encodes float32 data into BFloat16 format.
func (s *BFloat16Storage) Set(data []float32) {
	*s = *NewBFloat16Storage(data)
}

// DeviceType returns device.CPU.
func (s *BFloat16Storage) DeviceType() device.Type { return device.CPU }

// RawBytes returns the raw BF16 data as a byte slice (2 bytes per element).
func (s *BFloat16Storage) RawBytes() []byte {
	if len(s.data) == 0 {
		return nil
	}
	return unsafe.Slice((*byte)(unsafe.Pointer(&s.data[0])), len(s.data)*2)
}

// SetGPUPtr stores a pre-uploaded GPU device pointer for the raw BF16 bytes.
func (s *BFloat16Storage) SetGPUPtr(ptr unsafe.Pointer, byteSize, deviceID int) {
	s.gpuPtr = ptr
	s.gpuByteSize = byteSize
	s.gpuDeviceID = deviceID
}

// GPUPtr returns the cached GPU device pointer, byte size, and device ID.
// Returns nil if no GPU copy exists.
func (s *BFloat16Storage) GPUPtr() (unsafe.Pointer, int, int) {
	return s.gpuPtr, s.gpuByteSize, s.gpuDeviceID
}

// Ensure BFloat16Storage implements Storage[float32].
var _ Storage[float32] = (*BFloat16Storage)(nil)
