package tensor

import (
	"encoding/binary"
	"unsafe"

	"github.com/zerfoo/float16"
	"github.com/zerfoo/zerfoo/device"
)

// Float16Storage holds IEEE 754 half-precision (FP16) tensor data.
// Each element is stored as 2 bytes in little-endian order.
// This is a native FP16 storage type — no per-tensor scaling is needed.
type Float16Storage struct {
	data []byte
	len  int

	gpuPtr      unsafe.Pointer
	gpuByteSize int
	gpuDeviceID int
}

// NewFloat16StorageFromF32 converts float32 data to FP16 and returns a Float16Storage.
func NewFloat16StorageFromF32(src []float32) *Float16Storage {
	n := len(src)
	if n == 0 {
		return &Float16Storage{len: 0}
	}

	data := make([]byte, n*2)
	for i, v := range src {
		bits := uint16(float16.FromFloat32(v))
		binary.LittleEndian.PutUint16(data[i*2:], bits)
	}

	return &Float16Storage{data: data, len: n}
}

// Len returns the number of logical FP16 elements.
func (s *Float16Storage) Len() int { return s.len }

// Slice decodes FP16 data to float32.
// Returns a zero-filled slice if no host data is available (GPU-only storage).
func (s *Float16Storage) Slice() []float32 {
	if len(s.data) == 0 {
		return make([]float32, s.len)
	}
	dst := make([]float32, s.len)
	for i := 0; i < s.len; i++ {
		bits := binary.LittleEndian.Uint16(s.data[i*2:])
		dst[i] = float16.Float16(bits).ToFloat32()
	}
	return dst
}

// Set encodes float32 data into FP16 format.
func (s *Float16Storage) Set(data []float32) {
	*s = *NewFloat16StorageFromF32(data)
}

// DeviceType returns device.CPU.
func (s *Float16Storage) DeviceType() device.Type { return device.CPU }

// RawBytes returns the raw FP16 data as a byte slice (2 bytes per element).
func (s *Float16Storage) RawBytes() []byte { return s.data }

// SubSlice returns a zero-copy view into the storage from element offset for length elements.
// The caller must ensure the parent outlives the returned view.
func (s *Float16Storage) SubSlice(offset, length int) *Float16Storage {
	byteOffset := offset * 2
	byteEnd := byteOffset + length*2
	return &Float16Storage{
		data:        s.data[byteOffset:byteEnd],
		len:         length,
		gpuPtr:      nil,
		gpuByteSize: 0,
		gpuDeviceID: s.gpuDeviceID,
	}
}

// SetGPUPtr stores a pre-uploaded GPU device pointer for the raw FP16 bytes.
func (s *Float16Storage) SetGPUPtr(ptr unsafe.Pointer, byteSize, deviceID int) {
	s.gpuPtr = ptr
	s.gpuByteSize = byteSize
	s.gpuDeviceID = deviceID
}

// GPUPtr returns the cached GPU device pointer, byte size, and device ID.
// Returns nil if no GPU copy exists.
func (s *Float16Storage) GPUPtr() (unsafe.Pointer, int, int) {
	return s.gpuPtr, s.gpuByteSize, s.gpuDeviceID
}

// SetGPUByteSize updates the GPU byte size. This is useful when the GPU
// allocation size differs from the logical byte size (e.g. padded allocations).
func (s *Float16Storage) SetGPUByteSize(byteSize int) {
	s.gpuByteSize = byteSize
}

// NewFloat16StorageGPU creates a Float16Storage backed by a GPU device pointer.
// The storage has no host data; only the GPU pointer is set.
func NewFloat16StorageGPU(ptr unsafe.Pointer, numElems, deviceID int) *Float16Storage {
	byteSize := numElems * 2
	return &Float16Storage{
		len:         numElems,
		gpuPtr:      ptr,
		gpuByteSize: byteSize,
		gpuDeviceID: deviceID,
	}
}

// Ensure Float16Storage implements Storage[float32].
var _ Storage[float32] = (*Float16Storage)(nil)
