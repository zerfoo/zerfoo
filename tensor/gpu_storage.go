//go:build cuda

package tensor

import (
	"unsafe"

	"github.com/zerfoo/zerfoo/device"
	"github.com/zerfoo/zerfoo/internal/cuda"
)

// GPUStorage is a CUDA device-backed Storage implementation.
// Slice() copies data from the GPU to a new CPU slice (not zero-copy).
// Set() copies data from a CPU slice to the GPU.
type GPUStorage[T Numeric] struct {
	devicePtr unsafe.Pointer // CUDA device pointer from cudaMalloc
	length    int            // number of elements
	byteSize  int            // total bytes = length * sizeof(T)
}

// NewGPUStorage allocates CUDA device memory for the given number of elements.
func NewGPUStorage[T Numeric](length int) (*GPUStorage[T], error) {
	var zero T
	elemSize := int(unsafe.Sizeof(zero))
	byteSize := length * elemSize

	devPtr, err := cuda.Malloc(byteSize)
	if err != nil {
		return nil, err
	}

	return &GPUStorage[T]{
		devicePtr: devPtr,
		length:    length,
		byteSize:  byteSize,
	}, nil
}

// NewGPUStorageFromSlice allocates CUDA device memory and copies data from a CPU slice.
func NewGPUStorageFromSlice[T Numeric](data []T) (*GPUStorage[T], error) {
	s, err := NewGPUStorage[T](len(data))
	if err != nil {
		return nil, err
	}

	if len(data) > 0 {
		src := unsafe.Pointer(unsafe.SliceData(data))
		if err := cuda.Memcpy(s.devicePtr, src, s.byteSize, cuda.MemcpyHostToDevice); err != nil {
			// Clean up on failure
			_ = cuda.Free(s.devicePtr)

			return nil, err
		}
	}

	return s, nil
}

// Len returns the number of elements.
func (s *GPUStorage[T]) Len() int { return s.length }

// Slice copies device memory to a new CPU slice and returns it.
func (s *GPUStorage[T]) Slice() []T {
	if s.length == 0 {
		return []T{}
	}

	host := make([]T, s.length)
	dst := unsafe.Pointer(unsafe.SliceData(host))

	// Memcpy errors are not expected here; panic to surface bugs early.
	if err := cuda.Memcpy(dst, s.devicePtr, s.byteSize, cuda.MemcpyDeviceToHost); err != nil {
		panic("GPUStorage.Slice: " + err.Error())
	}

	return host
}

// Set copies data from a CPU slice to the GPU, replacing the current contents.
// If the new slice has a different length, the old device memory is freed and
// new memory is allocated.
func (s *GPUStorage[T]) Set(data []T) {
	var zero T
	elemSize := int(unsafe.Sizeof(zero))
	newByteSize := len(data) * elemSize

	if len(data) != s.length {
		// Reallocate
		_ = cuda.Free(s.devicePtr)

		ptr, err := cuda.Malloc(newByteSize)
		if err != nil {
			panic("GPUStorage.Set: " + err.Error())
		}

		s.devicePtr = ptr
		s.length = len(data)
		s.byteSize = newByteSize
	}

	if len(data) > 0 {
		src := unsafe.Pointer(unsafe.SliceData(data))
		if err := cuda.Memcpy(s.devicePtr, src, s.byteSize, cuda.MemcpyHostToDevice); err != nil {
			panic("GPUStorage.Set: " + err.Error())
		}
	}
}

// DeviceType returns device.CUDA.
func (s *GPUStorage[T]) DeviceType() device.Type { return device.CUDA }

// Ptr returns the raw CUDA device pointer.
func (s *GPUStorage[T]) Ptr() unsafe.Pointer { return s.devicePtr }

// Free releases the CUDA device memory. After calling Free, the storage must
// not be used.
func (s *GPUStorage[T]) Free() error {
	if s.devicePtr == nil {
		return nil
	}

	err := cuda.Free(s.devicePtr)
	s.devicePtr = nil
	s.length = 0
	s.byteSize = 0

	return err
}

// Statically assert that GPUStorage satisfies the Storage interface.
var _ Storage[float32] = (*GPUStorage[float32])(nil)
