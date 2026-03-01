//go:build cuda

package tensor

import (
	"fmt"
	"log"
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

// TrySlice copies device memory to a new CPU slice.
// Returns an error if the D2H copy fails instead of panicking.
func (s *GPUStorage[T]) TrySlice() ([]T, error) {
	if s.length == 0 {
		return []T{}, nil
	}

	host := make([]T, s.length)
	dst := unsafe.Pointer(unsafe.SliceData(host))

	if err := cuda.Memcpy(dst, s.devicePtr, s.byteSize, cuda.MemcpyDeviceToHost); err != nil {
		return nil, fmt.Errorf("GPUStorage.TrySlice: %w", err)
	}

	return host, nil
}

// Slice copies device memory to a new CPU slice and returns it.
// On error, logs a warning and returns a zero-valued slice.
func (s *GPUStorage[T]) Slice() []T {
	data, err := s.TrySlice()
	if err != nil {
		log.Printf("WARNING: %v; returning zero slice of length %d", err, s.length)

		return make([]T, s.length)
	}

	return data
}

// TrySet copies data from a CPU slice to the GPU, replacing the current contents.
// If the new slice has a different length, the old device memory is freed and
// new memory is allocated. Returns an error instead of panicking on failure.
func (s *GPUStorage[T]) TrySet(data []T) error {
	var zero T
	elemSize := int(unsafe.Sizeof(zero))
	newByteSize := len(data) * elemSize

	if len(data) != s.length {
		_ = cuda.Free(s.devicePtr)

		ptr, err := cuda.Malloc(newByteSize)
		if err != nil {
			s.devicePtr = nil
			s.length = 0
			s.byteSize = 0

			return fmt.Errorf("GPUStorage.TrySet: malloc: %w", err)
		}

		s.devicePtr = ptr
		s.length = len(data)
		s.byteSize = newByteSize
	}

	if len(data) > 0 {
		src := unsafe.Pointer(unsafe.SliceData(data))
		if err := cuda.Memcpy(s.devicePtr, src, s.byteSize, cuda.MemcpyHostToDevice); err != nil {
			return fmt.Errorf("GPUStorage.TrySet: memcpy: %w", err)
		}
	}

	return nil
}

// Set copies data from a CPU slice to the GPU, replacing the current contents.
// On error, logs a warning instead of panicking.
func (s *GPUStorage[T]) Set(data []T) {
	if err := s.TrySet(data); err != nil {
		log.Printf("WARNING: %v", err)
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
