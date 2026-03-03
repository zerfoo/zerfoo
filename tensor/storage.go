package tensor

import (
	"fmt"

	"github.com/zerfoo/zerfoo/device"
)

// Storage abstracts over CPU and GPU tensor data storage.
// For CPU storage, Slice() returns the underlying slice directly (zero copy).
// For GPU storage, Slice() copies device memory to a new host slice.
type Storage[T Numeric] interface {
	// Len returns the number of elements.
	Len() int
	// Slice returns a CPU-accessible []T.
	Slice() []T
	// Set replaces the storage contents from a CPU slice.
	Set(data []T)
	// DeviceType returns the device type this storage resides on.
	DeviceType() device.Type
}

// NewWithStorage creates a TensorNumeric backed by the given Storage.
// This allows creating tensors with GPUStorage or any other Storage implementation.
func NewWithStorage[T Numeric](shape []int, s Storage[T]) (*TensorNumeric[T], error) {
	size := 1
	for _, d := range shape {
		if d < 0 {
			return nil, fmt.Errorf("invalid shape dimension: %d", d)
		}

		size *= d
	}

	if s.Len() != size {
		return nil, fmt.Errorf("storage length (%d) does not match tensor size (%d)", s.Len(), size)
	}

	strides := make([]int, len(shape))

	stride := 1
	for i := len(shape) - 1; i >= 0; i-- {
		strides[i] = stride
		stride *= shape[i]
	}

	return &TensorNumeric[T]{
		shape:   shape,
		strides: strides,
		storage: s,
	}, nil
}

// CPUStorage is a CPU-backed Storage implementation wrapping a Go slice.
// Slice() returns the underlying slice with zero copy.
type CPUStorage[T Numeric] struct {
	data []T
}

// NewCPUStorage creates a new CPUStorage wrapping the provided data slice.
func NewCPUStorage[T Numeric](data []T) *CPUStorage[T] {
	return &CPUStorage[T]{data: data}
}

// Len returns the number of elements.
func (s *CPUStorage[T]) Len() int { return len(s.data) }

// Slice returns the underlying data slice directly (zero copy).
func (s *CPUStorage[T]) Slice() []T { return s.data }

// Set replaces the underlying data slice.
func (s *CPUStorage[T]) Set(data []T) { s.data = data }

// DeviceType returns device.CPU.
func (s *CPUStorage[T]) DeviceType() device.Type { return device.CPU }
