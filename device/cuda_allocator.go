//go:build cuda

package device

import (
	"fmt"
	"unsafe"

	"github.com/zerfoo/zerfoo/internal/cuda"
)

// cudaAllocator is the memory allocator for CUDA devices.
type cudaAllocator struct{}

// NewCUDAAllocator creates a new CUDA device memory allocator.
func NewCUDAAllocator() Allocator {
	return &cudaAllocator{}
}

// Allocate allocates size bytes of CUDA device memory and returns the device pointer.
func (a *cudaAllocator) Allocate(size int) (any, error) {
	if size < 0 {
		return nil, fmt.Errorf("allocation size cannot be negative: %d", size)
	}

	if size == 0 {
		return unsafe.Pointer(nil), nil
	}

	ptr, err := cuda.Malloc(size)
	if err != nil {
		return nil, fmt.Errorf("CUDA allocation of %d bytes failed: %w", size, err)
	}

	return ptr, nil
}

// Free releases CUDA device memory previously allocated with Allocate.
func (a *cudaAllocator) Free(ptr any) error {
	devPtr, ok := ptr.(unsafe.Pointer)
	if !ok {
		return fmt.Errorf("expected unsafe.Pointer, got %T", ptr)
	}

	if devPtr == nil {
		return nil
	}

	return cuda.Free(devPtr)
}

// Statically assert that the type implements the interface.
var _ Allocator = (*cudaAllocator)(nil)
