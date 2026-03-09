package device

import (
	"fmt"
	"unsafe"

	"github.com/zerfoo/zerfoo/internal/cuda"
)

// cudaAllocator is the memory allocator for a specific CUDA device.
type cudaAllocator struct {
	deviceID int
}

// NewCUDAAllocator creates a new CUDA device memory allocator bound to the
// given device.
func NewCUDAAllocator(deviceID int) Allocator {
	return &cudaAllocator{deviceID: deviceID}
}

// Allocate allocates size bytes of CUDA device memory and returns the device pointer.
// SetDevice is called before cudaMalloc to ensure allocation on the correct GPU.
func (a *cudaAllocator) Allocate(size int) (any, error) {
	if size < 0 {
		return nil, fmt.Errorf("allocation size cannot be negative: %d", size)
	}

	if size == 0 {
		return unsafe.Pointer(nil), nil
	}

	if err := cuda.SetDevice(a.deviceID); err != nil {
		return nil, fmt.Errorf("CUDA SetDevice(%d) failed: %w", a.deviceID, err)
	}

	ptr, err := cuda.Malloc(size)
	if err != nil {
		return nil, fmt.Errorf("CUDA allocation of %d bytes on device %d failed: %w", size, a.deviceID, err)
	}

	return ptr, nil
}

// Free releases CUDA device memory previously allocated with Allocate.
// SetDevice is called before cudaFree to ensure correct device context.
func (a *cudaAllocator) Free(ptr any) error {
	devPtr, ok := ptr.(unsafe.Pointer)
	if !ok {
		return fmt.Errorf("expected unsafe.Pointer, got %T", ptr)
	}

	if devPtr == nil {
		return nil
	}

	if err := cuda.SetDevice(a.deviceID); err != nil {
		return fmt.Errorf("CUDA SetDevice(%d) failed: %w", a.deviceID, err)
	}

	return cuda.Free(devPtr)
}

// Statically assert that the type implements the interface.
var _ Allocator = (*cudaAllocator)(nil)
