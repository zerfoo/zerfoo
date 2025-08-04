// Package device provides device abstraction and memory allocation interfaces.
package device

import "fmt"

// Allocator defines the interface for a memory allocator.
// It is responsible for allocating and freeing memory on a specific device.
type Allocator interface {
	// Allocate allocates a block of memory of the given size in bytes.
	// For the CPU, this will be a Go slice. For a GPU, it would be a device pointer.
	Allocate(size int) (any, error)
	// Free releases the allocated memory.
	// For the CPU allocator, this is a no-op as Go's garbage collector manages memory.
	Free(ptr any) error
}

// --- CPU Allocator ---

// cpuAllocator is the memory allocator for the CPU.
// It uses standard Go slices and relies on the Go garbage collector.
type cpuAllocator struct{}

// NewCPUAllocator creates a new CPU memory allocator.
func NewCPUAllocator() Allocator {
	return &cpuAllocator{}
}

// Allocate creates a new Go slice of the given size.
// Note: In a real implementation, this would allocate `size` bytes, but for
// our generic tensor, we deal with number of elements. The actual byte size
// depends on the type `T`, which is handled when the slice is created (`make([]T, size)`).
// This interface simplifies the concept. We'll allocate a slice of bytes for now.
func (a *cpuAllocator) Allocate(size int) (any, error) {
	if size < 0 {
		return nil, fmt.Errorf("allocation size cannot be negative: %d", size)
	}
	// We allocate a slice of bytes as a generic representation of memory.
	// The tensor will later hold a slice of a specific numeric type.
	return make([]byte, size), nil
}

// Free is a no-op for the CPU allocator because the Go garbage collector
// automatically manages memory for slices.
func (a *cpuAllocator) Free(_ any) error {
	// No-op
	return nil
}
