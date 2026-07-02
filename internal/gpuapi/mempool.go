package gpuapi

import "unsafe"

// MemPool abstracts a GPU device memory pool with size-bucketed caching.
// Each vendor can reuse the same pool logic since the pool operates on
// opaque device pointers, but the underlying Malloc/Free come from the
// vendor's Runtime.
type MemPool interface {
	// Alloc returns a device pointer of at least byteSize bytes on the given device.
	// May return a cached pointer from a previous Free call.
	Alloc(deviceID, byteSize int) (unsafe.Pointer, error)
	// Free returns a device pointer to the pool for reuse.
	Free(deviceID int, ptr unsafe.Pointer, byteSize int)
	// AllocManaged returns a unified memory pointer accessible from both host
	// and device. Returns an error on backends that do not support managed memory.
	AllocManaged(deviceID, byteSize int) (unsafe.Pointer, error)
	// FreeManaged returns a managed memory pointer to the pool for reuse.
	FreeManaged(deviceID int, ptr unsafe.Pointer, byteSize int)
	// Drain frees all cached pointers back to the device.
	Drain() error
	// Stats returns the number of cached allocations and their total bytes.
	Stats() (allocations int, totalBytes int)
}
