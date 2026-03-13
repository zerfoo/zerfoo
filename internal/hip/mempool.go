package hip

import (
	"sync"
	"unsafe"
)

// MemPool is a per-device, size-bucketed free-list allocator for HIP device
// memory. It caches freed allocations by (deviceID, byteSize) for reuse,
// avoiding the overhead of hipMalloc/hipFree on every operation and
// preventing cross-device pointer reuse in multi-GPU setups.
type MemPool struct {
	mu    sync.Mutex
	cache map[int]map[int][]unsafe.Pointer // deviceID -> byteSize -> list of free device pointers
}

// NewMemPool creates a new empty memory pool.
func NewMemPool() *MemPool {
	return &MemPool{
		cache: make(map[int]map[int][]unsafe.Pointer),
	}
}

// Alloc returns a device pointer of the given byte size on the specified
// device. If a cached allocation of the exact (deviceID, byteSize) exists,
// it is reused. Otherwise SetDevice is called and a fresh hipMalloc is
// performed.
func (p *MemPool) Alloc(deviceID, byteSize int) (unsafe.Pointer, error) {
	p.mu.Lock()
	if devCache := p.cache[deviceID]; devCache != nil {
		if ptrs := devCache[byteSize]; len(ptrs) > 0 {
			ptr := ptrs[len(ptrs)-1]
			devCache[byteSize] = ptrs[:len(ptrs)-1]
			p.mu.Unlock()

			return ptr, nil
		}
	}
	p.mu.Unlock()

	if err := SetDevice(deviceID); err != nil {
		return nil, err
	}

	return Malloc(byteSize)
}

// Free returns a device pointer to the pool for later reuse.
// The caller must provide the same deviceID and byteSize that were used
// to allocate.
func (p *MemPool) Free(deviceID int, ptr unsafe.Pointer, byteSize int) {
	p.mu.Lock()
	devCache := p.cache[deviceID]
	if devCache == nil {
		devCache = make(map[int][]unsafe.Pointer)
		p.cache[deviceID] = devCache
	}
	devCache[byteSize] = append(devCache[byteSize], ptr)
	p.mu.Unlock()
}

// Drain releases all cached device memory back to HIP. Iterates all
// devices, calling SetDevice before freeing each device's pointers.
// Returns the first error encountered, but attempts to free all pointers.
func (p *MemPool) Drain() error {
	p.mu.Lock()
	defer p.mu.Unlock()

	var firstErr error

	for deviceID, devCache := range p.cache {
		if err := SetDevice(deviceID); err != nil && firstErr == nil {
			firstErr = err
		}

		for size, ptrs := range devCache {
			for _, ptr := range ptrs {
				if err := Free(ptr); err != nil && firstErr == nil {
					firstErr = err
				}
			}

			delete(devCache, size)
		}

		delete(p.cache, deviceID)
	}

	return firstErr
}

// Stats returns the number of cached allocations and total cached bytes
// across all devices.
func (p *MemPool) Stats() (allocations int, totalBytes int) {
	p.mu.Lock()
	defer p.mu.Unlock()

	for _, devCache := range p.cache {
		for size, ptrs := range devCache {
			allocations += len(ptrs)
			totalBytes += size * len(ptrs)
		}
	}

	return allocations, totalBytes
}
