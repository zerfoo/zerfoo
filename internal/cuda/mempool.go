package cuda

import (
	"sync"
	"sync/atomic"
	"unsafe"
)

var defaultPoolInst *MemPool

// DefaultMemPool returns a process-wide MemPool singleton.
// Returns nil if called before SetDefaultMemPool.
func DefaultMemPool() *MemPool {
	return defaultPoolInst
}

// SetDefaultMemPool registers a MemPool as the process-wide default.
// Typically called by GPUEngine during initialization.
func SetDefaultMemPool(p *MemPool) {
	defaultPoolInst = p
}

// MemPool is a per-device, size-bucketed free-list allocator for CUDA device
// memory. It caches freed allocations by (deviceID, byteSize) for reuse,
// avoiding the overhead of cudaMalloc/cudaFree on every operation and
// preventing cross-device pointer reuse in multi-GPU setups.
type MemPool struct {
	mu           sync.Mutex
	cache        map[int]map[int][]unsafe.Pointer // deviceID -> byteSize -> list of free device pointers
	managedCache map[int]map[int][]unsafe.Pointer // same structure for managed (unified) memory
	hits         atomic.Int64
	misses       atomic.Int64
	frees        atomic.Int64
}

// NewMemPool creates a new empty memory pool.
func NewMemPool() *MemPool {
	return &MemPool{
		cache:        make(map[int]map[int][]unsafe.Pointer),
		managedCache: make(map[int]map[int][]unsafe.Pointer),
	}
}

// bucketSize rounds byteSize up to the next reuse bucket.
// Sizes <= 256 are kept exact (these are typically small scalar or shape
// metadata). Sizes > 256 are rounded up to the next power of two, enabling
// cache reuse across slightly varying allocation sizes (e.g., attention
// intermediates that grow with kvSeqLen).
func bucketSize(byteSize int) int {
	const threshold = 256
	if byteSize <= threshold {
		return byteSize
	}
	// Round up to next power of 2.
	v := byteSize - 1
	v |= v >> 1
	v |= v >> 2
	v |= v >> 4
	v |= v >> 8
	v |= v >> 16
	v |= v >> 32
	return v + 1
}

// Alloc returns a device pointer of at least the given byte size on the
// specified device. Sizes >= 4KB are rounded up to power-of-2 buckets for
// better reuse across slightly varying allocation sizes. If a cached
// allocation exists for the bucket, it is reused. Otherwise SetDevice is
// called and a fresh cudaMalloc is performed at the bucketed size.
func (p *MemPool) Alloc(deviceID, byteSize int) (unsafe.Pointer, error) {
	bucket := bucketSize(byteSize)
	p.mu.Lock()
	if devCache := p.cache[deviceID]; devCache != nil {
		if ptrs := devCache[bucket]; len(ptrs) > 0 {
			ptr := ptrs[len(ptrs)-1]
			devCache[bucket] = ptrs[:len(ptrs)-1]
			p.mu.Unlock()
			p.hits.Add(1)

			return ptr, nil
		}
	}
	p.mu.Unlock()
	p.misses.Add(1)

	if err := SetDevice(deviceID); err != nil {
		return nil, err
	}

	return Malloc(bucket)
}

// Free returns a device pointer to the pool for later reuse.
// The byteSize is bucketed to match the Alloc bucket so the pointer
// can be found on the next Alloc of a similar size.
func (p *MemPool) Free(deviceID int, ptr unsafe.Pointer, byteSize int) {
	bucket := bucketSize(byteSize)
	p.frees.Add(1)
	p.mu.Lock()
	devCache := p.cache[deviceID]
	if devCache == nil {
		devCache = make(map[int][]unsafe.Pointer)
		p.cache[deviceID] = devCache
	}
	devCache[bucket] = append(devCache[bucket], ptr)
	p.mu.Unlock()
}

// AllocManaged returns a unified memory pointer of at least the given byte
// size. Uses the same power-of-2 bucketing as Alloc.
func (p *MemPool) AllocManaged(deviceID, byteSize int) (unsafe.Pointer, error) {
	bucket := bucketSize(byteSize)
	p.mu.Lock()
	if devCache := p.managedCache[deviceID]; devCache != nil {
		if ptrs := devCache[bucket]; len(ptrs) > 0 {
			ptr := ptrs[len(ptrs)-1]
			devCache[bucket] = ptrs[:len(ptrs)-1]
			p.mu.Unlock()

			return ptr, nil
		}
	}
	p.mu.Unlock()

	if err := SetDevice(deviceID); err != nil {
		return nil, err
	}

	return MallocManaged(bucket)
}

// FreeManaged returns a managed memory pointer to the pool for later reuse.
func (p *MemPool) FreeManaged(deviceID int, ptr unsafe.Pointer, byteSize int) {
	bucket := bucketSize(byteSize)
	p.mu.Lock()
	devCache := p.managedCache[deviceID]
	if devCache == nil {
		devCache = make(map[int][]unsafe.Pointer)
		p.managedCache[deviceID] = devCache
	}
	devCache[bucket] = append(devCache[bucket], ptr)
	p.mu.Unlock()
}

// Drain releases all cached device memory back to CUDA. Iterates all
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

	// Drain managed cache (cudaFree works for managed memory).
	for deviceID, devCache := range p.managedCache {
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

		delete(p.managedCache, deviceID)
	}

	return firstErr
}

// HitMissStats returns the cache hit, miss, and free counts since the pool
// was created. Used for diagnosing pool effectiveness.
func (p *MemPool) HitMissStats() (hits, misses, frees int64) {
	return p.hits.Load(), p.misses.Load(), p.frees.Load()
}

// ResetHitMissStats resets the cache hit/miss/free counters.
func (p *MemPool) ResetHitMissStats() {
	p.hits.Store(0)
	p.misses.Store(0)
	p.frees.Store(0)
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
