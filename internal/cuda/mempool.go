//go:build cuda

package cuda

import (
	"sync"
	"unsafe"
)

// MemPool is a size-bucketed free-list allocator for CUDA device memory.
// It caches freed allocations by byte size for reuse, avoiding the overhead
// of cudaMalloc/cudaFree on every operation.
type MemPool struct {
	mu    sync.Mutex
	cache map[int][]unsafe.Pointer // byteSize -> list of free device pointers
}

// NewMemPool creates a new empty memory pool.
func NewMemPool() *MemPool {
	return &MemPool{
		cache: make(map[int][]unsafe.Pointer),
	}
}

// Alloc returns a device pointer of the given byte size. If a cached
// allocation of the exact size exists, it is reused. Otherwise a fresh
// cudaMalloc is performed.
func (p *MemPool) Alloc(byteSize int) (unsafe.Pointer, error) {
	p.mu.Lock()
	if ptrs := p.cache[byteSize]; len(ptrs) > 0 {
		ptr := ptrs[len(ptrs)-1]
		p.cache[byteSize] = ptrs[:len(ptrs)-1]
		p.mu.Unlock()

		return ptr, nil
	}
	p.mu.Unlock()

	return Malloc(byteSize)
}

// Free returns a device pointer to the pool for later reuse.
// The caller must provide the same byteSize that was used to allocate.
func (p *MemPool) Free(ptr unsafe.Pointer, byteSize int) {
	p.mu.Lock()
	p.cache[byteSize] = append(p.cache[byteSize], ptr)
	p.mu.Unlock()
}

// Drain releases all cached device memory back to CUDA. Returns the
// first error encountered, but attempts to free all pointers.
func (p *MemPool) Drain() error {
	p.mu.Lock()
	defer p.mu.Unlock()

	var firstErr error

	for size, ptrs := range p.cache {
		for _, ptr := range ptrs {
			if err := Free(ptr); err != nil && firstErr == nil {
				firstErr = err
			}
		}

		delete(p.cache, size)
	}

	return firstErr
}

// Stats returns the number of cached allocations and total cached bytes.
func (p *MemPool) Stats() (allocations int, totalBytes int) {
	p.mu.Lock()
	defer p.mu.Unlock()

	for size, ptrs := range p.cache {
		allocations += len(ptrs)
		totalBytes += size * len(ptrs)
	}

	return allocations, totalBytes
}
