package compute

import (
	"math/bits"
	"sync"
)

// TensorArena is a power-of-2 bucketed pool for float32 backing arrays.
// It reduces GC pressure during inference by reusing allocations.
// Thread-safe with per-bucket mutex protection.
type TensorArena struct {
	buckets [32]arenaPool // bucket[i] holds slices of capacity 2^i
}

type arenaPool struct {
	mu   sync.Mutex
	free [][]float32
}

// Get returns a float32 slice of at least n elements from the arena.
// The returned slice has length n but may have capacity rounded up to
// a power of 2.
func (a *TensorArena) Get(n int) []float32 {
	if n <= 0 {
		return nil
	}
	bucket := bucketIndex(n)
	pool := &a.buckets[bucket]
	pool.mu.Lock()
	if len(pool.free) > 0 {
		buf := pool.free[len(pool.free)-1]
		pool.free = pool.free[:len(pool.free)-1]
		pool.mu.Unlock()
		// Zero the returned range for safety.
		clear(buf[:n])
		return buf[:n]
	}
	pool.mu.Unlock()
	cap := 1 << bucket
	return make([]float32, n, cap)
}

// Put returns a buffer to the arena for reuse.
func (a *TensorArena) Put(buf []float32) {
	c := cap(buf)
	if c == 0 {
		return
	}
	bucket := bucketIndex(c)
	// Only accept buffers that match the bucket capacity exactly.
	if 1<<bucket != c {
		return
	}
	pool := &a.buckets[bucket]
	pool.mu.Lock()
	pool.free = append(pool.free, buf[:c])
	pool.mu.Unlock()
}

// Reset clears all pooled buffers, allowing GC to collect them.
func (a *TensorArena) Reset() {
	for i := range a.buckets {
		pool := &a.buckets[i]
		pool.mu.Lock()
		pool.free = pool.free[:0]
		pool.mu.Unlock()
	}
}

// bucketIndex returns the index for the smallest power-of-2 >= n.
func bucketIndex(n int) int {
	if n <= 1 {
		return 0
	}
	return bits.Len(uint(n - 1))
}
