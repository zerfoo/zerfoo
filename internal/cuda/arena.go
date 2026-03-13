package cuda

import (
	"fmt"
	"os"
	"sync"
	"sync/atomic"
	"unsafe"
)

var defaultArenaInst *ArenaPool

// DefaultArenaPool returns the process-wide ArenaPool singleton, or nil.
func DefaultArenaPool() *ArenaPool {
	return defaultArenaInst
}

// SetDefaultArenaPool registers an ArenaPool as the process-wide default.
func SetDefaultArenaPool(a *ArenaPool) {
	defaultArenaInst = a
}

// ArenaPool is a bump-pointer allocator backed by a single large CUDA
// allocation. Each Alloc advances the offset within the pre-allocated region.
// Free is a no-op for individual pointers. Call Reset() between forward passes
// to reclaim all arena memory at once (zero-cost compared to per-pointer free).
//
// This eliminates cudaMalloc/cudaFree overhead during inference, which is the
// #1 bottleneck for per-token latency on the DGX Spark GPU.
//
// On devices with concurrent managed memory support (e.g., GB10 with
// NVLink-C2C and shared LPDDR5x), the arena is allocated with
// cudaMallocManaged. This makes the arena accessible from both CPU and GPU
// without explicit H2D copies.
//
// Weight tensors and KV cache should NOT use the arena (they persist across
// passes). The arena is only for per-pass intermediates.
type ArenaPool struct {
	mu         sync.Mutex
	base       unsafe.Pointer // start of the CUDA allocation
	capacity   int            // total bytes allocated
	offset     int            // current bump offset
	resetFloor int            // minimum offset after Reset (protects captured graph buffers)
	deviceID   int
	managed    bool // true if base was allocated with cudaMallocManaged
	hits       atomic.Int64
	misses     atomic.Int64 // only incremented if arena is full and falls back
	resets     atomic.Int64

	// fallback is the MemPool used when the arena is exhausted.
	fallback *MemPool
	// fallbackPtrs tracks pointers allocated from fallback so Free can route them.
	fallbackPtrs map[unsafe.Pointer]int // ptr -> byteSize
}

// NewArenaPool allocates a contiguous GPU region of the given capacity bytes
// on the specified device. A fallback MemPool handles any overflow.
// On devices with concurrent managed memory support, the arena uses
// cudaMallocManaged to enable zero-copy CPU/GPU access.
func NewArenaPool(deviceID, capacityBytes int, fallback *MemPool) (*ArenaPool, error) {
	if err := SetDevice(deviceID); err != nil {
		return nil, fmt.Errorf("ArenaPool: SetDevice: %w", err)
	}

	managed := ManagedMemorySupported(deviceID) && os.Getenv("ZERFOO_ENABLE_MANAGED_MEM") != ""

	var ptr unsafe.Pointer
	var err error
	if managed {
		ptr, err = MallocManaged(capacityBytes)
		if err != nil {
			// Fall back to regular malloc if managed allocation fails.
			managed = false
			ptr, err = Malloc(capacityBytes)
		}
	} else {
		ptr, err = Malloc(capacityBytes)
	}
	if err != nil {
		return nil, fmt.Errorf("ArenaPool: alloc(%d): %w", capacityBytes, err)
	}

	return &ArenaPool{
		base:         ptr,
		capacity:     capacityBytes,
		deviceID:     deviceID,
		managed:      managed,
		fallback:     fallback,
		fallbackPtrs: make(map[unsafe.Pointer]int),
	}, nil
}

// IsManaged returns true if the arena was allocated with managed memory.
func (a *ArenaPool) IsManaged() bool {
	return a.managed
}

// Alloc returns a device pointer of at least byteSize bytes from the arena.
// Allocations are 256-byte aligned for GPU coalescing. If the arena is full,
// falls back to the MemPool.
func (a *ArenaPool) Alloc(deviceID, byteSize int) (unsafe.Pointer, error) {
	// Align to 256 bytes for GPU memory access patterns.
	aligned := (byteSize + 255) &^ 255

	a.mu.Lock()
	if a.offset+aligned <= a.capacity {
		ptr := unsafe.Add(a.base, a.offset)
		a.offset += aligned
		a.mu.Unlock()
		a.hits.Add(1)
		return ptr, nil
	}
	a.mu.Unlock()

	// Arena exhausted -- fall back to MemPool.
	a.misses.Add(1)
	ptr, err := a.fallback.Alloc(deviceID, byteSize)
	if err != nil {
		return nil, err
	}
	a.mu.Lock()
	a.fallbackPtrs[ptr] = byteSize
	a.mu.Unlock()
	return ptr, nil
}

// Free is a no-op for arena pointers (reclaimed in bulk via Reset).
// Fallback pointers are returned to the MemPool.
func (a *ArenaPool) Free(deviceID int, ptr unsafe.Pointer, byteSize int) {
	a.mu.Lock()
	if _, ok := a.fallbackPtrs[ptr]; ok {
		delete(a.fallbackPtrs, ptr)
		a.mu.Unlock()
		a.fallback.Free(deviceID, ptr, byteSize)
		return
	}
	a.mu.Unlock()
	// Arena pointer: no-op, reclaimed by Reset.
}

// AllocManaged delegates to the fallback MemPool (arena is device-only).
func (a *ArenaPool) AllocManaged(deviceID, byteSize int) (unsafe.Pointer, error) {
	return a.fallback.AllocManaged(deviceID, byteSize)
}

// FreeManaged delegates to the fallback MemPool.
func (a *ArenaPool) FreeManaged(deviceID int, ptr unsafe.Pointer, byteSize int) {
	a.fallback.FreeManaged(deviceID, ptr, byteSize)
}

// Reset rewinds the arena offset to the reset floor (default 0), reclaiming
// per-pass allocations while preserving buffers below the floor (e.g. CUDA
// graph captured buffers).
func (a *ArenaPool) Reset() {
	a.mu.Lock()
	a.offset = a.resetFloor
	a.mu.Unlock()
	a.resets.Add(1)
}

// SetResetFloor sets the minimum offset that Reset will rewind to. Allocations
// below this offset are preserved across resets. This is used by CUDA graph
// capture to protect GPU buffers that the captured graph references.
func (a *ArenaPool) SetResetFloor(floor int) {
	a.mu.Lock()
	a.resetFloor = floor
	a.mu.Unlock()
}

// Drain frees the underlying CUDA allocation and drains the fallback pool.
func (a *ArenaPool) Drain() error {
	a.mu.Lock()
	defer a.mu.Unlock()

	var firstErr error
	if a.base != nil {
		if err := SetDevice(a.deviceID); err != nil && firstErr == nil {
			firstErr = err
		}
		if err := Free(a.base); err != nil && firstErr == nil {
			firstErr = err
		}
		a.base = nil
		a.offset = 0
		a.capacity = 0
	}

	if err := a.fallback.Drain(); err != nil && firstErr == nil {
		firstErr = err
	}
	return firstErr
}

// Stats returns the arena utilization and fallback pool stats.
func (a *ArenaPool) Stats() (allocations int, totalBytes int) {
	a.mu.Lock()
	arenaUsed := a.offset
	a.mu.Unlock()
	fbAllocs, fbBytes := a.fallback.Stats()
	return 1 + fbAllocs, arenaUsed + fbBytes
}

// HitMissStats returns arena hits, fallback misses, and reset count.
func (a *ArenaPool) HitMissStats() (hits, misses, resets int64) {
	return a.hits.Load(), a.misses.Load(), a.resets.Load()
}

// UsedBytes returns the current arena offset (bytes in use).
func (a *ArenaPool) UsedBytes() int {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.offset
}

// Capacity returns the total arena capacity in bytes.
func (a *ArenaPool) Capacity() int {
	return a.capacity
}
