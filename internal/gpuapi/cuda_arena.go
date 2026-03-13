package gpuapi

import (
	"fmt"
	"os"
	"runtime"
	"sync/atomic"
	"unsafe"

	"github.com/zerfoo/zerfoo/internal/cuda"
)

// arenaProfilingEnabled controls diagnostic allocation logging.
// Set ZERFOO_ARENA_PROFILE=1 to enable.
var arenaProfilingEnabled = os.Getenv("ZERFOO_ARENA_PROFILE") != ""

// arenaRunningTotal tracks cumulative bytes allocated from the arena.
var arenaRunningTotal atomic.Int64

// CUDAArenaPool adapts cuda.ArenaPool to the gpuapi.MemPool interface.
// It also exposes Reset() for use between forward passes.
type CUDAArenaPool struct {
	inner *cuda.ArenaPool
}

// NewCUDAArenaPool creates a new arena-backed pool on the given device.
// capacityBytes is the size of the pre-allocated arena region.
// fallback is the MemPool used when the arena is exhausted.
func NewCUDAArenaPool(deviceID, capacityBytes int, fallback *cuda.MemPool) (*CUDAArenaPool, error) {
	arena, err := cuda.NewArenaPool(deviceID, capacityBytes, fallback)
	if err != nil {
		return nil, err
	}
	return &CUDAArenaPool{inner: arena}, nil
}

func (p *CUDAArenaPool) Alloc(deviceID, byteSize int) (unsafe.Pointer, error) {
	if arenaProfilingEnabled {
		// Capture arena state before allocation.
		usedBefore := p.inner.UsedBytes()
		ptr, err := p.inner.Alloc(deviceID, byteSize)
		usedAfter := p.inner.UsedBytes()
		miss := usedAfter == usedBefore // offset didn't move = arena miss
		total := arenaRunningTotal.Add(int64(byteSize))

		caller := "unknown"
		var pcs [4]uintptr
		if n := runtime.Callers(2, pcs[:]); n > 0 {
			frames := runtime.CallersFrames(pcs[:n])
			if f, more := frames.Next(); more || f.Function != "" {
				caller = fmt.Sprintf("%s:%d", f.Function, f.Line)
			}
		}

		fmt.Fprintf(os.Stderr, "[ARENA] alloc=%d used=%d/%d total=%d miss=%v caller=%s\n",
			byteSize, usedAfter, p.inner.Capacity(), total, miss, caller)
		return ptr, err
	}
	return p.inner.Alloc(deviceID, byteSize)
}

func (p *CUDAArenaPool) Free(deviceID int, ptr unsafe.Pointer, byteSize int) {
	p.inner.Free(deviceID, ptr, byteSize)
}

func (p *CUDAArenaPool) AllocManaged(deviceID, byteSize int) (unsafe.Pointer, error) {
	return p.inner.AllocManaged(deviceID, byteSize)
}

func (p *CUDAArenaPool) FreeManaged(deviceID int, ptr unsafe.Pointer, byteSize int) {
	p.inner.FreeManaged(deviceID, ptr, byteSize)
}

func (p *CUDAArenaPool) Drain() error {
	return p.inner.Drain()
}

func (p *CUDAArenaPool) Stats() (int, int) {
	return p.inner.Stats()
}

// Reset rewinds the arena, reclaiming all per-pass allocations.
func (p *CUDAArenaPool) Reset() {
	if arenaProfilingEnabled {
		hits, misses, resets := p.inner.HitMissStats()
		used := p.inner.UsedBytes()
		fmt.Fprintf(os.Stderr, "[ARENA] RESET used=%d/%d hits=%d misses=%d resets=%d\n",
			used, p.inner.Capacity(), hits, misses, resets)
	}
	p.inner.Reset()
}

// Inner returns the underlying cuda.ArenaPool.
func (p *CUDAArenaPool) Inner() *cuda.ArenaPool {
	return p.inner
}

// SetResetFloor sets the minimum offset that Reset will rewind to.
func (p *CUDAArenaPool) SetResetFloor(floor int) {
	p.inner.SetResetFloor(floor)
}

// UsedBytes returns the current arena offset (bytes in use).
func (p *CUDAArenaPool) UsedBytes() int {
	return p.inner.UsedBytes()
}

// Compile-time interface assertion.
var _ MemPool = (*CUDAArenaPool)(nil)
