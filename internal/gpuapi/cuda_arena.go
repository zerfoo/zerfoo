package gpuapi

import (
	"unsafe"

	"github.com/zerfoo/zerfoo/internal/cuda"
)

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
	p.inner.Reset()
}

// Inner returns the underlying cuda.ArenaPool.
func (p *CUDAArenaPool) Inner() *cuda.ArenaPool {
	return p.inner
}

// Compile-time interface assertion.
var _ MemPool = (*CUDAArenaPool)(nil)
