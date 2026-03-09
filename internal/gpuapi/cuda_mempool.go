package gpuapi

import (
	"unsafe"

	"github.com/zerfoo/zerfoo/internal/cuda"
)

// CUDAMemPool implements the MemPool interface by wrapping cuda.MemPool.
type CUDAMemPool struct {
	inner *cuda.MemPool
}

// NewCUDAMemPool creates a new CUDA memory pool adapter.
func NewCUDAMemPool() *CUDAMemPool {
	return &CUDAMemPool{inner: cuda.NewMemPool()}
}

// NewCUDAMemPoolFrom wraps an existing cuda.MemPool.
func NewCUDAMemPoolFrom(pool *cuda.MemPool) *CUDAMemPool {
	return &CUDAMemPool{inner: pool}
}

func (p *CUDAMemPool) Alloc(deviceID, byteSize int) (unsafe.Pointer, error) {
	return p.inner.Alloc(deviceID, byteSize)
}

func (p *CUDAMemPool) Free(deviceID int, ptr unsafe.Pointer, byteSize int) {
	p.inner.Free(deviceID, ptr, byteSize)
}

func (p *CUDAMemPool) AllocManaged(deviceID, byteSize int) (unsafe.Pointer, error) {
	return p.inner.AllocManaged(deviceID, byteSize)
}

func (p *CUDAMemPool) FreeManaged(deviceID int, ptr unsafe.Pointer, byteSize int) {
	p.inner.FreeManaged(deviceID, ptr, byteSize)
}

func (p *CUDAMemPool) Drain() error {
	return p.inner.Drain()
}

func (p *CUDAMemPool) Stats() (int, int) {
	return p.inner.Stats()
}

// Inner returns the underlying cuda.MemPool for backward compatibility.
func (p *CUDAMemPool) Inner() *cuda.MemPool {
	return p.inner
}

// Compile-time interface assertion.
var _ MemPool = (*CUDAMemPool)(nil)
