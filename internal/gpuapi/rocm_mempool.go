package gpuapi

import (
	"fmt"
	"unsafe"

	"github.com/zerfoo/zerfoo/internal/hip"
)

// ROCmMemPool implements the MemPool interface by wrapping hip.MemPool.
type ROCmMemPool struct {
	inner *hip.MemPool
}

// NewROCmMemPool creates a new ROCm memory pool adapter.
func NewROCmMemPool() *ROCmMemPool {
	return &ROCmMemPool{inner: hip.NewMemPool()}
}

// NewROCmMemPoolFrom wraps an existing hip.MemPool.
func NewROCmMemPoolFrom(pool *hip.MemPool) *ROCmMemPool {
	return &ROCmMemPool{inner: pool}
}

func (p *ROCmMemPool) Alloc(deviceID, byteSize int) (unsafe.Pointer, error) {
	return p.inner.Alloc(deviceID, byteSize)
}

func (p *ROCmMemPool) Free(deviceID int, ptr unsafe.Pointer, byteSize int) {
	p.inner.Free(deviceID, ptr, byteSize)
}

func (p *ROCmMemPool) AllocManaged(_, _ int) (unsafe.Pointer, error) {
	return nil, fmt.Errorf("AllocManaged: not supported on ROCm backend")
}

func (p *ROCmMemPool) FreeManaged(_ int, _ unsafe.Pointer, _ int) {}

func (p *ROCmMemPool) Drain() error {
	return p.inner.Drain()
}

func (p *ROCmMemPool) Stats() (int, int) {
	return p.inner.Stats()
}

// Inner returns the underlying hip.MemPool for backward compatibility.
func (p *ROCmMemPool) Inner() *hip.MemPool {
	return p.inner
}

// Compile-time interface assertion.
var _ MemPool = (*ROCmMemPool)(nil)
