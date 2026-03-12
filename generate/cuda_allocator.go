package generate

import (
	"unsafe"

	"github.com/zerfoo/zerfoo/internal/cuda"
)

// cudaAllocator implements GPUAllocator using the cuda package.
type cudaAllocator struct{}

func (cudaAllocator) Alloc(size int) (unsafe.Pointer, error) {
	return cuda.Malloc(size)
}

func (cudaAllocator) Free(ptr unsafe.Pointer) error {
	return cuda.Free(ptr)
}

func (cudaAllocator) Memcpy(dst, src unsafe.Pointer, size int, kind int) error {
	// GPUAllocator convention: 0 = HostToDevice, 1 = DeviceToHost.
	// cuda package convention: 1 = HostToDevice, 2 = DeviceToHost.
	return cuda.Memcpy(dst, src, size, cuda.MemcpyKind(kind+1))
}
