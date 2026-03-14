//go:build !cuda

package kernels

import (
	"fmt"
	"unsafe"

	"github.com/zerfoo/zerfoo/internal/cuda"
)

// OffsetMemcpy copies dim floats from src to dst at offset counter*dim.
// counter is a GPU-resident int32. Used for GPU-driven KV cache append.
func OffsetMemcpy(dst, src, counter unsafe.Pointer, dim, maxSeqLen int, s unsafe.Pointer) error {
	k := klib()
	if k == nil {
		return fmt.Errorf("offset_memcpy kernel: kernels not available")
	}
	ret := cuda.Ccall(k.launchOffsetMemcpy,
		uintptr(dst), uintptr(src), uintptr(counter),
		uintptr(dim), uintptr(maxSeqLen), uintptr(s))
	return checkKernel(ret, "offset_memcpy")
}
