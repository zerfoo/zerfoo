//go:build !cuda

package kernels

import (
	"fmt"
	"unsafe"

	"github.com/zerfoo/zerfoo/internal/cuda"
)

// RoPESelect copies halfRotary cos/sin values from the precomputed table
// at position counter[0]. Used for GPU-driven RoPE angle selection.
func RoPESelect(cosTable, sinTable, cosOut, sinOut, counter unsafe.Pointer,
	halfRotary int, s unsafe.Pointer) error {
	k := klib()
	if k == nil {
		return fmt.Errorf("rope_select kernel: kernels not available")
	}
	ret := cuda.Ccall(k.launchRoPESelect,
		uintptr(cosTable), uintptr(sinTable),
		uintptr(cosOut), uintptr(sinOut),
		uintptr(counter), uintptr(halfRotary), uintptr(s))
	return checkKernel(ret, "rope_select")
}
