//go:build !cuda

package kernels

import (
	"fmt"
	"unsafe"

	"github.com/zerfoo/zerfoo/internal/cuda"
)

// Gather launches the embedding table gather kernel.
// table: [V, D], indices: [N], output: [N, D].
func Gather(table unsafe.Pointer, indices unsafe.Pointer,
	output unsafe.Pointer, N, D, V int, s unsafe.Pointer) error { //nolint:gocritic // match CGo API
	k := klib()
	if k == nil {
		return fmt.Errorf("gather kernel: kernels not available")
	}
	ret := cuda.Ccall(k.launchGather,
		uintptr(table), uintptr(indices), uintptr(output),
		uintptr(N), uintptr(D), uintptr(V), uintptr(s))
	return checkKernel(ret, "gather")
}
