//go:build !cuda

package kernels

import (
	"fmt"
	"unsafe"

	"github.com/zerfoo/zerfoo/internal/cuda"
)

// Gather launches the embedding table gather kernel with int64 indices.
// table: [V, D], indices: [N] int64, output: [N, D].
func Gather(table unsafe.Pointer, indices unsafe.Pointer,
	output unsafe.Pointer, N, D, V int, s unsafe.Pointer) error {
	k := klib()
	if k == nil {
		return fmt.Errorf("gather kernel: kernels not available")
	}
	ret := cuda.Ccall(k.launchGather,
		uintptr(table), uintptr(indices), uintptr(output),
		uintptr(N), uintptr(D), uintptr(V), uintptr(s))
	return checkKernel(ret, "gather")
}

// GatherI32 launches the embedding table gather kernel with int32 indices.
// table: [V, D], indices: [N] int32, output: [N, D].
func GatherI32(table unsafe.Pointer, indices unsafe.Pointer,
	output unsafe.Pointer, N, D, V int, s unsafe.Pointer) error {
	k := klib()
	if k == nil {
		return fmt.Errorf("gather_i32 kernel: kernels not available")
	}
	ret := cuda.Ccall(k.launchGatherI32,
		uintptr(table), uintptr(indices), uintptr(output),
		uintptr(N), uintptr(D), uintptr(V), uintptr(s))
	return checkKernel(ret, "gather_i32")
}
