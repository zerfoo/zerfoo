//go:build !cuda

package kernels

import (
	"fmt"
	"unsafe"

	"github.com/zerfoo/zerfoo/internal/cuda"
)

// Argmax launches the GPU argmax kernel.
// input: [n] float32 on device, result: single int32 on device,
// scratch: device temp storage of at least 2*ceil(n/256)*4 bytes.
func Argmax(input unsafe.Pointer, result unsafe.Pointer,
	scratch unsafe.Pointer, n int, s unsafe.Pointer) error {
	k := klib()
	if k == nil {
		return fmt.Errorf("argmax kernel: kernels not available")
	}
	ret := cuda.Ccall(k.launchArgmax,
		uintptr(input), uintptr(result), uintptr(scratch),
		uintptr(n), uintptr(s))
	return checkKernel(ret, "argmax")
}
