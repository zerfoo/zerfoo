//go:build !cuda

package kernels

import (
	"fmt"
	"unsafe"

	"github.com/zerfoo/zerfoo/internal/cuda"
)

// FusedSwiGLUF32 applies fused SwiGLU activation: output[i] = w1[i] * sigmoid(w1[i]) * w3[i].
func FusedSwiGLUF32(
	w1, w3, output unsafe.Pointer,
	n int,
	stream unsafe.Pointer,
) error {
	k := klib()
	if k == nil {
		return fmt.Errorf("fused_swiglu_f32 kernel: kernels not available")
	}
	ret := cuda.Ccall(k.launchFusedSwiGLUF32,
		uintptr(w1), uintptr(w3), uintptr(output),
		uintptr(n),
		uintptr(stream))
	return checkKernel(ret, "fused_swiglu_f32")
}
