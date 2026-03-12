//go:build !cuda

package kernels

import (
	"fmt"
	"unsafe"

	"github.com/zerfoo/zerfoo/internal/cuda"
)

// ScaledSoftmaxF32 applies fused scaled softmax: output = softmax(input * scale).
func ScaledSoftmaxF32(
	input, output unsafe.Pointer,
	outer, inner, axisSize int,
	scale float32,
	stream unsafe.Pointer,
) error {
	k := klib()
	if k == nil {
		return fmt.Errorf("scaled_softmax_f32 kernel: kernels not available")
	}
	ret := cuda.Ccall(k.launchScaledSoftmaxF32,
		uintptr(input), uintptr(output),
		uintptr(outer), uintptr(inner), uintptr(axisSize),
		floatBits(scale),
		uintptr(stream))
	return checkKernel(ret, "scaled_softmax_f32")
}
