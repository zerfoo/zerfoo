//go:build !cuda

package kernels

import (
	"fmt"
	"unsafe"

	"github.com/zerfoo/zerfoo/internal/cuda"
)

// FusedRoPEF32 applies fused rotary positional embedding (RoPE) to FP32 data.
func FusedRoPEF32(
	input, cosAngles, sinAngles, output unsafe.Pointer,
	batch, seqLen, headDim, halfRotary, cosStride int,
	stream unsafe.Pointer,
) error {
	k := klib()
	if k == nil {
		return fmt.Errorf("fused_rope_f32 kernel: kernels not available")
	}
	ret := cuda.Ccall(k.launchFusedRoPEF32,
		uintptr(input), uintptr(cosAngles), uintptr(sinAngles), uintptr(output),
		uintptr(batch), uintptr(seqLen), uintptr(headDim), uintptr(halfRotary), uintptr(cosStride),
		uintptr(stream))
	return checkKernel(ret, "fused_rope_f32")
}
