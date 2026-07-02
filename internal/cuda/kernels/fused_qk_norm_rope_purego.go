//go:build !cuda

package kernels

import (
	"fmt"
	"unsafe"

	"github.com/zerfoo/zerfoo/internal/cuda"
)

// FusedQKNormRoPEF32 applies per-head RMSNorm + RoPE to combined Q+K heads.
// input: [totalHeads, headDim], weightQ/weightK: [headDim],
// cosAngles/sinAngles: [halfRotary], output: [totalHeads, headDim].
func FusedQKNormRoPEF32(input, weightQ, weightK, cosAngles, sinAngles, output unsafe.Pointer, eps float32, totalHeads, headDim, numQHeads, halfRotary int, s unsafe.Pointer) error {
	k := klib()
	if k == nil {
		return fmt.Errorf("fused_qk_norm_rope_f32 kernel: kernels not available")
	}
	ret := cuda.Ccall(k.launchFusedQKNormRoPEF32,
		uintptr(input), uintptr(weightQ), uintptr(weightK),
		uintptr(cosAngles), uintptr(sinAngles), uintptr(output),
		floatBits(eps), uintptr(totalHeads), uintptr(headDim),
		uintptr(numQHeads), uintptr(halfRotary), uintptr(s))
	return checkKernel(ret, "fused_qk_norm_rope_f32")
}
