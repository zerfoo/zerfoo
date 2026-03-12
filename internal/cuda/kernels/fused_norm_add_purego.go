//go:build !cuda

package kernels

import (
	"fmt"
	"unsafe"

	"github.com/zerfoo/zerfoo/internal/cuda"
)

// FusedNormAddF32 applies RMSNorm then adds residual in one kernel launch.
// output = rmsnorm(input, weight, eps) + residual.
// input: [rows, D], weight: [D], residual: [rows, D], output: [rows, D].
func FusedNormAddF32(input, weight, residual, output unsafe.Pointer, eps float32, rows, D int, s unsafe.Pointer) error {
	k := klib()
	if k == nil {
		return fmt.Errorf("fused_norm_add_f32 kernel: kernels not available")
	}
	ret := cuda.Ccall(k.launchFusedNormAddF32,
		uintptr(input), uintptr(weight), uintptr(residual), uintptr(output),
		floatBits(eps), uintptr(rows), uintptr(D), uintptr(s))
	return checkKernel(ret, "fused_norm_add_f32")
}
