//go:build !cuda

package kernels

import (
	"fmt"
	"unsafe"

	"github.com/zerfoo/zerfoo/internal/cuda"
)

// FusedAddRMSNormF32 performs fused residual add + RMSNorm in one kernel launch.
// input: [rows, D] (read-only), residual: [rows, D] (read-only),
// weight: [D], normedOut: [rows, D], sumOut: [rows, D].
func FusedAddRMSNormF32(input, residual, weight, normedOut, sumOut unsafe.Pointer, eps float32, rows, D int, s unsafe.Pointer) error {
	k := klib()
	if k == nil {
		return fmt.Errorf("fused_add_rmsnorm_f32 kernel: kernels not available")
	}
	ret := cuda.Ccall(k.launchFusedAddRMSNormF32,
		uintptr(input), uintptr(residual), uintptr(weight), uintptr(normedOut),
		uintptr(sumOut), floatBits(eps), uintptr(rows), uintptr(D), uintptr(s))
	return checkKernel(ret, "fused_add_rmsnorm_f32")
}
