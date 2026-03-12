//go:build !cuda

package kernels

import (
	"fmt"
	"unsafe"

	"github.com/zerfoo/zerfoo/internal/cuda"
)

// FusedAddRMSNormF32 performs fused residual add + RMSNorm in one kernel launch.
// input: [rows, D], residual: [rows, D] (updated in-place), weight: [D], output: [rows, D].
func FusedAddRMSNormF32(input, residual, weight, output unsafe.Pointer, eps float32, rows, D int, s unsafe.Pointer) error { //nolint:gocritic // match CGo API
	k := klib()
	if k == nil {
		return fmt.Errorf("fused_add_rmsnorm_f32 kernel: kernels not available")
	}
	ret := cuda.Ccall(k.launchFusedAddRMSNormF32,
		uintptr(input), uintptr(residual), uintptr(weight), uintptr(output),
		floatBits(eps), uintptr(rows), uintptr(D), uintptr(s))
	return checkKernel(ret, "fused_add_rmsnorm_f32")
}
