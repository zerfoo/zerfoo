//go:build !cuda

package kernels

import (
	"fmt"
	"unsafe"

	"github.com/zerfoo/zerfoo/internal/cuda"
)

// RMSNorm launches the fused RMSNorm kernel.
// input: [rows, D], weight: [D], output: [rows, D], scales: [rows].
func RMSNorm(input, weight, output, scales unsafe.Pointer, eps float32, rows, D int, s unsafe.Pointer) error { //nolint:gocritic // match CGo API
	k := klib()
	if k == nil {
		return fmt.Errorf("rmsnorm kernel: kernels not available")
	}
	ret := cuda.Ccall(k.launchRMSNorm,
		uintptr(input), uintptr(weight), uintptr(output), uintptr(scales),
		floatBits(eps), uintptr(rows), uintptr(D), uintptr(s))
	return checkKernel(ret, "rmsnorm")
}
