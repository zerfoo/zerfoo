//go:build !cuda

package kernels

import (
	"fmt"
	"unsafe"

	"github.com/zerfoo/zerfoo/internal/cuda"
)

// SgemvM1 computes y = A*x for M=1 decode (single-token GEMV).
// y[M], A[M x N] row-major, x[N]. All FP32.
func SgemvM1(y, A, x unsafe.Pointer, M, N int, s unsafe.Pointer) error {
	k := klib()
	if k == nil {
		return fmt.Errorf("sgemv_m1 kernel: kernels not available")
	}
	ret := cuda.Ccall(k.launchSgemvM1,
		uintptr(y), uintptr(A), uintptr(x),
		uintptr(M), uintptr(N), uintptr(s))
	return checkKernel(ret, "sgemv_m1")
}
