//go:build !cuda

package kernels

import (
	"fmt"
	"unsafe"

	"github.com/zerfoo/zerfoo/internal/cuda"
)

// GemvQ4KF32 performs Q4_K fused dequant-GEMV: y = dequant(W_q4k) * x.
// W_q4k is raw Q4_K super-blocks, x is [K] FP32, y is [M] FP32.
func GemvQ4KF32(
	W_q4k, x, y unsafe.Pointer,
	M, K int,
	stream unsafe.Pointer,
) error {
	k := klib()
	if k == nil {
		return fmt.Errorf("gemv_q4k_f32 kernel: kernels not available")
	}
	ret := cuda.Ccall(k.launchGemvQ4KF32,
		uintptr(W_q4k), uintptr(x), uintptr(y),
		uintptr(M), uintptr(K), uintptr(stream))
	return checkKernel(ret, "gemv_q4k_f32")
}
