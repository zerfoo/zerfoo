//go:build !cuda

package kernels

import (
	"fmt"
	"unsafe"

	"github.com/zerfoo/zerfoo/internal/cuda"
)

// GemmQ4F32 performs Q4_0 dequant-GEMM: C = dequant(A_q4) * B.
// A_q4 is packed Q4_0 blocks, B is [K, N] FP32, C is [M, N] FP32.
func GemmQ4F32(
	A_q4, B, C unsafe.Pointer, //nolint:gocritic // match CGo API
	M, K, N int,               //nolint:gocritic // match CGo API
	stream unsafe.Pointer,
) error {
	k := klib()
	if k == nil {
		return fmt.Errorf("gemm_q4_f32 kernel: kernels not available")
	}
	ret := cuda.Ccall(k.launchGemmQ4F32,
		uintptr(A_q4), uintptr(B), uintptr(C),
		uintptr(M), uintptr(K), uintptr(N), uintptr(stream))
	return checkKernel(ret, "gemm_q4_f32")
}
