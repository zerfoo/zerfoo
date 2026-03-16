//go:build !cuda

package kernels

import (
	"fmt"
	"unsafe"

	"github.com/zerfoo/zerfoo/internal/cuda"
)

// GemmQ8F32 performs Q8_0 dequant-GEMM: C = dequant(A_q8) * B.
// A_q8 is packed Q8_0 blocks, B is [K, N] FP32, C is [M, N] FP32.
func GemmQ8F32(
	A_q8, B, C unsafe.Pointer,
	M, K, N int,              
	stream unsafe.Pointer,
) error {
	k := klib()
	if k == nil {
		return fmt.Errorf("gemm_q8_f32 kernel: kernels not available")
	}
	ret := cuda.Ccall(k.launchGemmQ8F32,
		uintptr(A_q8), uintptr(B), uintptr(C),
		uintptr(M), uintptr(K), uintptr(N), uintptr(stream))
	return checkKernel(ret, "gemm_q8_f32")
}
