//go:build !cuda

package kernels

import (
	"fmt"
	"unsafe"

	"github.com/zerfoo/zerfoo/internal/cuda"
)

// GemmQ4F32 performs Q4_0 dequant-GEMM: C = dequant(A_q4) * B.
// A_q4 is in GPU separated layout (scales then data), B is [K, N] FP32, C is [M, N] FP32.
// dataOffset is the byte offset from A_q4 to the packed data region.
func GemmQ4F32(
	A_q4, B, C unsafe.Pointer,
	M, K, N, dataOffset int,
	stream unsafe.Pointer,
) error {
	k := klib()
	if k == nil {
		return fmt.Errorf("gemm_q4_f32 kernel: kernels not available")
	}
	ret := cuda.Ccall(k.launchGemmQ4F32,
		uintptr(A_q4), uintptr(B), uintptr(C),
		uintptr(M), uintptr(K), uintptr(N), uintptr(dataOffset), uintptr(stream))
	return checkKernel(ret, "gemm_q4_f32")
}
