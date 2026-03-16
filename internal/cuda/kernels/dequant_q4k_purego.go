//go:build !cuda

package kernels

import (
	"fmt"
	"unsafe"

	"github.com/zerfoo/zerfoo/internal/cuda"
)

// DequantQ4KF32 dequantizes Q4_K super-blocks to FP32 in global memory.
// src is raw Q4_K super-blocks, dst is [rows, K] FP32.
func DequantQ4KF32(
	src, dst unsafe.Pointer,
	rows, K int,
	stream unsafe.Pointer,
) error {
	k := klib()
	if k == nil {
		return fmt.Errorf("dequant_q4k_f32 kernel: kernels not available")
	}
	ret := cuda.Ccall(k.launchDequantQ4KF32,
		uintptr(src), uintptr(dst),
		uintptr(rows), uintptr(K), uintptr(stream))
	return checkKernel(ret, "dequant_q4k_f32")
}
