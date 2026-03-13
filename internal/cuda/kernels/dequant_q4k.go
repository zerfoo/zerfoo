//go:build cuda

package kernels

/*
#cgo LDFLAGS: -L${SRCDIR} -lkernels -lcudart -lstdc++
#include "dequant_q4k.h"
*/
import "C"

import (
	"fmt"
	"unsafe"
)

// DequantQ4KF32 dequantizes Q4_K super-blocks to FP32 in global memory.
// src is raw Q4_K super-blocks for matrix [rows, K]. dst is [rows, K] FP32.
// K must be a multiple of 256.
func DequantQ4KF32(
	src, dst unsafe.Pointer,
	rows, K int,
	stream unsafe.Pointer,
) error {
	err := C.dequant_q4k_f32(
		src, (*C.float)(dst),
		C.int(rows), C.int(K),
		C.cudaStream_t(stream),
	)
	if err != C.cudaSuccess {
		return fmt.Errorf("dequant_q4k_f32: %s",
			C.GoString(C.cudaGetErrorString(err)))
	}
	return nil
}
