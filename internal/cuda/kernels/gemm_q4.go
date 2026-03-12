//go:build cuda

package kernels

/*
#cgo LDFLAGS: -L${SRCDIR} -lkernels -lcudart -lstdc++
#include "gemm_q4.h"
*/
import "C"

import (
	"fmt"
	"unsafe"
)

// GemmQ4F32 performs Q4_0 dequant-GEMM: C = dequant(A_q4) * B.
// A_q4 is packed Q4_0 blocks for matrix [M, K] (M * ceil(K/32) blocks of 18 bytes each).
// B is [K, N] row-major FP32. C is [M, N] row-major FP32.
// K must be a multiple of 32.
func GemmQ4F32(
	A_q4, B, C unsafe.Pointer,
	M, K, N int,
	stream unsafe.Pointer,
) error {
	err := C.gemm_q4_f32(
		A_q4, (*C.float)(B), (*C.float)(C),
		C.int(M), C.int(K), C.int(N),
		C.cudaStream_t(stream),
	)
	if err != C.cudaSuccess {
		return fmt.Errorf("gemm_q4_f32: %s",
			C.GoString(C.cudaGetErrorString(err)))
	}
	return nil
}
