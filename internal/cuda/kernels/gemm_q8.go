//go:build cuda

package kernels

/*
#cgo LDFLAGS: -L${SRCDIR} -lkernels -lcudart -lstdc++
#include "gemm_q8.h"
*/
import "C"

import (
	"fmt"
	"unsafe"
)

// GemmQ8F32 performs Q8_0 dequant-GEMM: C = dequant(A_q8) * B.
// A_q8 is packed Q8_0 blocks for matrix [M, K] (M * ceil(K/32) blocks of 36 bytes each).
// B is [K, N] row-major FP32. C is [M, N] row-major FP32.
// K must be a multiple of 32.
func GemmQ8F32(
	A_q8, B, C unsafe.Pointer,
	M, K, N int,
	stream unsafe.Pointer,
) error {
	err := C.gemm_q8_f32(
		A_q8, (*C.float)(B), (*C.float)(C),
		C.int(M), C.int(K), C.int(N),
		C.cudaStream_t(stream),
	)
	if err != C.cudaSuccess {
		return fmt.Errorf("gemm_q8_f32: %s",
			C.GoString(C.cudaGetErrorString(err)))
	}
	return nil
}
