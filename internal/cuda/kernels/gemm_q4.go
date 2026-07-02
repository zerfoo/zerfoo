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
// A_q4 is in GPU separated layout (scales then data).
// B is [K, N] row-major FP32. C is [M, N] row-major FP32.
// dataOffset is the byte offset from A_q4 to the packed data region.
// K must be a multiple of 32.
func GemmQ4F32(
	A_q4, B, COut unsafe.Pointer,
	M, K, N, dataOffset int,
	stream unsafe.Pointer,
) error {
	err := C.gemm_q4_f32(
		A_q4, (*C.float)(B), (*C.float)(COut),
		C.int(M), C.int(K), C.int(N),
		C.int(dataOffset),
		C.cudaStream_t(stream),
	)
	if err != C.cudaSuccess {
		return fmt.Errorf("gemm_q4_f32: %s",
			C.GoString(C.cudaGetErrorString(err)))
	}
	return nil
}
