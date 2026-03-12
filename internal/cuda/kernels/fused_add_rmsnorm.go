//go:build cuda

package kernels

/*
#cgo LDFLAGS: -L${SRCDIR} -lkernels -lcudart -lstdc++
#include "fused_add_rmsnorm.h"
*/
import "C"

import (
	"fmt"
	"math"
	"unsafe"
)

// FusedAddRMSNormF32 performs fused residual add + RMSNorm in one kernel launch.
// input: [rows, D] (read-only), residual: [rows, D] (read-only),
// weight: [D], normedOut: [rows, D], sumOut: [rows, D].
func FusedAddRMSNormF32(input, residual, weight, normedOut, sumOut unsafe.Pointer, eps float32, rows, D int, s unsafe.Pointer) error { //nolint:gocritic // match purego API
	err := C.fused_add_rmsnorm_f32(
		(*C.float)(input), (*C.float)(residual),
		(*C.float)(weight), (*C.float)(normedOut), (*C.float)(sumOut),
		C.uint(math.Float32bits(eps)),
		C.int(rows), C.int(D),
		C.cudaStream_t(s),
	)
	if err != C.cudaSuccess {
		return fmt.Errorf("fused_add_rmsnorm_f32: %s",
			C.GoString(C.cudaGetErrorString(err)))
	}
	return nil
}
