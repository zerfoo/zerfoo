//go:build cuda

package kernels

/*
#cgo LDFLAGS: -L${SRCDIR} -lkernels -lcudart -lstdc++
#include "fused_norm_add.h"
*/
import "C"

import (
	"fmt"
	"math"
	"unsafe"
)

// FusedNormAddF32 applies RMSNorm then adds residual in one kernel launch.
// output = rmsnorm(input, weight, eps) + residual.
// input: [rows, D], weight: [D], residual: [rows, D], output: [rows, D].
func FusedNormAddF32(input, weight, residual, output unsafe.Pointer, eps float32, rows, D int, s unsafe.Pointer) error {
	err := C.fused_norm_add_f32(
		(*C.float)(input), (*C.float)(weight), (*C.float)(residual),
		(*C.float)(output),
		C.uint(math.Float32bits(eps)),
		C.int(rows), C.int(D),
		C.cudaStream_t(s),
	)
	if err != C.cudaSuccess {
		return fmt.Errorf("fused_norm_add_f32: %s",
			C.GoString(C.cudaGetErrorString(err)))
	}
	return nil
}
