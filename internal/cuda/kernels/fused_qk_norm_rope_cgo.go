//go:build cuda

package kernels

/*
#cgo LDFLAGS: -L${SRCDIR} -lkernels -lcudart -lstdc++
#include "fused_qk_norm_rope.h"
*/
import "C"

import (
	"fmt"
	"math"
	"unsafe"
)

// FusedQKNormRoPEF32 applies per-head RMSNorm + RoPE to combined Q+K heads.
// input: [totalHeads, headDim], weightQ/weightK: [headDim],
// cosAngles/sinAngles: [halfRotary], output: [totalHeads, headDim].
func FusedQKNormRoPEF32(input, weightQ, weightK, cosAngles, sinAngles, output unsafe.Pointer, eps float32, totalHeads, headDim, numQHeads, halfRotary int, s unsafe.Pointer) error {
	err := C.fused_qk_norm_rope_f32(
		(*C.float)(input), (*C.float)(weightQ), (*C.float)(weightK),
		(*C.float)(cosAngles), (*C.float)(sinAngles), (*C.float)(output),
		C.uint(math.Float32bits(eps)),
		C.int(totalHeads), C.int(headDim), C.int(numQHeads), C.int(halfRotary),
		C.cudaStream_t(s),
	)
	if err != C.cudaSuccess {
		return fmt.Errorf("fused_qk_norm_rope_f32: %s",
			C.GoString(C.cudaGetErrorString(err)))
	}
	return nil
}
