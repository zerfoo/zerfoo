//go:build cuda

package kernels

/*
#cgo LDFLAGS: -L${SRCDIR} -lkernels -lcudart -lstdc++
#include "fused_rope.h"
*/
import "C"

import (
	"fmt"
	"unsafe"
)

// FusedRoPEF32 applies fused rotary positional embedding (RoPE) to FP32 data.
func FusedRoPEF32(
	input, cosAngles, sinAngles, output unsafe.Pointer,
	batch, seqLen, headDim, halfRotary, cosStride int,
	stream unsafe.Pointer,
) error {
	err := C.fused_rope_f32(
		(*C.float)(input), (*C.float)(cosAngles), (*C.float)(sinAngles),
		(*C.float)(output),
		C.int(batch), C.int(seqLen), C.int(headDim), C.int(halfRotary), C.int(cosStride),
		C.cudaStream_t(stream),
	)
	if err != C.cudaSuccess {
		return fmt.Errorf("fused_rope_f32: %s",
			C.GoString(C.cudaGetErrorString(err)))
	}
	return nil
}
