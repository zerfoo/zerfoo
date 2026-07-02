//go:build cuda

package kernels

/*
#cgo LDFLAGS: -L${SRCDIR} -lkernels -lcudart -lstdc++
#include "fused_swiglu.h"
*/
import "C"

import (
	"fmt"
	"unsafe"
)

// FusedSwiGLUF32 applies fused SwiGLU activation: output[i] = w1[i] * sigmoid(w1[i]) * w3[i].
func FusedSwiGLUF32(
	w1, w3, output unsafe.Pointer,
	n int,
	stream unsafe.Pointer,
) error {
	err := C.fused_swiglu_f32(
		(*C.float)(w1), (*C.float)(w3),
		(*C.float)(output),
		C.int(n),
		C.cudaStream_t(stream),
	)
	if err != C.cudaSuccess {
		return fmt.Errorf("fused_swiglu_f32: %s",
			C.GoString(C.cudaGetErrorString(err)))
	}
	return nil
}
