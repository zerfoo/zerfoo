//go:build cuda

package kernels

/*
#cgo LDFLAGS: -L${SRCDIR} -lkernels -lcudart -lstdc++
#include "scaled_softmax.h"
*/
import "C"

import (
	"fmt"
	"math"
	"unsafe"
)

// ScaledSoftmaxF32 applies fused scaled softmax: output = softmax(input * scale).
func ScaledSoftmaxF32(
	input, output unsafe.Pointer,
	outer, inner, axisSize int,
	scale float32,
	stream unsafe.Pointer,
) error {
	err := C.scaled_softmax_f32(
		(*C.float)(input), (*C.float)(output),
		C.int(outer), C.int(inner), C.int(axisSize),
		C.uint(math.Float32bits(scale)),
		C.cudaStream_t(stream),
	)
	if err != C.cudaSuccess {
		return fmt.Errorf("scaled_softmax_f32: %s",
			C.GoString(C.cudaGetErrorString(err)))
	}
	return nil
}
