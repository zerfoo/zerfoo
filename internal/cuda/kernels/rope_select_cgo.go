//go:build cuda

package kernels

/*
#cgo LDFLAGS: -L${SRCDIR} -lkernels -lcudart -lstdc++
#include "rope_select.h"
*/
import "C"

import (
	"fmt"
	"unsafe"
)

// RoPESelect copies halfRotary cos/sin values from the precomputed table
// at position counter[0]. Used for GPU-driven RoPE angle selection.
func RoPESelect(cosTable, sinTable, cosOut, sinOut, counter unsafe.Pointer,
	halfRotary int, s unsafe.Pointer) error {
	err := C.launch_rope_select(
		(*C.float)(cosTable), (*C.float)(sinTable),
		(*C.float)(cosOut), (*C.float)(sinOut),
		(*C.int)(counter), C.int(halfRotary),
		C.cudaStream_t(s),
	)
	if err != C.cudaSuccess {
		return fmt.Errorf("rope_select: %s",
			C.GoString(C.cudaGetErrorString(err)))
	}
	return nil
}
