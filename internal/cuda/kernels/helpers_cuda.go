//go:build cuda

package kernels

/*
#include <cuda_runtime.h>
*/
import "C"

import (
	"fmt"
	"unsafe"
)

// checkCUDA converts a CUDA error code into a Go error.
func checkCUDA(err C.cudaError_t, op string) error {
	if err != C.cudaSuccess {
		return fmt.Errorf("%s: %s", op, C.GoString(C.cudaGetErrorString(err)))
	}
	return nil
}

// stream converts an unsafe.Pointer to a cudaStream_t.
func stream(s unsafe.Pointer) C.cudaStream_t {
	return C.cudaStream_t(s)
}
