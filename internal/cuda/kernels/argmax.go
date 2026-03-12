//go:build cuda

package kernels

/*
#cgo LDFLAGS: -L${SRCDIR} -lkernels -lcudart -lstdc++
#include <cuda_runtime.h>

extern cudaError_t launch_argmax(const float* input, int* result,
                                  void* scratch, int n,
                                  cudaStream_t stream);
*/
import "C"

import "unsafe"

// Argmax launches the GPU argmax kernel.
// input: [n] float32 on device, result: single int32 on device,
// scratch: device temp storage of at least 2*ceil(n/256)*4 bytes.
func Argmax(input unsafe.Pointer, result unsafe.Pointer,
	scratch unsafe.Pointer, n int, s unsafe.Pointer) error {
	return checkCUDA(C.launch_argmax(
		(*C.float)(input), (*C.int)(result),
		scratch, C.int(n), stream(s),
	), "argmax")
}
