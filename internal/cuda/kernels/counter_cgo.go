//go:build cuda

package kernels

/*
#cgo LDFLAGS: -L${SRCDIR} -lkernels -lcudart -lstdc++
#include <cuda_runtime.h>

extern cudaError_t launch_increment_counter(int* counter, int delta, cudaStream_t stream);
extern cudaError_t launch_reset_counter(int* counter, int value, cudaStream_t stream);
*/
import "C"
import "unsafe"

// IncrementCounter atomically increments a GPU-resident int32 by delta.
func IncrementCounter(counter unsafe.Pointer, delta int, s unsafe.Pointer) error {
	return checkCUDA(C.launch_increment_counter((*C.int)(counter), C.int(delta), stream(s)), "increment_counter")
}

// ResetCounter sets a GPU-resident int32 to value.
func ResetCounter(counter unsafe.Pointer, value int, s unsafe.Pointer) error {
	return checkCUDA(C.launch_reset_counter((*C.int)(counter), C.int(value), stream(s)), "reset_counter")
}
