//go:build cuda

package kernels

/*
#cgo LDFLAGS: -L${SRCDIR} -lkernels -lcudart -lstdc++
#include <cuda_runtime.h>

extern cudaError_t launch_sgemv_m1(float* y, const float* A, const float* x,
                                    int M, int N, cudaStream_t stream);
*/
import "C"

import "unsafe"

// SgemvM1 computes y = A*x for M=1 decode (single-token GEMV).
// y[M], A[M x N] row-major, x[N]. All FP32.
func SgemvM1(y, A, x unsafe.Pointer, M, N int, s unsafe.Pointer) error {
	return checkCUDA(C.launch_sgemv_m1(
		(*C.float)(y), (*C.float)(A), (*C.float)(x),
		C.int(M), C.int(N), stream(s),
	), "sgemv_m1")
}
