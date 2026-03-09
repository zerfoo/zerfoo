//go:build cuda

package kernels

/*
#cgo LDFLAGS: -L${SRCDIR} -lkernels -lcudart -lstdc++
#include <cuda_runtime.h>

extern cudaError_t launch_rmsnorm(const float* input, const float* weight,
                                   float* output, float* scales, float eps,
                                   int rows, int D, cudaStream_t stream);
*/
import "C"

import "unsafe"

// RMSNorm launches the fused RMSNorm kernel.
// input: [rows, D], weight: [D], output: [rows, D], scales: [rows].
// Computes: output = input * rsqrt(mean(input^2) + eps) * weight.
// Writes per-row rsqrt values to scales for backward pass.
func RMSNorm(input, weight, output, scales unsafe.Pointer, eps float32, rows, D int, s unsafe.Pointer) error {
	return checkCUDA(C.launch_rmsnorm(
		(*C.float)(input), (*C.float)(weight), (*C.float)(output), (*C.float)(scales),
		C.float(eps), C.int(rows), C.int(D), stream(s),
	), "rmsnorm")
}
