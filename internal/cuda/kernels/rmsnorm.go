//go:build cuda

package kernels

/*
#cgo LDFLAGS: -L${SRCDIR} -lkernels -lcudart -lstdc++
#include <cuda_runtime.h>
#include <math.h>

extern cudaError_t launch_rmsnorm(const float* input, const float* weight,
                                   float* output, float* scales,
                                   unsigned int eps_bits,
                                   int rows, int D, cudaStream_t stream);
*/
import "C"

import (
	"math"
	"unsafe"
)

// RMSNorm launches the fused RMSNorm kernel.
// input: [rows, D], weight: [D], output: [rows, D].
// scales: [rows] per-row RMS values (optional, may be nil).
// Computes: output = input * rsqrt(mean(input^2) + eps) * weight.
func RMSNorm(input, weight, output, scales unsafe.Pointer, eps float32, rows, D int, s unsafe.Pointer) error { //nolint:gocritic // match purego API
	return checkCUDA(C.launch_rmsnorm(
		(*C.float)(input), (*C.float)(weight), (*C.float)(output),
		(*C.float)(scales), C.uint(math.Float32bits(eps)),
		C.int(rows), C.int(D), stream(s),
	), "rmsnorm")
}
