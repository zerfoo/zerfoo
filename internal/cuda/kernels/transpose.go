//go:build cuda

package kernels

/*
#cgo LDFLAGS: -L${SRCDIR} -lkernels -lcudart -lstdc++
#include <cuda_runtime.h>

extern cudaError_t launch_transpose_2d(const float* input, float* output,
                                        int rows, int cols, cudaStream_t stream);
extern cudaError_t launch_transpose_nd(const float* input, float* output,
                                        const int* in_strides, const int* out_strides,
                                        const int* perm, int ndim, int total,
                                        cudaStream_t stream);
*/
import "C"

import "unsafe"

// Transpose2D launches the tiled 2D transpose kernel.
// Input: [rows, cols] -> Output: [cols, rows].
func Transpose2D(input, output unsafe.Pointer, rows, cols int, s unsafe.Pointer) error {
	return checkCUDA(C.launch_transpose_2d(
		(*C.float)(input), (*C.float)(output),
		C.int(rows), C.int(cols), stream(s),
	), "transpose_2d")
}

// TransposeND launches the general N-D transpose kernel.
// inStrides: strides of the input tensor (on host).
// outStrides: precomputed output strides (on host).
// perm: permutation array (on host).
// ndim: number of dimensions.
// total: total number of elements.
func TransposeND(input, output unsafe.Pointer,
	inStrides, outStrides, perm []int32,
	ndim, total int, s unsafe.Pointer) error {
	return checkCUDA(C.launch_transpose_nd(
		(*C.float)(input), (*C.float)(output),
		(*C.int)(unsafe.Pointer(&inStrides[0])),
		(*C.int)(unsafe.Pointer(&outStrides[0])),
		(*C.int)(unsafe.Pointer(&perm[0])),
		C.int(ndim), C.int(total), stream(s),
	), "transpose_nd")
}
