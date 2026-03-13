//go:build cuda

package kernels

/*
#cgo LDFLAGS: -L${SRCDIR} -lkernels -lcudart -lstdc++
#include <cuda_runtime.h>

extern cudaError_t launch_gather(const float* table, const long long* indices,
                                  float* output, int N, int D, int V,
                                  cudaStream_t stream);

extern cudaError_t launch_gather_i32(const float* table, const int* indices,
                                      float* output, int N, int D, int V,
                                      cudaStream_t stream);
*/
import "C"

import "unsafe"

// Gather launches the embedding table gather kernel with int64 indices.
// table: [V, D] embedding table on device.
// indices: [N] int64 indices on device.
// output: [N, D] output on device.
func Gather(table unsafe.Pointer, indices unsafe.Pointer,
	output unsafe.Pointer, N, D, V int, s unsafe.Pointer) error { //nolint:gocritic // match CGo API
	return checkCUDA(C.launch_gather(
		(*C.float)(table), (*C.longlong)(indices),
		(*C.float)(output),
		C.int(N), C.int(D), C.int(V), stream(s),
	), "gather")
}

// GatherI32 launches the embedding table gather kernel with int32 indices.
// table: [V, D] embedding table on device.
// indices: [N] int32 indices on device.
// output: [N, D] output on device.
func GatherI32(table unsafe.Pointer, indices unsafe.Pointer,
	output unsafe.Pointer, N, D, V int, s unsafe.Pointer) error { //nolint:gocritic // match CGo API
	return checkCUDA(C.launch_gather_i32(
		(*C.float)(table), (*C.int)(indices),
		(*C.float)(output),
		C.int(N), C.int(D), C.int(V), stream(s),
	), "gather_i32")
}
