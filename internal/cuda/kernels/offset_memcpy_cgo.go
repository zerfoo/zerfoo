//go:build cuda

package kernels

/*
#cgo LDFLAGS: -L${SRCDIR} -lkernels -lcudart -lstdc++
#include <cuda_runtime.h>

extern cudaError_t launch_offset_memcpy(float* dst, const float* src,
                                         const int* counter, int dim,
                                         int maxSeqLen, cudaStream_t stream);
extern cudaError_t launch_offset_memcpy_fp16(void* dst, const float* src,
                                              const int* counter, int dim,
                                              int maxSeqLen, cudaStream_t stream);
*/
import "C"
import "unsafe"

// OffsetMemcpy copies dim floats from src to dst at offset counter*dim.
// counter is a GPU-resident int32. Used for GPU-driven KV cache append.
func OffsetMemcpy(dst, src, counter unsafe.Pointer, dim, maxSeqLen int, s unsafe.Pointer) error {
	return checkCUDA(C.launch_offset_memcpy(
		(*C.float)(dst), (*C.float)(src), (*C.int)(counter),
		C.int(dim), C.int(maxSeqLen), stream(s),
	), "offset_memcpy")
}

// OffsetMemcpyFP16 copies dim floats from F32 src to FP16 dst at offset counter*dim.
// counter is a GPU-resident int32. Used for GPU-driven FP16 KV cache append.
func OffsetMemcpyFP16(dst, src, counter unsafe.Pointer, dim, maxSeqLen int, s unsafe.Pointer) error {
	return checkCUDA(C.launch_offset_memcpy_fp16(
		dst, (*C.float)(src), (*C.int)(counter),
		C.int(dim), C.int(maxSeqLen), stream(s),
	), "offset_memcpy_fp16")
}
