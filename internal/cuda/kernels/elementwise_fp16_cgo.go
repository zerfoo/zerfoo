//go:build cuda

package kernels

/*
#cgo LDFLAGS: -L${SRCDIR} -lkernels -lcudart -lstdc++
#include <cuda_runtime.h>

extern cudaError_t launch_add_fp16(const void* a, const void* b, void* c, int n, cudaStream_t stream);
extern cudaError_t launch_sub_fp16(const void* a, const void* b, void* c, int n, cudaStream_t stream);
extern cudaError_t launch_mul_fp16(const void* a, const void* b, void* c, int n, cudaStream_t stream);
extern cudaError_t launch_div_fp16(const void* a, const void* b, void* c, int n, cudaStream_t stream);
extern cudaError_t launch_rmsnorm_fp16(const void* input, const void* weight, void* output, unsigned int eps_bits, int rows, int D, cudaStream_t stream);
extern cudaError_t launch_scaled_softmax_fp16(const void* input, void* output, int outer, int inner, int axisSize, unsigned int scale_bits, cudaStream_t stream);
extern cudaError_t launch_f32_to_fp16(const void* src, void* dst, int n, cudaStream_t stream);
extern cudaError_t launch_fp16_to_f32(const void* src, void* dst, int n, cudaStream_t stream);
*/
import "C"

import (
	"fmt"
	"math"
	"unsafe"
)

func AddFP16(a, b, c unsafe.Pointer, n int, s unsafe.Pointer) error {
	err := C.launch_add_fp16(a, b, c, C.int(n), C.cudaStream_t(s))
	if err != C.cudaSuccess {
		return fmt.Errorf("add_fp16: %s", C.GoString(C.cudaGetErrorString(err)))
	}
	return nil
}

func SubFP16(a, b, c unsafe.Pointer, n int, s unsafe.Pointer) error {
	err := C.launch_sub_fp16(a, b, c, C.int(n), C.cudaStream_t(s))
	if err != C.cudaSuccess {
		return fmt.Errorf("sub_fp16: %s", C.GoString(C.cudaGetErrorString(err)))
	}
	return nil
}

func MulFP16(a, b, c unsafe.Pointer, n int, s unsafe.Pointer) error {
	err := C.launch_mul_fp16(a, b, c, C.int(n), C.cudaStream_t(s))
	if err != C.cudaSuccess {
		return fmt.Errorf("mul_fp16: %s", C.GoString(C.cudaGetErrorString(err)))
	}
	return nil
}

func DivFP16(a, b, c unsafe.Pointer, n int, s unsafe.Pointer) error {
	err := C.launch_div_fp16(a, b, c, C.int(n), C.cudaStream_t(s))
	if err != C.cudaSuccess {
		return fmt.Errorf("div_fp16: %s", C.GoString(C.cudaGetErrorString(err)))
	}
	return nil
}

func RMSNormFP16(input, weight, output unsafe.Pointer, eps float32, rows, D int, s unsafe.Pointer) error {
	err := C.launch_rmsnorm_fp16(input, weight, output, C.uint(math.Float32bits(eps)), C.int(rows), C.int(D), C.cudaStream_t(s))
	if err != C.cudaSuccess {
		return fmt.Errorf("rmsnorm_fp16: %s", C.GoString(C.cudaGetErrorString(err)))
	}
	return nil
}

func ScaledSoftmaxFP16(input, output unsafe.Pointer, outer, inner, axisSize int, scale float32, stream unsafe.Pointer) error {
	err := C.launch_scaled_softmax_fp16(input, output, C.int(outer), C.int(inner), C.int(axisSize), C.uint(math.Float32bits(scale)), C.cudaStream_t(stream))
	if err != C.cudaSuccess {
		return fmt.Errorf("scaled_softmax_fp16: %s", C.GoString(C.cudaGetErrorString(err)))
	}
	return nil
}

func F32ToFP16(src, dst unsafe.Pointer, n int, s unsafe.Pointer) error {
	err := C.launch_f32_to_fp16(src, dst, C.int(n), C.cudaStream_t(s))
	if err != C.cudaSuccess {
		return fmt.Errorf("f32_to_fp16: %s", C.GoString(C.cudaGetErrorString(err)))
	}
	return nil
}

func FP16ToF32(src, dst unsafe.Pointer, n int, s unsafe.Pointer) error {
	err := C.launch_fp16_to_f32(src, dst, C.int(n), C.cudaStream_t(s))
	if err != C.cudaSuccess {
		return fmt.Errorf("fp16_to_f32: %s", C.GoString(C.cudaGetErrorString(err)))
	}
	return nil
}
