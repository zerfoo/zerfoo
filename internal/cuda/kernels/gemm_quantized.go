//go:build cuda && cutlass

package kernels

/*
#cgo LDFLAGS: -L${SRCDIR} -lkernels -lcudart -lstdc++
#include "gemm_int8.h"
#include "gemm_int4.h"
*/
import "C"

import (
	"fmt"
	"unsafe"
)

// GemmInt8F32 performs mixed-precision GEMM: out = int8(A) * float32(B).
// A is [M, K] row-major INT8 weights, B is [K, N] row-major FP32 activations.
// out is [M, N] row-major FP32 output.
func GemmInt8F32(
	A, B, out unsafe.Pointer,
	M, K, N int,
	stream unsafe.Pointer,
) error {
	err := C.gemm_int8_f32(
		A, (*C.float)(B), (*C.float)(out),
		C.int(M), C.int(K), C.int(N),
		C.cudaStream_t(stream),
	)
	if err != C.cudaSuccess {
		return fmt.Errorf("gemm_int8_f32: %s",
			C.GoString(C.cudaGetErrorString(err)))
	}
	return nil
}

// GemmInt4F32 performs block-quantized INT4 mixed-precision GEMM:
// out = dequant(A) * float32(B) where dequant uses per-group scales and zeros.
// A is [M, K/2] packed INT4 (two values per byte, low nibble first).
// B is [K, N] row-major FP32 activations.
// out is [M, N] row-major FP32 output.
// scales is [M, K/group_size] FP32 per-group scale factors.
// zeros is [M, K/group_size] uint8 per-group zero points.
// K must be even. group_size is typically 32 or 128.
func GemmInt4F32(
	A, B, out unsafe.Pointer,
	scales, zeros unsafe.Pointer,
	M, K, N, groupSize int,
	stream unsafe.Pointer,
) error {
	err := C.gemm_int4_f32(
		A, (*C.float)(B), (*C.float)(out),
		(*C.float)(scales), zeros,
		C.int(M), C.int(K), C.int(N), C.int(groupSize),
		C.cudaStream_t(stream),
	)
	if err != C.cudaSuccess {
		return fmt.Errorf("gemm_int4_f32: %s",
			C.GoString(C.cudaGetErrorString(err)))
	}
	return nil
}

// GemmInt4F32RMul performs right-multiply: out = input * dequant(W).
// This is the standard neural network forward pass operation.
// W is [inFeatures, outFeatures/2] packed INT4 weights.
// input is [batch, inFeatures] row-major FP32 activations.
// out is [batch, outFeatures] row-major FP32 output.
// scales is [inFeatures, numGroups] FP32 per-group scale factors.
// zeros is [inFeatures, numGroups] uint8 per-group zero points.
// outFeatures must be even. groupSize is typically 32 or 128.
func GemmInt4F32RMul(
	W, input, out unsafe.Pointer,
	scales, zeros unsafe.Pointer,
	batch, inFeatures, outFeatures, groupSize int,
	stream unsafe.Pointer,
) error {
	err := C.gemm_int4_f32_rmul(
		W, (*C.float)(input), (*C.float)(out),
		(*C.float)(scales), zeros,
		C.int(batch), C.int(inFeatures), C.int(outFeatures), C.int(groupSize),
		C.cudaStream_t(stream),
	)
	if err != C.cudaSuccess {
		return fmt.Errorf("gemm_int4_f32_rmul: %s",
			C.GoString(C.cudaGetErrorString(err)))
	}
	return nil
}
