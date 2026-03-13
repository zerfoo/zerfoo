//go:build !cuda

package kernels

import (
	"fmt"
	"unsafe"

	"github.com/zerfoo/zerfoo/internal/cuda"
)

// AddFP16 launches the FP16 elementwise add kernel: c = a + b.
func AddFP16(a, b, c unsafe.Pointer, n int, s unsafe.Pointer) error {
	k := klib()
	if k == nil {
		return fmt.Errorf("add_fp16 kernel: kernels not available")
	}
	ret := cuda.Ccall(k.launchAddFP16, uintptr(a), uintptr(b), uintptr(c), uintptr(n), uintptr(s))
	return checkKernel(ret, "add_fp16")
}

// SubFP16 launches the FP16 elementwise subtract kernel: c = a - b.
func SubFP16(a, b, c unsafe.Pointer, n int, s unsafe.Pointer) error {
	k := klib()
	if k == nil {
		return fmt.Errorf("sub_fp16 kernel: kernels not available")
	}
	ret := cuda.Ccall(k.launchSubFP16, uintptr(a), uintptr(b), uintptr(c), uintptr(n), uintptr(s))
	return checkKernel(ret, "sub_fp16")
}

// MulFP16 launches the FP16 elementwise multiply kernel: c = a * b.
func MulFP16(a, b, c unsafe.Pointer, n int, s unsafe.Pointer) error {
	k := klib()
	if k == nil {
		return fmt.Errorf("mul_fp16 kernel: kernels not available")
	}
	ret := cuda.Ccall(k.launchMulFP16, uintptr(a), uintptr(b), uintptr(c), uintptr(n), uintptr(s))
	return checkKernel(ret, "mul_fp16")
}

// DivFP16 launches the FP16 elementwise divide kernel: c = a / b.
func DivFP16(a, b, c unsafe.Pointer, n int, s unsafe.Pointer) error {
	k := klib()
	if k == nil {
		return fmt.Errorf("div_fp16 kernel: kernels not available")
	}
	ret := cuda.Ccall(k.launchDivFP16, uintptr(a), uintptr(b), uintptr(c), uintptr(n), uintptr(s))
	return checkKernel(ret, "div_fp16")
}

// RMSNormFP16 launches the FP16 RMSNorm kernel with FP32 accumulation.
// input: [rows, D], weight: [D], output: [rows, D].
func RMSNormFP16(input, weight, output unsafe.Pointer, eps float32, rows, D int, s unsafe.Pointer) error {
	k := klib()
	if k == nil {
		return fmt.Errorf("rmsnorm_fp16 kernel: kernels not available")
	}
	ret := cuda.Ccall(k.launchRMSNormFP16,
		uintptr(input), uintptr(weight), uintptr(output),
		floatBits(eps), uintptr(rows), uintptr(D), uintptr(s))
	return checkKernel(ret, "rmsnorm_fp16")
}

// ScaledSoftmaxFP16 applies fused scaled softmax on FP16 data with FP32 accumulation.
func ScaledSoftmaxFP16(
	input, output unsafe.Pointer,
	outer, inner, axisSize int,
	scale float32,
	stream unsafe.Pointer,
) error {
	k := klib()
	if k == nil {
		return fmt.Errorf("scaled_softmax_fp16 kernel: kernels not available")
	}
	ret := cuda.Ccall(k.launchScaledSoftmaxFP16,
		uintptr(input), uintptr(output),
		uintptr(outer), uintptr(inner), uintptr(axisSize),
		floatBits(scale),
		uintptr(stream))
	return checkKernel(ret, "scaled_softmax_fp16")
}
