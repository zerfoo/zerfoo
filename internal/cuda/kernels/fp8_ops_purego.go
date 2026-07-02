package kernels

import (
	"fmt"
	"unsafe"

	"github.com/zerfoo/zerfoo/internal/cuda"
)

// DequantFP8E4M3ToFP16 launches the FP8 E4M3 -> FP16 dequantization kernel.
// input: n bytes of FP8 E4M3 data, output: n FP16 values, scale: per-tensor scale factor.
func DequantFP8E4M3ToFP16(input, output unsafe.Pointer, scale float32, n int, s unsafe.Pointer) error {
	k := klib()
	if k == nil {
		return fmt.Errorf("dequant_fp8e4m3_to_fp16 kernel: kernels not available")
	}
	ret := cuda.Ccall(k.launchDequantFP8E4M3ToFP16,
		uintptr(input), uintptr(output), floatBits(scale), uintptr(n), uintptr(s))
	return checkKernel(ret, "dequant_fp8e4m3_to_fp16")
}

// FP8Add launches the FP8 dequant+add kernel: c[i] = dequant(a[i])*scaleA + dequant(b[i])*scaleB.
// a, b: FP8 E4M3 inputs, c: FP16 output.
func FP8Add(a, b, c unsafe.Pointer, scaleA, scaleB float32, n int, s unsafe.Pointer) error {
	k := klib()
	if k == nil {
		return fmt.Errorf("fp8_add kernel: kernels not available")
	}
	ret := cuda.Ccall(k.launchFP8Add,
		uintptr(a), uintptr(b), uintptr(c),
		floatBits(scaleA), floatBits(scaleB), uintptr(n), uintptr(s))
	return checkKernel(ret, "fp8_add")
}

// FP8Mul launches the FP8 dequant+mul kernel: c[i] = dequant(a[i])*scaleA * dequant(b[i])*scaleB.
// a, b: FP8 E4M3 inputs, c: FP16 output.
func FP8Mul(a, b, c unsafe.Pointer, scaleA, scaleB float32, n int, s unsafe.Pointer) error {
	k := klib()
	if k == nil {
		return fmt.Errorf("fp8_mul kernel: kernels not available")
	}
	ret := cuda.Ccall(k.launchFP8Mul,
		uintptr(a), uintptr(b), uintptr(c),
		floatBits(scaleA), floatBits(scaleB), uintptr(n), uintptr(s))
	return checkKernel(ret, "fp8_mul")
}

// FP8RMSNorm launches the FP8 dequant+RMSNorm kernel.
// input: FP8 E4M3 [rows, D], weight: FP16 [D], output: FP16 [rows, D].
// Dequantizes input on load, computes RMSNorm with FP32 accumulation, writes FP16.
func FP8RMSNorm(input, weight, output unsafe.Pointer, scale, eps float32, rows, D int, s unsafe.Pointer) error {
	k := klib()
	if k == nil {
		return fmt.Errorf("fp8_rmsnorm kernel: kernels not available")
	}
	ret := cuda.Ccall(k.launchFP8RMSNorm,
		uintptr(input), uintptr(weight), uintptr(output),
		floatBits(scale), floatBits(eps), uintptr(rows), uintptr(D), uintptr(s))
	return checkKernel(ret, "fp8_rmsnorm")
}
