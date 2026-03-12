//go:build !cuda

package kernels

import (
	"fmt"
	"math"
	"unsafe"

	"github.com/zerfoo/zerfoo/internal/cuda"
)

func checkKernel(ret uintptr, op string) error {
	if ret != 0 {
		return fmt.Errorf("%s kernel failed (cuda error %d)", op, ret)
	}
	return nil
}

// floatBits reinterprets a float32 as a uintptr for passing to ccall.
// CUDA C ABI passes float in integer registers on arm64.
func floatBits(f float32) uintptr {
	return uintptr(math.Float32bits(f))
}

// Add launches the elementwise add kernel: c = a + b.
func Add(a, b, c unsafe.Pointer, n int, s unsafe.Pointer) error {
	k := klib()
	if k == nil {
		return fmt.Errorf("add kernel: kernels not available")
	}
	ret := cuda.Ccall(k.launchAdd, uintptr(a), uintptr(b), uintptr(c), uintptr(n), uintptr(s))
	return checkKernel(ret, "add")
}

// Sub launches the elementwise subtract kernel: c = a - b.
func Sub(a, b, c unsafe.Pointer, n int, s unsafe.Pointer) error {
	k := klib()
	if k == nil {
		return fmt.Errorf("sub kernel: kernels not available")
	}
	ret := cuda.Ccall(k.launchSub, uintptr(a), uintptr(b), uintptr(c), uintptr(n), uintptr(s))
	return checkKernel(ret, "sub")
}

// Mul launches the elementwise multiply kernel: c = a * b.
func Mul(a, b, c unsafe.Pointer, n int, s unsafe.Pointer) error {
	k := klib()
	if k == nil {
		return fmt.Errorf("mul kernel: kernels not available")
	}
	ret := cuda.Ccall(k.launchMul, uintptr(a), uintptr(b), uintptr(c), uintptr(n), uintptr(s))
	return checkKernel(ret, "mul")
}

// Div launches the elementwise divide kernel: c = a / b.
func Div(a, b, c unsafe.Pointer, n int, s unsafe.Pointer) error {
	k := klib()
	if k == nil {
		return fmt.Errorf("div kernel: kernels not available")
	}
	ret := cuda.Ccall(k.launchDiv, uintptr(a), uintptr(b), uintptr(c), uintptr(n), uintptr(s))
	return checkKernel(ret, "div")
}

// Pow launches the elementwise power kernel: c = base ^ exp.
func Pow(base, exp, c unsafe.Pointer, n int, s unsafe.Pointer) error {
	k := klib()
	if k == nil {
		return fmt.Errorf("pow kernel: kernels not available")
	}
	ret := cuda.Ccall(k.launchPow, uintptr(base), uintptr(exp), uintptr(c), uintptr(n), uintptr(s))
	return checkKernel(ret, "pow")
}

// AddScalar launches the scalar add kernel: c = a + scalar.
func AddScalar(a unsafe.Pointer, scalar float32, c unsafe.Pointer, n int, s unsafe.Pointer) error {
	k := klib()
	if k == nil {
		return fmt.Errorf("add_scalar kernel: kernels not available")
	}
	ret := cuda.Ccall(k.launchAddScalar, uintptr(a), floatBits(scalar), uintptr(c), uintptr(n), uintptr(s))
	return checkKernel(ret, "add_scalar")
}

// MulScalar launches the scalar multiply kernel: c = a * scalar.
func MulScalar(a unsafe.Pointer, scalar float32, c unsafe.Pointer, n int, s unsafe.Pointer) error {
	k := klib()
	if k == nil {
		return fmt.Errorf("mul_scalar kernel: kernels not available")
	}
	ret := cuda.Ccall(k.launchMulScalar, uintptr(a), floatBits(scalar), uintptr(c), uintptr(n), uintptr(s))
	return checkKernel(ret, "mul_scalar")
}

// DivScalar launches the scalar divide kernel: c = a / scalar.
func DivScalar(a unsafe.Pointer, scalar float32, c unsafe.Pointer, n int, s unsafe.Pointer) error {
	k := klib()
	if k == nil {
		return fmt.Errorf("div_scalar kernel: kernels not available")
	}
	ret := cuda.Ccall(k.launchDivScalar, uintptr(a), floatBits(scalar), uintptr(c), uintptr(n), uintptr(s))
	return checkKernel(ret, "div_scalar")
}

// SubScalar launches the scalar subtract kernel: c = a - scalar.
func SubScalar(a unsafe.Pointer, scalar float32, c unsafe.Pointer, n int, s unsafe.Pointer) error {
	k := klib()
	if k == nil {
		return fmt.Errorf("sub_scalar kernel: kernels not available")
	}
	ret := cuda.Ccall(k.launchSubScalar, uintptr(a), floatBits(scalar), uintptr(c), uintptr(n), uintptr(s))
	return checkKernel(ret, "sub_scalar")
}

// PowScalar launches the scalar power kernel: c = pow(a, scalar).
func PowScalar(a unsafe.Pointer, scalar float32, c unsafe.Pointer, n int, s unsafe.Pointer) error {
	k := klib()
	if k == nil {
		return fmt.Errorf("pow_scalar kernel: kernels not available")
	}
	ret := cuda.Ccall(k.launchPowScalar, uintptr(a), floatBits(scalar), uintptr(c), uintptr(n), uintptr(s))
	return checkKernel(ret, "pow_scalar")
}

// Exp launches the elementwise exp kernel: c = exp(a).
func Exp(a, c unsafe.Pointer, n int, s unsafe.Pointer) error {
	k := klib()
	if k == nil {
		return fmt.Errorf("exp kernel: kernels not available")
	}
	ret := cuda.Ccall(k.launchExp, uintptr(a), uintptr(c), uintptr(n), uintptr(s))
	return checkKernel(ret, "exp")
}

// Log launches the elementwise log kernel: c = log(a).
func Log(a, c unsafe.Pointer, n int, s unsafe.Pointer) error {
	k := klib()
	if k == nil {
		return fmt.Errorf("log kernel: kernels not available")
	}
	ret := cuda.Ccall(k.launchLog, uintptr(a), uintptr(c), uintptr(n), uintptr(s))
	return checkKernel(ret, "log")
}

// Sqrt launches the elementwise sqrt kernel: c = sqrt(a).
func Sqrt(a, c unsafe.Pointer, n int, s unsafe.Pointer) error {
	k := klib()
	if k == nil {
		return fmt.Errorf("sqrt kernel: kernels not available")
	}
	ret := cuda.Ccall(k.launchSqrt, uintptr(a), uintptr(c), uintptr(n), uintptr(s))
	return checkKernel(ret, "sqrt")
}

// Rsqrt launches the elementwise rsqrt kernel: c = 1/sqrt(a).
func Rsqrt(a, c unsafe.Pointer, n int, s unsafe.Pointer) error {
	k := klib()
	if k == nil {
		return fmt.Errorf("rsqrt kernel: kernels not available")
	}
	ret := cuda.Ccall(k.launchRsqrt, uintptr(a), uintptr(c), uintptr(n), uintptr(s))
	return checkKernel(ret, "rsqrt")
}

// Tanh launches the elementwise tanh kernel: c = tanh(a).
func Tanh(a, c unsafe.Pointer, n int, s unsafe.Pointer) error {
	k := klib()
	if k == nil {
		return fmt.Errorf("tanh kernel: kernels not available")
	}
	ret := cuda.Ccall(k.launchTanh, uintptr(a), uintptr(c), uintptr(n), uintptr(s))
	return checkKernel(ret, "tanh")
}

// TanhPrime launches the tanh derivative kernel: c = (1 - tanh(a)^2) * upstream.
func TanhPrime(a, upstream, c unsafe.Pointer, n int, s unsafe.Pointer) error {
	k := klib()
	if k == nil {
		return fmt.Errorf("tanh_prime kernel: kernels not available")
	}
	ret := cuda.Ccall(k.launchTanhPrime, uintptr(a), uintptr(upstream), uintptr(c), uintptr(n), uintptr(s))
	return checkKernel(ret, "tanh_prime")
}

// Fill launches the fill kernel: sets all elements to value.
func Fill(data unsafe.Pointer, value float32, n int, s unsafe.Pointer) error {
	k := klib()
	if k == nil {
		return fmt.Errorf("fill kernel: kernels not available")
	}
	ret := cuda.Ccall(k.launchFill, uintptr(data), floatBits(value), uintptr(n), uintptr(s))
	return checkKernel(ret, "fill")
}

// SumAxis launches the sum-reduction kernel along an axis.
func SumAxis(input, output unsafe.Pointer, outer, inner, axisSize int, s unsafe.Pointer) error {
	k := klib()
	if k == nil {
		return fmt.Errorf("sum_axis kernel: kernels not available")
	}
	ret := cuda.Ccall(k.launchSumAxis, uintptr(input), uintptr(output),
		uintptr(outer), uintptr(inner), uintptr(axisSize), uintptr(s))
	return checkKernel(ret, "sum_axis")
}

// Softmax launches the softmax kernel along an axis.
func Softmax(input, output unsafe.Pointer, outer, inner, axisSize int, s unsafe.Pointer) error {
	k := klib()
	if k == nil {
		return fmt.Errorf("softmax kernel: kernels not available")
	}
	ret := cuda.Ccall(k.launchSoftmax, uintptr(input), uintptr(output),
		uintptr(outer), uintptr(inner), uintptr(axisSize), uintptr(s))
	return checkKernel(ret, "softmax")
}

// AddBroadcast launches the broadcast add kernel.
func AddBroadcast(a, b, c unsafe.Pointer, saRow, saCol, sbRow, sbCol, M, D int, s unsafe.Pointer) error { //nolint:gocritic // match CGo API
	k := klib()
	if k == nil {
		return fmt.Errorf("add_broadcast kernel: kernels not available")
	}
	ret := cuda.Ccall(k.launchAddBroadcast, uintptr(a), uintptr(b), uintptr(c),
		uintptr(saRow), uintptr(saCol), uintptr(sbRow), uintptr(sbCol),
		uintptr(M), uintptr(D), uintptr(s))
	return checkKernel(ret, "add_broadcast")
}

// SubBroadcast launches the broadcast sub kernel.
func SubBroadcast(a, b, c unsafe.Pointer, saRow, saCol, sbRow, sbCol, M, D int, s unsafe.Pointer) error { //nolint:gocritic // match CGo API
	k := klib()
	if k == nil {
		return fmt.Errorf("sub_broadcast kernel: kernels not available")
	}
	ret := cuda.Ccall(k.launchSubBroadcast, uintptr(a), uintptr(b), uintptr(c),
		uintptr(saRow), uintptr(saCol), uintptr(sbRow), uintptr(sbCol),
		uintptr(M), uintptr(D), uintptr(s))
	return checkKernel(ret, "sub_broadcast")
}

// MulBroadcast launches the broadcast mul kernel.
func MulBroadcast(a, b, c unsafe.Pointer, saRow, saCol, sbRow, sbCol, M, D int, s unsafe.Pointer) error { //nolint:gocritic // match CGo API
	k := klib()
	if k == nil {
		return fmt.Errorf("mul_broadcast kernel: kernels not available")
	}
	ret := cuda.Ccall(k.launchMulBroadcast, uintptr(a), uintptr(b), uintptr(c),
		uintptr(saRow), uintptr(saCol), uintptr(sbRow), uintptr(sbCol),
		uintptr(M), uintptr(D), uintptr(s))
	return checkKernel(ret, "mul_broadcast")
}

// DivBroadcast launches the broadcast div kernel.
func DivBroadcast(a, b, c unsafe.Pointer, saRow, saCol, sbRow, sbCol, M, D int, s unsafe.Pointer) error { //nolint:gocritic // match CGo API
	k := klib()
	if k == nil {
		return fmt.Errorf("div_broadcast kernel: kernels not available")
	}
	ret := cuda.Ccall(k.launchDivBroadcast, uintptr(a), uintptr(b), uintptr(c),
		uintptr(saRow), uintptr(saCol), uintptr(sbRow), uintptr(sbCol),
		uintptr(M), uintptr(D), uintptr(s))
	return checkKernel(ret, "div_broadcast")
}
