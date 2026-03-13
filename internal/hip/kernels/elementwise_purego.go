package kernels

import (
	"fmt"
	"math"
	"unsafe"

	"github.com/zerfoo/zerfoo/internal/cuda"
)

func checkHIP(ret uintptr, op string) error {
	if ret != 0 {
		return fmt.Errorf("%s kernel failed (hip error %d)", op, ret)
	}
	return nil
}

// floatBits reinterprets a float32 as a uintptr for passing to ccall.
func floatBits(f float32) uintptr {
	return uintptr(math.Float32bits(f))
}

// Add launches the elementwise add kernel: c = a + b.
func Add(a, b, c unsafe.Pointer, n int, s unsafe.Pointer) error {
	k := klib()
	if k == nil {
		return fmt.Errorf("add kernel: hip kernels not available")
	}
	ret := cuda.Ccall(k.launchAdd, uintptr(a), uintptr(b), uintptr(c), uintptr(n), uintptr(s))
	return checkHIP(ret, "add")
}

// Sub launches the elementwise subtract kernel: c = a - b.
func Sub(a, b, c unsafe.Pointer, n int, s unsafe.Pointer) error {
	k := klib()
	if k == nil {
		return fmt.Errorf("sub kernel: hip kernels not available")
	}
	ret := cuda.Ccall(k.launchSub, uintptr(a), uintptr(b), uintptr(c), uintptr(n), uintptr(s))
	return checkHIP(ret, "sub")
}

// Mul launches the elementwise multiply kernel: c = a * b.
func Mul(a, b, c unsafe.Pointer, n int, s unsafe.Pointer) error {
	k := klib()
	if k == nil {
		return fmt.Errorf("mul kernel: hip kernels not available")
	}
	ret := cuda.Ccall(k.launchMul, uintptr(a), uintptr(b), uintptr(c), uintptr(n), uintptr(s))
	return checkHIP(ret, "mul")
}

// Div launches the elementwise divide kernel: c = a / b.
func Div(a, b, c unsafe.Pointer, n int, s unsafe.Pointer) error {
	k := klib()
	if k == nil {
		return fmt.Errorf("div kernel: hip kernels not available")
	}
	ret := cuda.Ccall(k.launchDiv, uintptr(a), uintptr(b), uintptr(c), uintptr(n), uintptr(s))
	return checkHIP(ret, "div")
}

// Pow launches the elementwise power kernel: c = base ^ exp.
func Pow(base, exp, c unsafe.Pointer, n int, s unsafe.Pointer) error {
	k := klib()
	if k == nil {
		return fmt.Errorf("pow kernel: hip kernels not available")
	}
	ret := cuda.Ccall(k.launchPow, uintptr(base), uintptr(exp), uintptr(c), uintptr(n), uintptr(s))
	return checkHIP(ret, "pow")
}

// AddScalar launches the scalar add kernel: c = a + scalar.
func AddScalar(a unsafe.Pointer, scalar float32, c unsafe.Pointer, n int, s unsafe.Pointer) error {
	k := klib()
	if k == nil {
		return fmt.Errorf("add_scalar kernel: hip kernels not available")
	}
	ret := cuda.Ccall(k.launchAddScalar, uintptr(a), floatBits(scalar), uintptr(c), uintptr(n), uintptr(s))
	return checkHIP(ret, "add_scalar")
}

// MulScalar launches the scalar multiply kernel: c = a * scalar.
func MulScalar(a unsafe.Pointer, scalar float32, c unsafe.Pointer, n int, s unsafe.Pointer) error {
	k := klib()
	if k == nil {
		return fmt.Errorf("mul_scalar kernel: hip kernels not available")
	}
	ret := cuda.Ccall(k.launchMulScalar, uintptr(a), floatBits(scalar), uintptr(c), uintptr(n), uintptr(s))
	return checkHIP(ret, "mul_scalar")
}

// DivScalar launches the scalar divide kernel: c = a / scalar.
func DivScalar(a unsafe.Pointer, scalar float32, c unsafe.Pointer, n int, s unsafe.Pointer) error {
	k := klib()
	if k == nil {
		return fmt.Errorf("div_scalar kernel: hip kernels not available")
	}
	ret := cuda.Ccall(k.launchDivScalar, uintptr(a), floatBits(scalar), uintptr(c), uintptr(n), uintptr(s))
	return checkHIP(ret, "div_scalar")
}

// Exp launches the elementwise exp kernel: c = exp(a).
func Exp(a, c unsafe.Pointer, n int, s unsafe.Pointer) error {
	k := klib()
	if k == nil {
		return fmt.Errorf("exp kernel: hip kernels not available")
	}
	ret := cuda.Ccall(k.launchExp, uintptr(a), uintptr(c), uintptr(n), uintptr(s))
	return checkHIP(ret, "exp")
}

// Log launches the elementwise log kernel: c = log(a).
func Log(a, c unsafe.Pointer, n int, s unsafe.Pointer) error {
	k := klib()
	if k == nil {
		return fmt.Errorf("log kernel: hip kernels not available")
	}
	ret := cuda.Ccall(k.launchLog, uintptr(a), uintptr(c), uintptr(n), uintptr(s))
	return checkHIP(ret, "log")
}

// Sqrt launches the elementwise sqrt kernel: c = sqrt(a).
func Sqrt(a, c unsafe.Pointer, n int, s unsafe.Pointer) error {
	k := klib()
	if k == nil {
		return fmt.Errorf("sqrt kernel: hip kernels not available")
	}
	ret := cuda.Ccall(k.launchSqrt, uintptr(a), uintptr(c), uintptr(n), uintptr(s))
	return checkHIP(ret, "sqrt")
}

// Rsqrt launches the elementwise rsqrt kernel: c = 1/sqrt(a).
func Rsqrt(a, c unsafe.Pointer, n int, s unsafe.Pointer) error {
	k := klib()
	if k == nil {
		return fmt.Errorf("rsqrt kernel: hip kernels not available")
	}
	ret := cuda.Ccall(k.launchRsqrt, uintptr(a), uintptr(c), uintptr(n), uintptr(s))
	return checkHIP(ret, "rsqrt")
}

// Tanh launches the elementwise tanh kernel: c = tanh(a).
func Tanh(a, c unsafe.Pointer, n int, s unsafe.Pointer) error {
	k := klib()
	if k == nil {
		return fmt.Errorf("tanh kernel: hip kernels not available")
	}
	ret := cuda.Ccall(k.launchTanh, uintptr(a), uintptr(c), uintptr(n), uintptr(s))
	return checkHIP(ret, "tanh")
}

// TanhPrime launches the tanh derivative kernel: c = (1 - tanh(a)^2) * upstream.
func TanhPrime(a, upstream, c unsafe.Pointer, n int, s unsafe.Pointer) error {
	k := klib()
	if k == nil {
		return fmt.Errorf("tanh_prime kernel: hip kernels not available")
	}
	ret := cuda.Ccall(k.launchTanhPrime, uintptr(a), uintptr(upstream), uintptr(c), uintptr(n), uintptr(s))
	return checkHIP(ret, "tanh_prime")
}

// Fill launches the fill kernel: sets all elements to value.
func Fill(data unsafe.Pointer, value float32, n int, s unsafe.Pointer) error {
	k := klib()
	if k == nil {
		return fmt.Errorf("fill kernel: hip kernels not available")
	}
	ret := cuda.Ccall(k.launchFill, uintptr(data), floatBits(value), uintptr(n), uintptr(s))
	return checkHIP(ret, "fill")
}

// SumAxis launches the sum-reduction kernel along an axis.
func SumAxis(input, output unsafe.Pointer, outer, inner, axisSize int, s unsafe.Pointer) error {
	k := klib()
	if k == nil {
		return fmt.Errorf("sum_axis kernel: hip kernels not available")
	}
	ret := cuda.Ccall(k.launchSumAxis, uintptr(input), uintptr(output),
		uintptr(outer), uintptr(inner), uintptr(axisSize), uintptr(s))
	return checkHIP(ret, "sum_axis")
}

// Softmax launches the softmax kernel along an axis.
func Softmax(input, output unsafe.Pointer, outer, inner, axisSize int, s unsafe.Pointer) error {
	k := klib()
	if k == nil {
		return fmt.Errorf("softmax kernel: hip kernels not available")
	}
	ret := cuda.Ccall(k.launchSoftmax, uintptr(input), uintptr(output),
		uintptr(outer), uintptr(inner), uintptr(axisSize), uintptr(s))
	return checkHIP(ret, "softmax")
}

// FlashAttentionForward computes scaled dot-product attention using a fused
// tiled kernel. All tensors are in [batch, heads, seq_len, head_dim] layout.
// When causal is true, an upper-triangular mask is applied.
func FlashAttentionForward(
	Q, K, V, O unsafe.Pointer,
	batch, heads, seqLen, headDim int,
	causal bool,
	stream unsafe.Pointer,
) error {
	k := klib()
	if k == nil {
		return fmt.Errorf("flash_attention kernel: hip kernels not available")
	}
	var c uintptr
	if causal {
		c = 1
	}
	ret := cuda.Ccall(k.launchFlashAttentionF32,
		uintptr(Q), uintptr(K), uintptr(V), uintptr(O),
		uintptr(batch), uintptr(heads), uintptr(seqLen), uintptr(headDim),
		c, uintptr(stream))
	return checkHIP(ret, "flash_attention_forward_f32")
}
