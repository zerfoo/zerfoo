//go:build cuda

package kernels

/*
#cgo LDFLAGS: -L${SRCDIR} -lkernels -lcudart -lstdc++
#include <cuda_runtime.h>

// Forward declarations of launcher functions from elementwise.cu
extern cudaError_t launch_add(const float* a, const float* b, float* c, int n);
extern cudaError_t launch_sub(const float* a, const float* b, float* c, int n);
extern cudaError_t launch_mul(const float* a, const float* b, float* c, int n);
extern cudaError_t launch_div(const float* a, const float* b, float* c, int n);
extern cudaError_t launch_pow(const float* base, const float* exp, float* c, int n);
extern cudaError_t launch_add_scalar(const float* a, float scalar, float* c, int n);
extern cudaError_t launch_mul_scalar(const float* a, float scalar, float* c, int n);
extern cudaError_t launch_div_scalar(const float* a, float scalar, float* c, int n);
extern cudaError_t launch_exp(const float* a, float* c, int n);
extern cudaError_t launch_log(const float* a, float* c, int n);
extern cudaError_t launch_sqrt(const float* a, float* c, int n);
extern cudaError_t launch_rsqrt(const float* a, float* c, int n);
extern cudaError_t launch_tanh(const float* a, float* c, int n);
extern cudaError_t launch_tanh_prime(const float* a, const float* upstream, float* c, int n);
extern cudaError_t launch_fill(float* data, float value, int n);
extern cudaError_t launch_sum_axis(const float* input, float* output, int outer, int inner, int axisSize);
extern cudaError_t launch_softmax(const float* input, float* output, int outer, int inner, int axisSize);
*/
import "C"

import (
	"fmt"
	"unsafe"
)

func checkCUDA(err C.cudaError_t, op string) error {
	if err != C.cudaSuccess {
		return fmt.Errorf("%s kernel failed: %s", op, C.GoString(C.cudaGetErrorString(err)))
	}

	return nil
}

// Add launches the elementwise add kernel: c = a + b.
func Add(a, b, c unsafe.Pointer, n int) error {
	return checkCUDA(C.launch_add((*C.float)(a), (*C.float)(b), (*C.float)(c), C.int(n)), "add")
}

// Sub launches the elementwise subtract kernel: c = a - b.
func Sub(a, b, c unsafe.Pointer, n int) error {
	return checkCUDA(C.launch_sub((*C.float)(a), (*C.float)(b), (*C.float)(c), C.int(n)), "sub")
}

// Mul launches the elementwise multiply kernel: c = a * b.
func Mul(a, b, c unsafe.Pointer, n int) error {
	return checkCUDA(C.launch_mul((*C.float)(a), (*C.float)(b), (*C.float)(c), C.int(n)), "mul")
}

// Div launches the elementwise divide kernel: c = a / b.
func Div(a, b, c unsafe.Pointer, n int) error {
	return checkCUDA(C.launch_div((*C.float)(a), (*C.float)(b), (*C.float)(c), C.int(n)), "div")
}

// Pow launches the elementwise power kernel: c = base ^ exp.
func Pow(base, exp, c unsafe.Pointer, n int) error {
	return checkCUDA(C.launch_pow((*C.float)(base), (*C.float)(exp), (*C.float)(c), C.int(n)), "pow")
}

// AddScalar launches the scalar add kernel: c = a + scalar.
func AddScalar(a unsafe.Pointer, scalar float32, c unsafe.Pointer, n int) error {
	return checkCUDA(C.launch_add_scalar((*C.float)(a), C.float(scalar), (*C.float)(c), C.int(n)), "add_scalar")
}

// MulScalar launches the scalar multiply kernel: c = a * scalar.
func MulScalar(a unsafe.Pointer, scalar float32, c unsafe.Pointer, n int) error {
	return checkCUDA(C.launch_mul_scalar((*C.float)(a), C.float(scalar), (*C.float)(c), C.int(n)), "mul_scalar")
}

// DivScalar launches the scalar divide kernel: c = a / scalar.
func DivScalar(a unsafe.Pointer, scalar float32, c unsafe.Pointer, n int) error {
	return checkCUDA(C.launch_div_scalar((*C.float)(a), C.float(scalar), (*C.float)(c), C.int(n)), "div_scalar")
}

// Exp launches the elementwise exp kernel: c = exp(a).
func Exp(a, c unsafe.Pointer, n int) error {
	return checkCUDA(C.launch_exp((*C.float)(a), (*C.float)(c), C.int(n)), "exp")
}

// Log launches the elementwise log kernel: c = log(a).
func Log(a, c unsafe.Pointer, n int) error {
	return checkCUDA(C.launch_log((*C.float)(a), (*C.float)(c), C.int(n)), "log")
}

// Sqrt launches the elementwise sqrt kernel: c = sqrt(a).
func Sqrt(a, c unsafe.Pointer, n int) error {
	return checkCUDA(C.launch_sqrt((*C.float)(a), (*C.float)(c), C.int(n)), "sqrt")
}

// Rsqrt launches the elementwise rsqrt kernel: c = 1/sqrt(a).
func Rsqrt(a, c unsafe.Pointer, n int) error {
	return checkCUDA(C.launch_rsqrt((*C.float)(a), (*C.float)(c), C.int(n)), "rsqrt")
}

// Tanh launches the elementwise tanh kernel: c = tanh(a).
func Tanh(a, c unsafe.Pointer, n int) error {
	return checkCUDA(C.launch_tanh((*C.float)(a), (*C.float)(c), C.int(n)), "tanh")
}

// TanhPrime launches the tanh derivative kernel: c = (1 - tanh(a)^2) * upstream.
func TanhPrime(a, upstream, c unsafe.Pointer, n int) error {
	return checkCUDA(C.launch_tanh_prime((*C.float)(a), (*C.float)(upstream), (*C.float)(c), C.int(n)), "tanh_prime")
}

// Fill launches the fill kernel: sets all elements to value.
func Fill(data unsafe.Pointer, value float32, n int) error {
	return checkCUDA(C.launch_fill((*C.float)(data), C.float(value), C.int(n)), "fill")
}

// SumAxis launches the sum-reduction kernel along an axis defined by outer/inner/axisSize strides.
// Output has outer*inner elements, one sum per (outer, inner) stripe.
func SumAxis(input, output unsafe.Pointer, outer, inner, axisSize int) error {
	return checkCUDA(C.launch_sum_axis((*C.float)(input), (*C.float)(output), C.int(outer), C.int(inner), C.int(axisSize)), "sum_axis")
}

// Softmax launches the softmax kernel along an axis defined by outer/inner/axisSize strides.
func Softmax(input, output unsafe.Pointer, outer, inner, axisSize int) error {
	return checkCUDA(C.launch_softmax((*C.float)(input), (*C.float)(output), C.int(outer), C.int(inner), C.int(axisSize)), "softmax")
}
