//go:build cuda

package kernels

/*
#cgo LDFLAGS: -L${SRCDIR} -lkernels -lcudart -lstdc++
#include <cuda_runtime.h>

// Forward declarations of launcher functions from elementwise.cu
extern cudaError_t launch_add_broadcast(const float* a, const float* b, float* c,
    int stride_a_row, int stride_a_col, int stride_b_row, int stride_b_col,
    int M, int D, cudaStream_t stream);
extern cudaError_t launch_sub_broadcast(const float* a, const float* b, float* c,
    int stride_a_row, int stride_a_col, int stride_b_row, int stride_b_col,
    int M, int D, cudaStream_t stream);
extern cudaError_t launch_mul_broadcast(const float* a, const float* b, float* c,
    int stride_a_row, int stride_a_col, int stride_b_row, int stride_b_col,
    int M, int D, cudaStream_t stream);
extern cudaError_t launch_div_broadcast(const float* a, const float* b, float* c,
    int stride_a_row, int stride_a_col, int stride_b_row, int stride_b_col,
    int M, int D, cudaStream_t stream);
extern cudaError_t launch_add(const float* a, const float* b, float* c, int n, cudaStream_t stream);
extern cudaError_t launch_sub(const float* a, const float* b, float* c, int n, cudaStream_t stream);
extern cudaError_t launch_mul(const float* a, const float* b, float* c, int n, cudaStream_t stream);
extern cudaError_t launch_div(const float* a, const float* b, float* c, int n, cudaStream_t stream);
extern cudaError_t launch_pow(const float* base, const float* exp, float* c, int n, cudaStream_t stream);
extern cudaError_t launch_add_scalar(const float* a, float scalar, float* c, int n, cudaStream_t stream);
extern cudaError_t launch_mul_scalar(const float* a, float scalar, float* c, int n, cudaStream_t stream);
extern cudaError_t launch_div_scalar(const float* a, float scalar, float* c, int n, cudaStream_t stream);
extern cudaError_t launch_sub_scalar(const float* a, float scalar, float* c, int n, cudaStream_t stream);
extern cudaError_t launch_pow_scalar(const float* a, float scalar, float* c, int n, cudaStream_t stream);
extern cudaError_t launch_exp(const float* a, float* c, int n, cudaStream_t stream);
extern cudaError_t launch_log(const float* a, float* c, int n, cudaStream_t stream);
extern cudaError_t launch_sqrt(const float* a, float* c, int n, cudaStream_t stream);
extern cudaError_t launch_rsqrt(const float* a, float* c, int n, cudaStream_t stream);
extern cudaError_t launch_tanh(const float* a, float* c, int n, cudaStream_t stream);
extern cudaError_t launch_tanh_prime(const float* a, const float* upstream, float* c, int n, cudaStream_t stream);
extern cudaError_t launch_fill(float* data, float value, int n, cudaStream_t stream);
extern cudaError_t launch_sum_axis(const float* input, float* output, int outer, int inner, int axisSize, cudaStream_t stream);
extern cudaError_t launch_softmax(const float* input, float* output, int outer, int inner, int axisSize, cudaStream_t stream);
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

// stream converts an unsafe.Pointer to C.cudaStream_t.
// Pass nil for the default stream.
func stream(s unsafe.Pointer) C.cudaStream_t {
	return C.cudaStream_t(s)
}

// Add launches the elementwise add kernel: c = a + b.
func Add(a, b, c unsafe.Pointer, n int, s unsafe.Pointer) error {
	return checkCUDA(C.launch_add((*C.float)(a), (*C.float)(b), (*C.float)(c), C.int(n), stream(s)), "add")
}

// Sub launches the elementwise subtract kernel: c = a - b.
func Sub(a, b, c unsafe.Pointer, n int, s unsafe.Pointer) error {
	return checkCUDA(C.launch_sub((*C.float)(a), (*C.float)(b), (*C.float)(c), C.int(n), stream(s)), "sub")
}

// Mul launches the elementwise multiply kernel: c = a * b.
func Mul(a, b, c unsafe.Pointer, n int, s unsafe.Pointer) error {
	return checkCUDA(C.launch_mul((*C.float)(a), (*C.float)(b), (*C.float)(c), C.int(n), stream(s)), "mul")
}

// Div launches the elementwise divide kernel: c = a / b.
func Div(a, b, c unsafe.Pointer, n int, s unsafe.Pointer) error {
	return checkCUDA(C.launch_div((*C.float)(a), (*C.float)(b), (*C.float)(c), C.int(n), stream(s)), "div")
}

// Pow launches the elementwise power kernel: c = base ^ exp.
func Pow(base, exp, c unsafe.Pointer, n int, s unsafe.Pointer) error {
	return checkCUDA(C.launch_pow((*C.float)(base), (*C.float)(exp), (*C.float)(c), C.int(n), stream(s)), "pow")
}

// AddScalar launches the scalar add kernel: c = a + scalar.
func AddScalar(a unsafe.Pointer, scalar float32, c unsafe.Pointer, n int, s unsafe.Pointer) error {
	return checkCUDA(C.launch_add_scalar((*C.float)(a), C.float(scalar), (*C.float)(c), C.int(n), stream(s)), "add_scalar")
}

// MulScalar launches the scalar multiply kernel: c = a * scalar.
func MulScalar(a unsafe.Pointer, scalar float32, c unsafe.Pointer, n int, s unsafe.Pointer) error {
	return checkCUDA(C.launch_mul_scalar((*C.float)(a), C.float(scalar), (*C.float)(c), C.int(n), stream(s)), "mul_scalar")
}

// DivScalar launches the scalar divide kernel: c = a / scalar.
func DivScalar(a unsafe.Pointer, scalar float32, c unsafe.Pointer, n int, s unsafe.Pointer) error {
	return checkCUDA(C.launch_div_scalar((*C.float)(a), C.float(scalar), (*C.float)(c), C.int(n), stream(s)), "div_scalar")
}

// SubScalar launches the scalar subtract kernel: c = a - scalar.
func SubScalar(a unsafe.Pointer, scalar float32, c unsafe.Pointer, n int, s unsafe.Pointer) error {
	return checkCUDA(C.launch_sub_scalar((*C.float)(a), C.float(scalar), (*C.float)(c), C.int(n), stream(s)), "sub_scalar")
}

// PowScalar launches the scalar power kernel: c = pow(a, scalar).
func PowScalar(a unsafe.Pointer, scalar float32, c unsafe.Pointer, n int, s unsafe.Pointer) error {
	return checkCUDA(C.launch_pow_scalar((*C.float)(a), C.float(scalar), (*C.float)(c), C.int(n), stream(s)), "pow_scalar")
}

// Exp launches the elementwise exp kernel: c = exp(a).
func Exp(a, c unsafe.Pointer, n int, s unsafe.Pointer) error {
	return checkCUDA(C.launch_exp((*C.float)(a), (*C.float)(c), C.int(n), stream(s)), "exp")
}

// Log launches the elementwise log kernel: c = log(a).
func Log(a, c unsafe.Pointer, n int, s unsafe.Pointer) error {
	return checkCUDA(C.launch_log((*C.float)(a), (*C.float)(c), C.int(n), stream(s)), "log")
}

// Sqrt launches the elementwise sqrt kernel: c = sqrt(a).
func Sqrt(a, c unsafe.Pointer, n int, s unsafe.Pointer) error {
	return checkCUDA(C.launch_sqrt((*C.float)(a), (*C.float)(c), C.int(n), stream(s)), "sqrt")
}

// Rsqrt launches the elementwise rsqrt kernel: c = 1/sqrt(a).
func Rsqrt(a, c unsafe.Pointer, n int, s unsafe.Pointer) error {
	return checkCUDA(C.launch_rsqrt((*C.float)(a), (*C.float)(c), C.int(n), stream(s)), "rsqrt")
}

// Tanh launches the elementwise tanh kernel: c = tanh(a).
func Tanh(a, c unsafe.Pointer, n int, s unsafe.Pointer) error {
	return checkCUDA(C.launch_tanh((*C.float)(a), (*C.float)(c), C.int(n), stream(s)), "tanh")
}

// TanhPrime launches the tanh derivative kernel: c = (1 - tanh(a)^2) * upstream.
func TanhPrime(a, upstream, c unsafe.Pointer, n int, s unsafe.Pointer) error {
	return checkCUDA(C.launch_tanh_prime((*C.float)(a), (*C.float)(upstream), (*C.float)(c), C.int(n), stream(s)), "tanh_prime")
}

// Fill launches the fill kernel: sets all elements to value.
func Fill(data unsafe.Pointer, value float32, n int, s unsafe.Pointer) error {
	return checkCUDA(C.launch_fill((*C.float)(data), C.float(value), C.int(n), stream(s)), "fill")
}

// SumAxis launches the sum-reduction kernel along an axis defined by outer/inner/axisSize strides.
// Output has outer*inner elements, one sum per (outer, inner) stripe.
func SumAxis(input, output unsafe.Pointer, outer, inner, axisSize int, s unsafe.Pointer) error {
	return checkCUDA(C.launch_sum_axis((*C.float)(input), (*C.float)(output), C.int(outer), C.int(inner), C.int(axisSize), stream(s)), "sum_axis")
}

// Softmax launches the softmax kernel along an axis defined by outer/inner/axisSize strides.
func Softmax(input, output unsafe.Pointer, outer, inner, axisSize int, s unsafe.Pointer) error {
	return checkCUDA(C.launch_softmax((*C.float)(input), (*C.float)(output), C.int(outer), C.int(inner), C.int(axisSize), stream(s)), "softmax")
}

// AddBroadcast launches the broadcast add kernel: c[r,c] = a[r*sa_r + c*sa_c] + b[r*sb_r + c*sb_c].
func AddBroadcast(a, b, c unsafe.Pointer, saRow, saCol, sbRow, sbCol, M, D int, s unsafe.Pointer) error {
	return checkCUDA(C.launch_add_broadcast((*C.float)(a), (*C.float)(b), (*C.float)(c),
		C.int(saRow), C.int(saCol), C.int(sbRow), C.int(sbCol), C.int(M), C.int(D), stream(s)), "add_broadcast")
}

// SubBroadcast launches the broadcast sub kernel.
func SubBroadcast(a, b, c unsafe.Pointer, saRow, saCol, sbRow, sbCol, M, D int, s unsafe.Pointer) error {
	return checkCUDA(C.launch_sub_broadcast((*C.float)(a), (*C.float)(b), (*C.float)(c),
		C.int(saRow), C.int(saCol), C.int(sbRow), C.int(sbCol), C.int(M), C.int(D), stream(s)), "sub_broadcast")
}

// MulBroadcast launches the broadcast mul kernel.
func MulBroadcast(a, b, c unsafe.Pointer, saRow, saCol, sbRow, sbCol, M, D int, s unsafe.Pointer) error {
	return checkCUDA(C.launch_mul_broadcast((*C.float)(a), (*C.float)(b), (*C.float)(c),
		C.int(saRow), C.int(saCol), C.int(sbRow), C.int(sbCol), C.int(M), C.int(D), stream(s)), "mul_broadcast")
}

// DivBroadcast launches the broadcast div kernel.
func DivBroadcast(a, b, c unsafe.Pointer, saRow, saCol, sbRow, sbCol, M, D int, s unsafe.Pointer) error {
	return checkCUDA(C.launch_div_broadcast((*C.float)(a), (*C.float)(b), (*C.float)(c),
		C.int(saRow), C.int(saCol), C.int(sbRow), C.int(sbCol), C.int(M), C.int(D), stream(s)), "div_broadcast")
}
