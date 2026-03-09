//go:build rocm

package gpuapi

import (
	"fmt"
	"unsafe"

	"github.com/zerfoo/zerfoo/internal/hip/kernels"
)

// ROCmKernels implements the KernelRunner interface using custom HIP kernels.
type ROCmKernels struct{}

// NewROCmKernels returns a new ROCm kernel runner adapter.
func NewROCmKernels() *ROCmKernels {
	return &ROCmKernels{}
}

func rocmStreamPtr(s Stream) unsafe.Pointer {
	if s == nil {
		return nil
	}
	return s.Ptr()
}

func (k *ROCmKernels) Add(a, b, c unsafe.Pointer, n int, s Stream) error {
	return kernels.Add(a, b, c, n, rocmStreamPtr(s))
}

func (k *ROCmKernels) Sub(a, b, c unsafe.Pointer, n int, s Stream) error {
	return kernels.Sub(a, b, c, n, rocmStreamPtr(s))
}

func (k *ROCmKernels) Mul(a, b, c unsafe.Pointer, n int, s Stream) error {
	return kernels.Mul(a, b, c, n, rocmStreamPtr(s))
}

func (k *ROCmKernels) Div(a, b, c unsafe.Pointer, n int, s Stream) error {
	return kernels.Div(a, b, c, n, rocmStreamPtr(s))
}

func (k *ROCmKernels) Pow(base, exp, c unsafe.Pointer, n int, s Stream) error {
	return kernels.Pow(base, exp, c, n, rocmStreamPtr(s))
}

func (k *ROCmKernels) Exp(a, c unsafe.Pointer, n int, s Stream) error {
	return kernels.Exp(a, c, n, rocmStreamPtr(s))
}

func (k *ROCmKernels) Log(a, c unsafe.Pointer, n int, s Stream) error {
	return kernels.Log(a, c, n, rocmStreamPtr(s))
}

func (k *ROCmKernels) Sqrt(a, c unsafe.Pointer, n int, s Stream) error {
	return kernels.Sqrt(a, c, n, rocmStreamPtr(s))
}

func (k *ROCmKernels) Rsqrt(a, c unsafe.Pointer, n int, s Stream) error {
	return kernels.Rsqrt(a, c, n, rocmStreamPtr(s))
}

func (k *ROCmKernels) Tanh(a, c unsafe.Pointer, n int, s Stream) error {
	return kernels.Tanh(a, c, n, rocmStreamPtr(s))
}

func (k *ROCmKernels) TanhPrime(a, upstream, c unsafe.Pointer, n int, s Stream) error {
	return kernels.TanhPrime(a, upstream, c, n, rocmStreamPtr(s))
}

func (k *ROCmKernels) AddScalar(a unsafe.Pointer, scalar float32, c unsafe.Pointer, n int, s Stream) error {
	return kernels.AddScalar(a, scalar, c, n, rocmStreamPtr(s))
}

func (k *ROCmKernels) MulScalar(a unsafe.Pointer, scalar float32, c unsafe.Pointer, n int, s Stream) error {
	return kernels.MulScalar(a, scalar, c, n, rocmStreamPtr(s))
}

func (k *ROCmKernels) DivScalar(a unsafe.Pointer, scalar float32, c unsafe.Pointer, n int, s Stream) error {
	return kernels.DivScalar(a, scalar, c, n, rocmStreamPtr(s))
}

func (k *ROCmKernels) SubScalar(_ unsafe.Pointer, _ float32, _ unsafe.Pointer, _ int, _ Stream) error {
	return fmt.Errorf("SubScalar: not implemented for ROCm")
}

func (k *ROCmKernels) PowScalar(_ unsafe.Pointer, _ float32, _ unsafe.Pointer, _ int, _ Stream) error {
	return fmt.Errorf("PowScalar: not implemented for ROCm")
}

func (k *ROCmKernels) Fill(data unsafe.Pointer, value float32, n int, s Stream) error {
	return kernels.Fill(data, value, n, rocmStreamPtr(s))
}

func (k *ROCmKernels) SumAxis(input, output unsafe.Pointer, outer, inner, axisSize int, s Stream) error {
	return kernels.SumAxis(input, output, outer, inner, axisSize, rocmStreamPtr(s))
}

func (k *ROCmKernels) Softmax(input, output unsafe.Pointer, outer, inner, axisSize int, s Stream) error {
	return kernels.Softmax(input, output, outer, inner, axisSize, rocmStreamPtr(s))
}

func (k *ROCmKernels) GemmQ4F32(aQ4, b, c unsafe.Pointer, m, kk, n int, s Stream) error {
	return fmt.Errorf("GemmQ4F32: not implemented for ROCm")
}

func (k *ROCmKernels) AddBroadcast(a, b, c unsafe.Pointer, saRow, saCol, sbRow, sbCol, M, D int, _ Stream) error {
	return fmt.Errorf("AddBroadcast: not implemented for ROCm")
}

func (k *ROCmKernels) SubBroadcast(a, b, c unsafe.Pointer, saRow, saCol, sbRow, sbCol, M, D int, _ Stream) error {
	return fmt.Errorf("SubBroadcast: not implemented for ROCm")
}

func (k *ROCmKernels) MulBroadcast(a, b, c unsafe.Pointer, saRow, saCol, sbRow, sbCol, M, D int, _ Stream) error {
	return fmt.Errorf("MulBroadcast: not implemented for ROCm")
}

func (k *ROCmKernels) DivBroadcast(a, b, c unsafe.Pointer, saRow, saCol, sbRow, sbCol, M, D int, _ Stream) error {
	return fmt.Errorf("DivBroadcast: not implemented for ROCm")
}

func (k *ROCmKernels) Transpose2D(input, output unsafe.Pointer, rows, cols int, _ Stream) error {
	return fmt.Errorf("Transpose2D: not implemented for ROCm")
}

func (k *ROCmKernels) TransposeND(input, output unsafe.Pointer, inStrides, outShape, perm []int32, ndim, total int, _ Stream) error {
	return fmt.Errorf("TransposeND: not implemented for ROCm")
}

func (k *ROCmKernels) Gather(table, indices, output unsafe.Pointer, N, D, V int, _ Stream) error {
	return fmt.Errorf("Gather: not implemented for ROCm")
}

func (k *ROCmKernels) RMSNorm(input, weight, output, scales unsafe.Pointer, eps float32, rows, D int, _ Stream) error {
	return fmt.Errorf("RMSNorm: not implemented for ROCm")
}

// Compile-time interface assertion.
var _ KernelRunner = (*ROCmKernels)(nil)
