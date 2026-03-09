package gpuapi

import (
	"unsafe"

	"github.com/zerfoo/zerfoo/internal/cuda/kernels"
)

// CUDAKernels implements the KernelRunner interface using custom CUDA kernels.
type CUDAKernels struct{}

// NewCUDAKernels returns a new CUDA kernel runner adapter.
func NewCUDAKernels() *CUDAKernels {
	return &CUDAKernels{}
}

func streamPtr(s Stream) unsafe.Pointer {
	if s == nil {
		return nil
	}
	return s.Ptr()
}

func (k *CUDAKernels) Add(a, b, c unsafe.Pointer, n int, s Stream) error {
	return kernels.Add(a, b, c, n, streamPtr(s))
}

func (k *CUDAKernels) Sub(a, b, c unsafe.Pointer, n int, s Stream) error {
	return kernels.Sub(a, b, c, n, streamPtr(s))
}

func (k *CUDAKernels) Mul(a, b, c unsafe.Pointer, n int, s Stream) error {
	return kernels.Mul(a, b, c, n, streamPtr(s))
}

func (k *CUDAKernels) Div(a, b, c unsafe.Pointer, n int, s Stream) error {
	return kernels.Div(a, b, c, n, streamPtr(s))
}

func (k *CUDAKernels) Pow(base, exp, c unsafe.Pointer, n int, s Stream) error {
	return kernels.Pow(base, exp, c, n, streamPtr(s))
}

func (k *CUDAKernels) Exp(a, c unsafe.Pointer, n int, s Stream) error {
	return kernels.Exp(a, c, n, streamPtr(s))
}

func (k *CUDAKernels) Log(a, c unsafe.Pointer, n int, s Stream) error {
	return kernels.Log(a, c, n, streamPtr(s))
}

func (k *CUDAKernels) Sqrt(a, c unsafe.Pointer, n int, s Stream) error {
	return kernels.Sqrt(a, c, n, streamPtr(s))
}

func (k *CUDAKernels) Rsqrt(a, c unsafe.Pointer, n int, s Stream) error {
	return kernels.Rsqrt(a, c, n, streamPtr(s))
}

func (k *CUDAKernels) Tanh(a, c unsafe.Pointer, n int, s Stream) error {
	return kernels.Tanh(a, c, n, streamPtr(s))
}

func (k *CUDAKernels) TanhPrime(a, upstream, c unsafe.Pointer, n int, s Stream) error {
	return kernels.TanhPrime(a, upstream, c, n, streamPtr(s))
}

func (k *CUDAKernels) AddScalar(a unsafe.Pointer, scalar float32, c unsafe.Pointer, n int, s Stream) error {
	return kernels.AddScalar(a, scalar, c, n, streamPtr(s))
}

func (k *CUDAKernels) MulScalar(a unsafe.Pointer, scalar float32, c unsafe.Pointer, n int, s Stream) error {
	return kernels.MulScalar(a, scalar, c, n, streamPtr(s))
}

func (k *CUDAKernels) DivScalar(a unsafe.Pointer, scalar float32, c unsafe.Pointer, n int, s Stream) error {
	return kernels.DivScalar(a, scalar, c, n, streamPtr(s))
}

func (k *CUDAKernels) SubScalar(a unsafe.Pointer, scalar float32, c unsafe.Pointer, n int, s Stream) error {
	return kernels.SubScalar(a, scalar, c, n, streamPtr(s))
}

func (k *CUDAKernels) PowScalar(a unsafe.Pointer, scalar float32, c unsafe.Pointer, n int, s Stream) error {
	return kernels.PowScalar(a, scalar, c, n, streamPtr(s))
}

func (k *CUDAKernels) Fill(data unsafe.Pointer, value float32, n int, s Stream) error {
	return kernels.Fill(data, value, n, streamPtr(s))
}

func (k *CUDAKernels) SumAxis(input, output unsafe.Pointer, outer, inner, axisSize int, s Stream) error {
	return kernels.SumAxis(input, output, outer, inner, axisSize, streamPtr(s))
}

func (k *CUDAKernels) Softmax(input, output unsafe.Pointer, outer, inner, axisSize int, s Stream) error {
	return kernels.Softmax(input, output, outer, inner, axisSize, streamPtr(s))
}

func (k *CUDAKernels) GemmQ4F32(aQ4, b, c unsafe.Pointer, m, kk, n int, s Stream) error {
	return kernels.GemmQ4F32(aQ4, b, c, m, kk, n, streamPtr(s))
}

func (k *CUDAKernels) AddBroadcast(a, b, c unsafe.Pointer, saRow, saCol, sbRow, sbCol, M, D int, s Stream) error { //nolint:gocritic // interface match
	return kernels.AddBroadcast(a, b, c, saRow, saCol, sbRow, sbCol, M, D, streamPtr(s))
}

func (k *CUDAKernels) SubBroadcast(a, b, c unsafe.Pointer, saRow, saCol, sbRow, sbCol, M, D int, s Stream) error { //nolint:gocritic // interface match
	return kernels.SubBroadcast(a, b, c, saRow, saCol, sbRow, sbCol, M, D, streamPtr(s))
}

func (k *CUDAKernels) MulBroadcast(a, b, c unsafe.Pointer, saRow, saCol, sbRow, sbCol, M, D int, s Stream) error { //nolint:gocritic // interface match
	return kernels.MulBroadcast(a, b, c, saRow, saCol, sbRow, sbCol, M, D, streamPtr(s))
}

func (k *CUDAKernels) DivBroadcast(a, b, c unsafe.Pointer, saRow, saCol, sbRow, sbCol, M, D int, s Stream) error { //nolint:gocritic // interface match
	return kernels.DivBroadcast(a, b, c, saRow, saCol, sbRow, sbCol, M, D, streamPtr(s))
}

func (k *CUDAKernels) Transpose2D(input, output unsafe.Pointer, rows, cols int, s Stream) error {
	return kernels.Transpose2D(input, output, rows, cols, streamPtr(s))
}

func (k *CUDAKernels) TransposeND(input, output unsafe.Pointer, inStrides, outShape, perm []int32, ndim, total int, s Stream) error {
	return kernels.TransposeND(input, output, inStrides, outShape, perm, ndim, total, streamPtr(s))
}

func (k *CUDAKernels) Gather(table, indices, output unsafe.Pointer, N, D, V int, s Stream) error { //nolint:gocritic // interface match
	return kernels.Gather(table, indices, output, N, D, V, streamPtr(s))
}

func (k *CUDAKernels) RMSNorm(input, weight, output, scales unsafe.Pointer, eps float32, rows, D int, s Stream) error { //nolint:gocritic // interface match
	return kernels.RMSNorm(input, weight, output, scales, eps, rows, D, streamPtr(s))
}

// Compile-time interface assertion.
var _ KernelRunner = (*CUDAKernels)(nil)
