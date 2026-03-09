//go:build opencl

package gpuapi

import (
	"fmt"
	"unsafe"

	"github.com/zerfoo/zerfoo/internal/opencl/kernels"
)

// OpenCLKernels implements the KernelRunner interface using OpenCL kernels.
// Kernels are compiled from .cl source at initialization time.
type OpenCLKernels struct {
	prog *kernels.Program
}

// NewOpenCLKernels compiles the elementwise kernels and returns a runner.
// ctx, dev, and queue are the OpenCL context, device, and command queue pointers.
func NewOpenCLKernels(ctx, dev, queue unsafe.Pointer) (*OpenCLKernels, error) {
	prog, err := kernels.Compile(ctx, dev, queue)
	if err != nil {
		return nil, err
	}
	return &OpenCLKernels{prog: prog}, nil
}

// Destroy releases the compiled OpenCL program.
func (k *OpenCLKernels) Destroy() {
	if k.prog != nil {
		k.prog.Destroy()
	}
}

func (k *OpenCLKernels) Add(a, b, c unsafe.Pointer, n int, _ Stream) error {
	return k.prog.Add(a, b, c, n)
}

func (k *OpenCLKernels) Sub(a, b, c unsafe.Pointer, n int, _ Stream) error {
	return k.prog.Sub(a, b, c, n)
}

func (k *OpenCLKernels) Mul(a, b, c unsafe.Pointer, n int, _ Stream) error {
	return k.prog.Mul(a, b, c, n)
}

func (k *OpenCLKernels) Div(a, b, c unsafe.Pointer, n int, _ Stream) error {
	return k.prog.Div(a, b, c, n)
}

func (k *OpenCLKernels) Pow(base, exp, c unsafe.Pointer, n int, _ Stream) error {
	return k.prog.Pow(base, exp, c, n)
}

func (k *OpenCLKernels) Exp(a, c unsafe.Pointer, n int, _ Stream) error {
	return k.prog.Exp(a, c, n)
}

func (k *OpenCLKernels) Log(a, c unsafe.Pointer, n int, _ Stream) error {
	return k.prog.Log(a, c, n)
}

func (k *OpenCLKernels) Sqrt(a, c unsafe.Pointer, n int, _ Stream) error {
	return k.prog.Sqrt(a, c, n)
}

func (k *OpenCLKernels) Rsqrt(a, c unsafe.Pointer, n int, _ Stream) error {
	return k.prog.Rsqrt(a, c, n)
}

func (k *OpenCLKernels) Tanh(a, c unsafe.Pointer, n int, _ Stream) error {
	return k.prog.Tanh(a, c, n)
}

func (k *OpenCLKernels) TanhPrime(a, upstream, c unsafe.Pointer, n int, _ Stream) error {
	return k.prog.TanhPrime(a, upstream, c, n)
}

func (k *OpenCLKernels) AddScalar(a unsafe.Pointer, scalar float32, c unsafe.Pointer, n int, _ Stream) error {
	return k.prog.AddScalar(a, scalar, c, n)
}

func (k *OpenCLKernels) MulScalar(a unsafe.Pointer, scalar float32, c unsafe.Pointer, n int, _ Stream) error {
	return k.prog.MulScalar(a, scalar, c, n)
}

func (k *OpenCLKernels) DivScalar(a unsafe.Pointer, scalar float32, c unsafe.Pointer, n int, _ Stream) error {
	return k.prog.DivScalar(a, scalar, c, n)
}

func (k *OpenCLKernels) SubScalar(_ unsafe.Pointer, _ float32, _ unsafe.Pointer, _ int, _ Stream) error {
	return fmt.Errorf("SubScalar: not implemented for OpenCL")
}

func (k *OpenCLKernels) PowScalar(_ unsafe.Pointer, _ float32, _ unsafe.Pointer, _ int, _ Stream) error {
	return fmt.Errorf("PowScalar: not implemented for OpenCL")
}

func (k *OpenCLKernels) Fill(data unsafe.Pointer, value float32, n int, _ Stream) error {
	return k.prog.Fill(data, value, n)
}

func (k *OpenCLKernels) SumAxis(input, output unsafe.Pointer, outer, inner, axisSize int, _ Stream) error {
	return k.prog.SumAxis(input, output, outer, inner, axisSize)
}

func (k *OpenCLKernels) Softmax(input, output unsafe.Pointer, outer, inner, axisSize int, _ Stream) error {
	return k.prog.Softmax(input, output, outer, inner, axisSize)
}

func (k *OpenCLKernels) GemmQ4F32(aQ4, b, c unsafe.Pointer, m, kk, n int, _ Stream) error {
	return fmt.Errorf("GemmQ4F32: not implemented for OpenCL")
}

func (k *OpenCLKernels) AddBroadcast(a, b, c unsafe.Pointer, saRow, saCol, sbRow, sbCol, M, D int, _ Stream) error {
	return fmt.Errorf("AddBroadcast: not implemented for OpenCL")
}

func (k *OpenCLKernels) SubBroadcast(a, b, c unsafe.Pointer, saRow, saCol, sbRow, sbCol, M, D int, _ Stream) error {
	return fmt.Errorf("SubBroadcast: not implemented for OpenCL")
}

func (k *OpenCLKernels) MulBroadcast(a, b, c unsafe.Pointer, saRow, saCol, sbRow, sbCol, M, D int, _ Stream) error {
	return fmt.Errorf("MulBroadcast: not implemented for OpenCL")
}

func (k *OpenCLKernels) DivBroadcast(a, b, c unsafe.Pointer, saRow, saCol, sbRow, sbCol, M, D int, _ Stream) error {
	return fmt.Errorf("DivBroadcast: not implemented for OpenCL")
}

func (k *OpenCLKernels) Transpose2D(input, output unsafe.Pointer, rows, cols int, _ Stream) error {
	return fmt.Errorf("Transpose2D: not implemented for OpenCL")
}

func (k *OpenCLKernels) TransposeND(input, output unsafe.Pointer, inStrides, outShape, perm []int32, ndim, total int, _ Stream) error {
	return fmt.Errorf("TransposeND: not implemented for OpenCL")
}

func (k *OpenCLKernels) Gather(table, indices, output unsafe.Pointer, N, D, V int, _ Stream) error {
	return fmt.Errorf("Gather: not implemented for OpenCL")
}

func (k *OpenCLKernels) RMSNorm(input, weight, output, scales unsafe.Pointer, eps float32, rows, D int, _ Stream) error {
	return fmt.Errorf("RMSNorm: not implemented for OpenCL")
}

// Compile-time interface assertion.
var _ KernelRunner = (*OpenCLKernels)(nil)
