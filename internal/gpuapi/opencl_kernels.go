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

func (k *OpenCLKernels) Sin(a, c unsafe.Pointer, n int, _ Stream) error {
	return fmt.Errorf("sin kernel: not implemented for OpenCL")
}

func (k *OpenCLKernels) Cos(a, c unsafe.Pointer, n int, _ Stream) error {
	return fmt.Errorf("cos kernel: not implemented for OpenCL")
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

func (k *OpenCLKernels) GemmQ4F32(aQ4, b, c unsafe.Pointer, m, kk, n, dataOffset int, _ Stream) error {
	return fmt.Errorf("GemmQ4F32: not implemented for OpenCL")
}

func (k *OpenCLKernels) GemvQ4KF32(wQ4K, x, y unsafe.Pointer, M, K int, _ Stream) error {
	return fmt.Errorf("GemvQ4KF32: not implemented for OpenCL")
}

func (k *OpenCLKernels) DequantQ4KF32(src, dst unsafe.Pointer, rows, K int, _ Stream) error {
	return fmt.Errorf("DequantQ4KF32: not implemented for OpenCL")
}

func (k *OpenCLKernels) GemmQ8F32(aQ8, b, c unsafe.Pointer, m, kk, n int, _ Stream) error {
	return fmt.Errorf("GemmQ8F32: not implemented for OpenCL")
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

func (k *OpenCLKernels) AddBroadcast4D(_, _, _ unsafe.Pointer, _, _, _, _, _, _, _, _, _, _, _, _ int, _ Stream) error {
	return fmt.Errorf("AddBroadcast4D: not implemented for OpenCL")
}

func (k *OpenCLKernels) SubBroadcast4D(_, _, _ unsafe.Pointer, _, _, _, _, _, _, _, _, _, _, _, _ int, _ Stream) error {
	return fmt.Errorf("SubBroadcast4D: not implemented for OpenCL")
}

func (k *OpenCLKernels) MulBroadcast4D(_, _, _ unsafe.Pointer, _, _, _, _, _, _, _, _, _, _, _, _ int, _ Stream) error {
	return fmt.Errorf("MulBroadcast4D: not implemented for OpenCL")
}

func (k *OpenCLKernels) DivBroadcast4D(_, _, _ unsafe.Pointer, _, _, _, _, _, _, _, _, _, _, _, _ int, _ Stream) error {
	return fmt.Errorf("DivBroadcast4D: not implemented for OpenCL")
}

func (k *OpenCLKernels) Transpose2D(input, output unsafe.Pointer, rows, cols int, _ Stream) error {
	return fmt.Errorf("Transpose2D: not implemented for OpenCL")
}

func (k *OpenCLKernels) TransposeND(input, output unsafe.Pointer, inStrides, outStrides, perm []int32, ndim, total int, _ Stream) error {
	return fmt.Errorf("TransposeND: not implemented for OpenCL")
}

func (k *OpenCLKernels) Gather(table, indices, output unsafe.Pointer, N, D, V int, _ Stream) error {
	return fmt.Errorf("Gather: not implemented for OpenCL")
}

func (k *OpenCLKernels) RMSNorm(input, weight, output, scales unsafe.Pointer, eps float32, rows, D int, _ Stream) error {
	return fmt.Errorf("RMSNorm: not implemented for OpenCL")
}

func (k *OpenCLKernels) Repeat(_ unsafe.Pointer, _ unsafe.Pointer, _, _, _, _ int, _ Stream) error {
	return fmt.Errorf("Repeat: not implemented for OpenCL")
}

func (k *OpenCLKernels) Argmax(_, _, _ unsafe.Pointer, _ int, _ Stream) error {
	return fmt.Errorf("Argmax: not implemented for OpenCL")
}

func (k *OpenCLKernels) FusedRoPEF32(_, _, _, _ unsafe.Pointer, _, _, _, _, _ int, _ Stream) error {
	return fmt.Errorf("FusedRoPEF32: not implemented for OpenCL")
}

func (k *OpenCLKernels) FusedSwiGLUF32(_, _, _ unsafe.Pointer, _ int, _ Stream) error {
	return fmt.Errorf("FusedSwiGLUF32: not implemented for OpenCL")
}

func (k *OpenCLKernels) FusedAddRMSNormF32(_, _, _, _, _ unsafe.Pointer, _ float32, _, _ int, _ Stream) error {
	return fmt.Errorf("FusedAddRMSNormF32: not implemented for OpenCL")
}

func (k *OpenCLKernels) FusedNormAddF32(_, _, _, _ unsafe.Pointer, _ float32, _, _ int, _ Stream) error {
	return fmt.Errorf("FusedNormAddF32: not implemented for OpenCL")
}

func (k *OpenCLKernels) FusedQKNormRoPEF32(_, _, _, _, _, _ unsafe.Pointer, _ float32, _, _, _, _ int, _ Stream) error {
	return fmt.Errorf("FusedQKNormRoPEF32: not implemented for OpenCL")
}

func (k *OpenCLKernels) ScaledSoftmaxF32(_, _ unsafe.Pointer, _, _, _ int, _ float32, _ Stream) error {
	return fmt.Errorf("ScaledSoftmaxF32: not implemented for OpenCL")
}

func (k *OpenCLKernels) AddFP16(_, _, _ unsafe.Pointer, _ int, _ Stream) error {
	return fmt.Errorf("AddFP16: not implemented for OpenCL")
}

func (k *OpenCLKernels) SubFP16(_, _, _ unsafe.Pointer, _ int, _ Stream) error {
	return fmt.Errorf("SubFP16: not implemented for OpenCL")
}

func (k *OpenCLKernels) MulFP16(_, _, _ unsafe.Pointer, _ int, _ Stream) error {
	return fmt.Errorf("MulFP16: not implemented for OpenCL")
}

func (k *OpenCLKernels) DivFP16(_, _, _ unsafe.Pointer, _ int, _ Stream) error {
	return fmt.Errorf("DivFP16: not implemented for OpenCL")
}

func (k *OpenCLKernels) RMSNormFP16(_, _, _ unsafe.Pointer, _ float32, _, _ int, _ Stream) error {
	return fmt.Errorf("RMSNormFP16: not implemented for OpenCL")
}

func (k *OpenCLKernels) ScaledSoftmaxFP16(_, _ unsafe.Pointer, _, _, _ int, _ float32, _ Stream) error {
	return fmt.Errorf("ScaledSoftmaxFP16: not implemented for OpenCL")
}

func (k *OpenCLKernels) F32ToFP16(_, _ unsafe.Pointer, _ int, _ Stream) error {
	return fmt.Errorf("F32ToFP16: not implemented for OpenCL")
}

func (k *OpenCLKernels) FP16ToF32(_, _ unsafe.Pointer, _ int, _ Stream) error {
	return fmt.Errorf("FP16ToF32: not implemented for OpenCL")
}

func (k *OpenCLKernels) DequantFP8E4M3ToFP16(_, _ unsafe.Pointer, _ float32, _ int, _ Stream) error {
	return fmt.Errorf("DequantFP8E4M3ToFP16: not implemented for OpenCL")
}

func (k *OpenCLKernels) IncrementCounter(_ unsafe.Pointer, _ int, _ Stream) error {
	return fmt.Errorf("IncrementCounter: not implemented for OpenCL")
}

func (k *OpenCLKernels) ResetCounter(_ unsafe.Pointer, _ int, _ Stream) error {
	return fmt.Errorf("ResetCounter: not implemented for OpenCL")
}

func (k *OpenCLKernels) OffsetMemcpy(_, _, _ unsafe.Pointer, _, _ int, _ Stream) error {
	return fmt.Errorf("OffsetMemcpy: not implemented for OpenCL")
}

func (k *OpenCLKernels) OffsetMemcpyFP16(_, _, _ unsafe.Pointer, _, _ int, _ Stream) error {
	return fmt.Errorf("OffsetMemcpyFP16: not implemented for OpenCL")
}

func (k *OpenCLKernels) RoPESelect(_, _, _, _, _ unsafe.Pointer, _ int, _ Stream) error {
	return fmt.Errorf("RoPESelect: not implemented for OpenCL")
}

func (k *OpenCLKernels) SgemvM1(_, _, _ unsafe.Pointer, _, _ int, _ Stream) error {
	return fmt.Errorf("SgemvM1: not implemented for OpenCL")
}

// Compile-time interface assertion.
var _ KernelRunner = (*OpenCLKernels)(nil)
