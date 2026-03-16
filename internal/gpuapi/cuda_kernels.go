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

func (k *CUDAKernels) Sin(a, c unsafe.Pointer, n int, s Stream) error {
	return kernels.Sin(a, c, n, streamPtr(s))
}

func (k *CUDAKernels) Cos(a, c unsafe.Pointer, n int, s Stream) error {
	return kernels.Cos(a, c, n, streamPtr(s))
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

func (k *CUDAKernels) GemmQ4F32(aQ4, b, c unsafe.Pointer, m, kk, n, dataOffset int, s Stream) error {
	return kernels.GemmQ4F32(aQ4, b, c, m, kk, n, dataOffset, streamPtr(s))
}

func (k *CUDAKernels) GemvQ4KF32(wQ4K, x, y unsafe.Pointer, M, K int, s Stream) error {
	return kernels.GemvQ4KF32(wQ4K, x, y, M, K, streamPtr(s))
}

func (k *CUDAKernels) DequantQ4KF32(src, dst unsafe.Pointer, rows, K int, s Stream) error {
	return kernels.DequantQ4KF32(src, dst, rows, K, streamPtr(s))
}

func (k *CUDAKernels) GemmQ8F32(aQ8, b, c unsafe.Pointer, m, kk, n int, s Stream) error {
	return kernels.GemmQ8F32(aQ8, b, c, m, kk, n, streamPtr(s))
}

func (k *CUDAKernels) AddBroadcast(a, b, c unsafe.Pointer, saRow, saCol, sbRow, sbCol, M, D int, s Stream) error {
	return kernels.AddBroadcast(a, b, c, saRow, saCol, sbRow, sbCol, M, D, streamPtr(s))
}

func (k *CUDAKernels) SubBroadcast(a, b, c unsafe.Pointer, saRow, saCol, sbRow, sbCol, M, D int, s Stream) error {
	return kernels.SubBroadcast(a, b, c, saRow, saCol, sbRow, sbCol, M, D, streamPtr(s))
}

func (k *CUDAKernels) MulBroadcast(a, b, c unsafe.Pointer, saRow, saCol, sbRow, sbCol, M, D int, s Stream) error {
	return kernels.MulBroadcast(a, b, c, saRow, saCol, sbRow, sbCol, M, D, streamPtr(s))
}

func (k *CUDAKernels) DivBroadcast(a, b, c unsafe.Pointer, saRow, saCol, sbRow, sbCol, M, D int, s Stream) error {
	return kernels.DivBroadcast(a, b, c, saRow, saCol, sbRow, sbCol, M, D, streamPtr(s))
}

func (k *CUDAKernels) AddBroadcast4D(a, b, c unsafe.Pointer, d0, d1, d2, d3, sa0, sa1, sa2, sa3, sb0, sb1, sb2, sb3 int, s Stream) error {
	return kernels.AddBroadcast4D(a, b, c, d0, d1, d2, d3, sa0, sa1, sa2, sa3, sb0, sb1, sb2, sb3, streamPtr(s))
}

func (k *CUDAKernels) SubBroadcast4D(a, b, c unsafe.Pointer, d0, d1, d2, d3, sa0, sa1, sa2, sa3, sb0, sb1, sb2, sb3 int, s Stream) error {
	return kernels.SubBroadcast4D(a, b, c, d0, d1, d2, d3, sa0, sa1, sa2, sa3, sb0, sb1, sb2, sb3, streamPtr(s))
}

func (k *CUDAKernels) MulBroadcast4D(a, b, c unsafe.Pointer, d0, d1, d2, d3, sa0, sa1, sa2, sa3, sb0, sb1, sb2, sb3 int, s Stream) error {
	return kernels.MulBroadcast4D(a, b, c, d0, d1, d2, d3, sa0, sa1, sa2, sa3, sb0, sb1, sb2, sb3, streamPtr(s))
}

func (k *CUDAKernels) DivBroadcast4D(a, b, c unsafe.Pointer, d0, d1, d2, d3, sa0, sa1, sa2, sa3, sb0, sb1, sb2, sb3 int, s Stream) error {
	return kernels.DivBroadcast4D(a, b, c, d0, d1, d2, d3, sa0, sa1, sa2, sa3, sb0, sb1, sb2, sb3, streamPtr(s))
}

func (k *CUDAKernels) Transpose2D(input, output unsafe.Pointer, rows, cols int, s Stream) error {
	return kernels.Transpose2D(input, output, rows, cols, streamPtr(s))
}

func (k *CUDAKernels) TransposeND(input, output unsafe.Pointer, inStrides, outStrides, perm []int32, ndim, total int, s Stream) error {
	return kernels.TransposeND(input, output, inStrides, outStrides, perm, ndim, total, streamPtr(s))
}

func (k *CUDAKernels) Gather(table, indices, output unsafe.Pointer, N, D, V int, s Stream) error {
	return kernels.Gather(table, indices, output, N, D, V, streamPtr(s))
}

func (k *CUDAKernels) RMSNorm(input, weight, output, scales unsafe.Pointer, eps float32, rows, D int, s Stream) error {
	return kernels.RMSNorm(input, weight, output, scales, eps, rows, D, streamPtr(s))
}

func (k *CUDAKernels) Repeat(src, dst unsafe.Pointer, outerSize, axisDim, innerSize, reps int, s Stream) error {
	return kernels.Repeat(src, dst, outerSize, axisDim, innerSize, reps, streamPtr(s))
}

func (k *CUDAKernels) Argmax(input, result, scratch unsafe.Pointer, n int, s Stream) error {
	return kernels.Argmax(input, result, scratch, n, streamPtr(s))
}

func (k *CUDAKernels) FusedRoPEF32(input, cosAngles, sinAngles, output unsafe.Pointer, batch, seqLen, headDim, halfRotary, cosStride int, s Stream) error {
	return kernels.FusedRoPEF32(input, cosAngles, sinAngles, output, batch, seqLen, headDim, halfRotary, cosStride, streamPtr(s))
}

func (k *CUDAKernels) FusedSwiGLUF32(w1, w3, output unsafe.Pointer, n int, s Stream) error {
	return kernels.FusedSwiGLUF32(w1, w3, output, n, streamPtr(s))
}

func (k *CUDAKernels) FusedAddRMSNormF32(input, residual, weight, normedOut, sumOut unsafe.Pointer, eps float32, rows, D int, s Stream) error {
	return kernels.FusedAddRMSNormF32(input, residual, weight, normedOut, sumOut, eps, rows, D, streamPtr(s))
}

func (k *CUDAKernels) FusedNormAddF32(input, weight, residual, output unsafe.Pointer, eps float32, rows, D int, s Stream) error {
	return kernels.FusedNormAddF32(input, weight, residual, output, eps, rows, D, streamPtr(s))
}

func (k *CUDAKernels) FusedQKNormRoPEF32(input, weightQ, weightK, cosAngles, sinAngles, output unsafe.Pointer, eps float32, totalHeads, headDim, numQHeads, halfRotary int, s Stream) error {
	return kernels.FusedQKNormRoPEF32(input, weightQ, weightK, cosAngles, sinAngles, output, eps, totalHeads, headDim, numQHeads, halfRotary, streamPtr(s))
}

func (k *CUDAKernels) ScaledSoftmaxF32(input, output unsafe.Pointer, outer, inner, axisSize int, scale float32, s Stream) error {
	return kernels.ScaledSoftmaxF32(input, output, outer, inner, axisSize, scale, streamPtr(s))
}

func (k *CUDAKernels) AddFP16(a, b, c unsafe.Pointer, n int, s Stream) error {
	return kernels.AddFP16(a, b, c, n, streamPtr(s))
}

func (k *CUDAKernels) SubFP16(a, b, c unsafe.Pointer, n int, s Stream) error {
	return kernels.SubFP16(a, b, c, n, streamPtr(s))
}

func (k *CUDAKernels) MulFP16(a, b, c unsafe.Pointer, n int, s Stream) error {
	return kernels.MulFP16(a, b, c, n, streamPtr(s))
}

func (k *CUDAKernels) DivFP16(a, b, c unsafe.Pointer, n int, s Stream) error {
	return kernels.DivFP16(a, b, c, n, streamPtr(s))
}

func (k *CUDAKernels) RMSNormFP16(input, weight, output unsafe.Pointer, eps float32, rows, D int, s Stream) error {
	return kernels.RMSNormFP16(input, weight, output, eps, rows, D, streamPtr(s))
}

func (k *CUDAKernels) ScaledSoftmaxFP16(input, output unsafe.Pointer, outer, inner, axisSize int, scale float32, s Stream) error {
	return kernels.ScaledSoftmaxFP16(input, output, outer, inner, axisSize, scale, streamPtr(s))
}

func (k *CUDAKernels) F32ToFP16(src, dst unsafe.Pointer, n int, s Stream) error {
	return kernels.F32ToFP16(src, dst, n, streamPtr(s))
}

func (k *CUDAKernels) FP16ToF32(src, dst unsafe.Pointer, n int, s Stream) error {
	return kernels.FP16ToF32(src, dst, n, streamPtr(s))
}

func (k *CUDAKernels) DequantFP8E4M3ToFP16(input, output unsafe.Pointer, scale float32, n int, s Stream) error {
	return kernels.DequantFP8E4M3ToFP16(input, output, scale, n, streamPtr(s))
}

func (k *CUDAKernels) IncrementCounter(counter unsafe.Pointer, delta int, s Stream) error {
	return kernels.IncrementCounter(counter, delta, streamPtr(s))
}

func (k *CUDAKernels) ResetCounter(counter unsafe.Pointer, value int, s Stream) error {
	return kernels.ResetCounter(counter, value, streamPtr(s))
}

func (k *CUDAKernels) OffsetMemcpy(dst, src, counter unsafe.Pointer, dim, maxSeqLen int, s Stream) error {
	return kernels.OffsetMemcpy(dst, src, counter, dim, maxSeqLen, streamPtr(s))
}

func (k *CUDAKernels) OffsetMemcpyFP16(dst, src, counter unsafe.Pointer, dim, maxSeqLen int, s Stream) error {
	return kernels.OffsetMemcpyFP16(dst, src, counter, dim, maxSeqLen, streamPtr(s))
}

func (k *CUDAKernels) RoPESelect(cosTable, sinTable, cosOut, sinOut, counter unsafe.Pointer,
	halfRotary int, s Stream) error {
	return kernels.RoPESelect(cosTable, sinTable, cosOut, sinOut, counter, halfRotary, streamPtr(s))
}

func (k *CUDAKernels) SgemvM1(y, A, x unsafe.Pointer, M, N int, s Stream) error {
	return kernels.SgemvM1(y, A, x, M, N, streamPtr(s))
}

// Compile-time interface assertion.
var _ KernelRunner = (*CUDAKernels)(nil)
