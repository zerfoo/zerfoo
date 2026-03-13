package gpuapi

import "unsafe"

// KernelRunner abstracts GPU compute kernels for elementwise, scalar,
// reduction, and utility operations. Each vendor provides an implementation
// using its own kernel compilation toolchain (CUDA .cu, HIP .hip, OpenCL .cl).
type KernelRunner interface {
	// Binary elementwise operations: c[i] = op(a[i], b[i])
	Add(a, b, c unsafe.Pointer, n int, stream Stream) error
	Sub(a, b, c unsafe.Pointer, n int, stream Stream) error
	Mul(a, b, c unsafe.Pointer, n int, stream Stream) error
	Div(a, b, c unsafe.Pointer, n int, stream Stream) error
	Pow(base, exp, c unsafe.Pointer, n int, stream Stream) error

	// Unary elementwise operations: c[i] = op(a[i])
	Exp(a, c unsafe.Pointer, n int, stream Stream) error
	Log(a, c unsafe.Pointer, n int, stream Stream) error
	Sqrt(a, c unsafe.Pointer, n int, stream Stream) error
	Rsqrt(a, c unsafe.Pointer, n int, stream Stream) error
	Tanh(a, c unsafe.Pointer, n int, stream Stream) error

	// TanhPrime: c[i] = (1 - tanh(a[i])^2) * upstream[i]
	TanhPrime(a, upstream, c unsafe.Pointer, n int, stream Stream) error

	// Scalar operations: c[i] = op(a[i], scalar)
	AddScalar(a unsafe.Pointer, scalar float32, c unsafe.Pointer, n int, stream Stream) error
	MulScalar(a unsafe.Pointer, scalar float32, c unsafe.Pointer, n int, stream Stream) error
	DivScalar(a unsafe.Pointer, scalar float32, c unsafe.Pointer, n int, stream Stream) error

	SubScalar(a unsafe.Pointer, scalar float32, c unsafe.Pointer, n int, stream Stream) error
	PowScalar(a unsafe.Pointer, scalar float32, c unsafe.Pointer, n int, stream Stream) error

	// Fill sets all n elements to value.
	Fill(data unsafe.Pointer, value float32, n int, stream Stream) error

	// SumAxis reduces along one axis: output[outer][inner] = sum(input[outer][k][inner], k=0..axisSize-1).
	SumAxis(input, output unsafe.Pointer, outer, inner, axisSize int, stream Stream) error

	// Softmax computes softmax along one axis.
	Softmax(input, output unsafe.Pointer, outer, inner, axisSize int, stream Stream) error

	// GemmQ4F32 performs Q4_0 dequant-GEMM: C = dequant(A_q4) * B.
	// A_q4 is in GPU separated layout (scales then data), B is [K,N] float32, C is [M,N] float32.
	// dataOffset is the byte offset from A_q4 to the packed data region.
	GemmQ4F32(aQ4, b, c unsafe.Pointer, m, k, n, dataOffset int, stream Stream) error

	// GemmQ8F32 performs Q8_0 dequant-GEMM: C = dequant(A_q8) * B.
	// A_q8 is packed Q8_0 blocks (36 bytes per 32 values), B is [K,N] float32, C is [M,N] float32.
	GemmQ8F32(aQ8, b, c unsafe.Pointer, m, k, n int, stream Stream) error

	// Broadcast binary ops: c[r,c] = op(a[r*saRow+c*saCol], b[r*sbRow+c*sbCol]).
	// Strides encode broadcasting: D for full row, 1 for full col, 0 for broadcast.
	AddBroadcast(a, b, c unsafe.Pointer, saRow, saCol, sbRow, sbCol, M, D int, stream Stream) error
	SubBroadcast(a, b, c unsafe.Pointer, saRow, saCol, sbRow, sbCol, M, D int, stream Stream) error
	MulBroadcast(a, b, c unsafe.Pointer, saRow, saCol, sbRow, sbCol, M, D int, stream Stream) error
	DivBroadcast(a, b, c unsafe.Pointer, saRow, saCol, sbRow, sbCol, M, D int, stream Stream) error

	// Transpose2D transposes a [rows, cols] matrix to [cols, rows] using tiled shared memory.
	Transpose2D(input, output unsafe.Pointer, rows, cols int, stream Stream) error

	// TransposeND permutes dimensions of an N-D tensor.
	// inStrides/outStrides/perm are int32 slices on host.
	TransposeND(input, output unsafe.Pointer, inStrides, outStrides, perm []int32, ndim, total int, stream Stream) error

	// Gather performs embedding table lookup: output[i,:] = table[indices[i],:].
	// table: [V, D], indices: [N] int32 on device, output: [N, D].
	Gather(table, indices, output unsafe.Pointer, N, D, V int, stream Stream) error

	// RMSNorm computes fused RMSNorm: output = input * rsqrt(mean(input^2) + eps) * weight.
	// input: [rows, D], weight: [D], output: [rows, D], scales: [rows] (per-row rsqrt values for backward).
	RMSNorm(input, weight, output, scales unsafe.Pointer, eps float32, rows, D int, stream Stream) error

	// Repeat replicates elements along an axis.
	// outerSize = product of dims before axis, axisDim = size of axis,
	// innerSize = product of dims after axis, reps = number of repetitions.
	Repeat(src, dst unsafe.Pointer, outerSize, axisDim, innerSize, reps int, stream Stream) error

	// Argmax finds the index of the maximum element in a float32 array on device.
	// input: [n] float32, result: single int32 on device, scratch: temp storage.
	// scratch must be at least 2*ceil(n/256)*4 bytes.
	Argmax(input, result, scratch unsafe.Pointer, n int, stream Stream) error

	// FusedRoPEF32 applies rotary positional embedding in one kernel launch.
	// input/output: [batch * seqLen * headDim], cos/sin: [seqLen * cosStride].
	FusedRoPEF32(input, cosAngles, sinAngles, output unsafe.Pointer, batch, seqLen, headDim, halfRotary, cosStride int, stream Stream) error

	// FusedSwiGLUF32 applies SwiGLU activation in one kernel launch.
	// output[i] = w1[i] * sigmoid(w1[i]) * w3[i]. All arrays have n elements.
	FusedSwiGLUF32(w1, w3, output unsafe.Pointer, n int, stream Stream) error

	// FusedAddRMSNormF32 fuses residual addition and RMSNorm into one kernel launch.
	// sum_out = input + residual, normed_out = rmsnorm(sum_out, weight, eps).
	// input: [rows, D], residual: [rows, D], weight: [D],
	// normedOut: [rows, D], sumOut: [rows, D].
	FusedAddRMSNormF32(input, residual, weight, normedOut, sumOut unsafe.Pointer, eps float32, rows, D int, stream Stream) error

	// FusedNormAddF32 applies RMSNorm then adds residual in one kernel launch.
	// output = rmsnorm(input, weight, eps) + residual.
	// input: [rows, D], weight: [D], residual: [rows, D], output: [rows, D].
	FusedNormAddF32(input, weight, residual, output unsafe.Pointer, eps float32, rows, D int, stream Stream) error

	// FusedQKNormRoPEF32 applies per-head RMSNorm + RoPE to combined Q+K heads.
	// Replaces 4 kernel launches (Q_norm + K_norm + Q_RoPE + K_RoPE) with 1.
	// input: [totalHeads, headDim], weightQ/weightK: [headDim],
	// cosAngles/sinAngles: [halfRotary], output: [totalHeads, headDim].
	FusedQKNormRoPEF32(input, weightQ, weightK, cosAngles, sinAngles, output unsafe.Pointer, eps float32, totalHeads, headDim, numQHeads, halfRotary int, stream Stream) error

	// ScaledSoftmaxF32 computes softmax(input * scale) in one kernel launch,
	// replacing the MulScalar + Softmax chain (saves 1 kernel launch per call).
	ScaledSoftmaxF32(input, output unsafe.Pointer, outer, inner, axisSize int, scale float32, stream Stream) error
}
