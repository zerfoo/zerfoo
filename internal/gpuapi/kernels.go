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
	// A_q4 is packed Q4_0 blocks, B is [K,N] float32, C is [M,N] float32.
	GemmQ4F32(aQ4, b, c unsafe.Pointer, m, k, n int, stream Stream) error

	// Broadcast binary ops: c[r,c] = op(a[r*saRow+c*saCol], b[r*sbRow+c*sbCol]).
	// Strides encode broadcasting: D for full row, 1 for full col, 0 for broadcast.
	AddBroadcast(a, b, c unsafe.Pointer, saRow, saCol, sbRow, sbCol, M, D int, stream Stream) error
	SubBroadcast(a, b, c unsafe.Pointer, saRow, saCol, sbRow, sbCol, M, D int, stream Stream) error
	MulBroadcast(a, b, c unsafe.Pointer, saRow, saCol, sbRow, sbCol, M, D int, stream Stream) error
	DivBroadcast(a, b, c unsafe.Pointer, saRow, saCol, sbRow, sbCol, M, D int, stream Stream) error

	// Transpose2D transposes a [rows, cols] matrix to [cols, rows] using tiled shared memory.
	Transpose2D(input, output unsafe.Pointer, rows, cols int, stream Stream) error

	// TransposeND permutes dimensions of an N-D tensor.
	// inStrides/outShape/perm are int32 slices on host.
	TransposeND(input, output unsafe.Pointer, inStrides, outShape, perm []int32, ndim, total int, stream Stream) error

	// Gather performs embedding table lookup: output[i,:] = table[indices[i],:].
	// table: [V, D], indices: [N] int32 on device, output: [N, D].
	Gather(table, indices, output unsafe.Pointer, N, D, V int, stream Stream) error

	// RMSNorm computes fused RMSNorm: output = input * rsqrt(mean(input^2) + eps) * weight.
	// input: [rows, D], weight: [D], output: [rows, D], scales: [rows] (per-row rsqrt values for backward).
	RMSNorm(input, weight, output, scales unsafe.Pointer, eps float32, rows, D int, stream Stream) error
}
