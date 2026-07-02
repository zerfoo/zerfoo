package gpuapi

import "unsafe"

// BLAS abstracts GPU-accelerated Basic Linear Algebra Subprograms.
// Each vendor (cuBLAS, rocBLAS, CLBlast) provides an implementation.
type BLAS interface {
	// Sgemm performs single-precision general matrix multiplication:
	//   C = alpha * A * B + beta * C
	// where A is m x k, B is k x n, and C is m x n.
	// All matrices are contiguous row-major. The implementation handles
	// the row-major to column-major conversion internally.
	Sgemm(m, n, k int, alpha float32,
		a unsafe.Pointer, b unsafe.Pointer,
		beta float32, c unsafe.Pointer,
	) error

	// BFloat16Gemm performs BFloat16 general matrix multiplication:
	//   C = alpha * A * B + beta * C
	// where A is m x k, B is k x n, and C is m x n.
	// All matrices are contiguous row-major BFloat16 elements.
	// Computation is performed in float32 for precision (CUBLAS_COMPUTE_32F).
	// Returns an error on backends that do not support BFloat16 GEMM.
	BFloat16Gemm(m, n, k int, alpha float32,
		a unsafe.Pointer, b unsafe.Pointer,
		beta float32, c unsafe.Pointer,
	) error

	// Float16Gemm performs FP16 general matrix multiplication:
	//   C = alpha * A * B + beta * C
	// where A is m x k, B is k x n, and C is m x n.
	// All matrices are contiguous row-major FP16 elements.
	// Computation is performed in float32 for precision (CUBLAS_COMPUTE_32F).
	Float16Gemm(m, n, k int, alpha float32,
		a unsafe.Pointer, b unsafe.Pointer,
		beta float32, c unsafe.Pointer,
	) error

	// MixedFP16Gemm performs mixed-precision GEMM with FP16 inputs and FP32 output:
	//   C_f32 = alpha * A_fp16 * B_fp16 + beta * C_f32
	MixedFP16Gemm(m, n, k int, alpha float32,
		a unsafe.Pointer, b unsafe.Pointer,
		beta float32, c unsafe.Pointer,
	) error

	// MixedBF16Gemm performs mixed-precision GEMM with BF16 weights and FP32 output:
	//   C_f32 = alpha * A_bf16 * B_bf16 + beta * C_f32
	// where A is m x k, B is k x n (both BFloat16), and C is m x n (float32).
	// Computation uses CUBLAS_COMPUTE_32F for precision.
	// Returns an error on backends that do not support mixed-precision GEMM.
	MixedBF16Gemm(m, n, k int, alpha float32,
		a unsafe.Pointer, b unsafe.Pointer,
		beta float32, c unsafe.Pointer,
	) error

	// SetStream associates the BLAS handle with an asynchronous stream.
	SetStream(stream Stream) error

	// Destroy releases the BLAS handle resources.
	Destroy() error
}

// BLASTransposeB is an optional extension that supports computing
// C = alpha * A * B^T + beta * C without explicitly transposing B.
// A is m x k (row-major), B is n x k (row-major), C is m x n.
type BLASTransposeB interface {
	SgemmNT(m, n, k int, alpha float32,
		a unsafe.Pointer, b unsafe.Pointer,
		beta float32, c unsafe.Pointer,
	) error
}

// BLASBatched is an optional extension that supports strided batched GEMM.
// All batch elements share the same m, n, k dimensions and alpha/beta scalars.
// Matrices are accessed at base + i*stride for batch element i.
type BLASBatched interface {
	SgemmStridedBatched(m, n, k int, alpha float32,
		a unsafe.Pointer, strideA int64,
		b unsafe.Pointer, strideB int64,
		beta float32,
		c unsafe.Pointer, strideC int64,
		batch int,
	) error
}

// BLASBatchedTransposeB is an optional extension that supports strided batched
// C = A * B^T without explicitly transposing B. A is [m, k], B is [n, k] per batch.
type BLASBatchedTransposeB interface {
	SgemmNTStridedBatched(m, n, k int, alpha float32,
		a unsafe.Pointer, strideA int64,
		b unsafe.Pointer, strideB int64,
		beta float32,
		c unsafe.Pointer, strideC int64,
		batch int,
	) error
}
