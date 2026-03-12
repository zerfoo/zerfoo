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
