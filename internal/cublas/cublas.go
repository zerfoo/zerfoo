//go:build cuda

package cublas

/*
#cgo LDFLAGS: -lcublas
#include <cublas_v2.h>
*/
import "C"

import (
	"fmt"
	"unsafe"
)

// Handle wraps a cuBLAS handle.
type Handle struct {
	h C.cublasHandle_t
}

// CreateHandle creates a new cuBLAS context handle.
func CreateHandle() (*Handle, error) {
	var h C.cublasHandle_t

	status := C.cublasCreate(&h)
	if status != C.CUBLAS_STATUS_SUCCESS {
		return nil, fmt.Errorf("cublasCreate failed with status %d", int(status))
	}

	return &Handle{h: h}, nil
}

// Destroy releases the cuBLAS handle resources.
func (h *Handle) Destroy() error {
	status := C.cublasDestroy(h.h)
	if status != C.CUBLAS_STATUS_SUCCESS {
		return fmt.Errorf("cublasDestroy failed with status %d", int(status))
	}

	return nil
}

// SetStream associates a CUDA stream with this cuBLAS handle.
// All subsequent cuBLAS operations will execute on the given stream.
// Pass nil to use the default stream.
func (h *Handle) SetStream(streamPtr unsafe.Pointer) error {
	status := C.cublasSetStream(h.h, C.cudaStream_t(streamPtr))
	if status != C.CUBLAS_STATUS_SUCCESS {
		return fmt.Errorf("cublasSetStream failed with status %d", int(status))
	}

	return nil
}

// Sgemm performs single-precision general matrix multiplication.
//
// This function handles the row-major to column-major conversion internally.
// cuBLAS uses column-major order, but Go uses row-major. The trick:
//
//	For row-major C = A * B (m x n = m x k * k x n):
//	Call cublasSgemm with B as first arg and A as second, swapping m/n,
//	because in column-major: B^T * A^T = (A * B)^T, and since cuBLAS reads
//	row-major data as the transpose of what it expects, this yields the
//	correct row-major result in C.
//
// Parameters (in row-major terms):
//
//	m     - rows of A and C
//	n     - columns of B and C
//	k     - columns of A / rows of B
//	alpha - scalar multiplier for A*B
//	a     - device pointer to A (m x k, row-major)
//	b     - device pointer to B (k x n, row-major)
//	beta  - scalar multiplier for C
//	c     - device pointer to C (m x n, row-major), output
func Sgemm(h *Handle, m, n, k int, alpha float32,
	a unsafe.Pointer, b unsafe.Pointer,
	beta float32, c unsafe.Pointer,
) error {
	cAlpha := C.float(alpha)
	cBeta := C.float(beta)

	// Row-major to column-major conversion:
	// cublasSgemm(handle, transB, transA, n, m, k, alpha, B, n, A, k, beta, C, n)
	// Because cuBLAS reads our row-major B (k x n) as column-major (n x k) = B^T,
	// and our row-major A (m x k) as column-major (k x m) = A^T.
	// So it computes B^T * A^T in column-major = (A * B)^T in column-major = A * B in row-major.
	status := C.cublasSgemm(
		h.h,
		C.CUBLAS_OP_N, // transB = no-transpose (but cuBLAS reads row-major as transposed)
		C.CUBLAS_OP_N, // transA = no-transpose
		C.int(n),      // number of rows of op(B) and C (in column-major = cols of C in row-major)
		C.int(m),      // number of columns of op(A) and C (in column-major = rows of C in row-major)
		C.int(k),      // inner dimension
		(*C.float)(unsafe.Pointer(&cAlpha)),
		(*C.float)(b), // B comes first
		C.int(n),      // leading dimension of B (row-major width = n)
		(*C.float)(a), // A comes second
		C.int(k),      // leading dimension of A (row-major width = k)
		(*C.float)(unsafe.Pointer(&cBeta)),
		(*C.float)(c),
		C.int(n), // leading dimension of C (row-major width = n)
	)

	if status != C.CUBLAS_STATUS_SUCCESS {
		return fmt.Errorf("cublasSgemm failed with status %d", int(status))
	}

	return nil
}
