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

// CudaDataType identifies the element data type for cublasGemmEx.
type CudaDataType int

const (
	CudaR32F  CudaDataType = 0  // CUDA_R_32F  (float32)
	CudaR16F  CudaDataType = 2  // CUDA_R_16F  (float16)
	CudaR16BF CudaDataType = 14 // CUDA_R_16BF (bfloat16)
)

// CublasComputeType identifies the compute precision for cublasGemmEx.
type CublasComputeType int

const (
	CublasCompute32F CublasComputeType = 68 // CUBLAS_COMPUTE_32F
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

// SgemmNT performs single-precision C = A * B^T where A is [m, k] and
// B is [n, k] (row-major). Uses CUBLAS_OP_T on the first cuBLAS argument
// (our B) to transpose B without an explicit copy.
//
// Row-major to column-major conversion:
//
//	cuBLAS sees B_rm[n, k] as B_cm[k, n]. With CUBLAS_OP_T, op(B_cm) = B_cm^T = B_rm[n, k].
//	cuBLAS sees A_rm[m, k] as A_cm[k, m]. With CUBLAS_OP_N, op(A_cm) = A_cm[k, m].
//	op(B_cm) * op(A_cm) = [n, k] * [k, m] = [n, m] in column-major.
//	Reading as row-major: [m, n] = A_rm * B_rm^T. ✓
func SgemmNT(h *Handle, m, n, k int, alpha float32,
	a unsafe.Pointer, b unsafe.Pointer,
	beta float32, c unsafe.Pointer,
) error {
	cAlpha := C.float(alpha)
	cBeta := C.float(beta)

	status := C.cublasSgemm(
		h.h,
		C.CUBLAS_OP_T, // transpose B (cuBLAS first arg)
		C.CUBLAS_OP_N, // no-transpose A (cuBLAS second arg)
		C.int(n),      // rows of op(B) = n
		C.int(m),      // cols of op(A) = m
		C.int(k),      // inner dimension
		(*C.float)(unsafe.Pointer(&cAlpha)),
		(*C.float)(b), // B comes first (cuBLAS convention)
		C.int(k),      // leading dim of B_cm = k (B_rm row width)
		(*C.float)(a), // A comes second
		C.int(k),      // leading dim of A_cm = k (A_rm row width)
		(*C.float)(unsafe.Pointer(&cBeta)),
		(*C.float)(c),
		C.int(n), // leading dim of C_cm = n (C_rm row width)
	)

	if status != C.CUBLAS_STATUS_SUCCESS {
		return fmt.Errorf("cublasSgemm(NT) failed with status %d", int(status))
	}

	return nil
}

// SgemmStridedBatched performs batched single-precision GEMM with strided access.
// Row-major to column-major conversion: swap A/B and m/n (same trick as Sgemm).
func SgemmStridedBatched(h *Handle, m, n, k int, alpha float32,
	a unsafe.Pointer, strideA int64,
	b unsafe.Pointer, strideB int64,
	beta float32,
	c unsafe.Pointer, strideC int64,
	batch int,
) error {
	cAlpha := C.float(alpha)
	cBeta := C.float(beta)

	// Row-major to column-major: swap A<->B, swap m<->n, swap strides.
	status := C.cublasSgemmStridedBatched(
		h.h,
		C.CUBLAS_OP_N,
		C.CUBLAS_OP_N,
		C.int(n),
		C.int(m),
		C.int(k),
		(*C.float)(unsafe.Pointer(&cAlpha)),
		(*C.float)(b), C.int(n), C.longlong(strideB),
		(*C.float)(a), C.int(k), C.longlong(strideA),
		(*C.float)(unsafe.Pointer(&cBeta)),
		(*C.float)(c), C.int(n), C.longlong(strideC),
		C.int(batch),
	)

	if status != C.CUBLAS_STATUS_SUCCESS {
		return fmt.Errorf("cublasSgemmStridedBatched failed with status %d", int(status))
	}

	return nil
}

// SgemmNTStridedBatched performs batched C = A * B^T using strided batched GEMM
// with CUBLAS_OP_T on the B operand. A is [m, k], B is [n, k] per batch element.
func SgemmNTStridedBatched(h *Handle, m, n, k int, alpha float32,
	a unsafe.Pointer, strideA int64,
	b unsafe.Pointer, strideB int64,
	beta float32,
	c unsafe.Pointer, strideC int64,
	batch int,
) error {
	cAlpha := C.float(alpha)
	cBeta := C.float(beta)

	// Row-major to column-major: B with OP_T first, A with OP_N second.
	status := C.cublasSgemmStridedBatched(
		h.h,
		C.CUBLAS_OP_T,
		C.CUBLAS_OP_N,
		C.int(n),
		C.int(m),
		C.int(k),
		(*C.float)(unsafe.Pointer(&cAlpha)),
		(*C.float)(b), C.int(k), C.longlong(strideB),
		(*C.float)(a), C.int(k), C.longlong(strideA),
		(*C.float)(unsafe.Pointer(&cBeta)),
		(*C.float)(c), C.int(n), C.longlong(strideC),
		C.int(batch),
	)

	if status != C.CUBLAS_STATUS_SUCCESS {
		return fmt.Errorf("cublasSgemmNTStridedBatched failed with status %d", int(status))
	}

	return nil
}

// GemmEx performs mixed-precision general matrix multiplication using
// cublasGemmEx. Supports BFloat16, Float16, and Float32 element types.
//
// Row-major to column-major conversion uses the same swap-A-B trick as Sgemm.
//
// Parameters (in row-major terms):
//
//	m           - rows of A and C
//	n           - columns of B and C
//	k           - columns of A / rows of B
//	alpha       - scalar multiplier for A*B
//	a           - device pointer to A (m x k, row-major)
//	aType       - element data type of A
//	b           - device pointer to B (k x n, row-major)
//	bType       - element data type of B
//	beta        - scalar multiplier for C
//	c           - device pointer to C (m x n, row-major), output
//	cType       - element data type of C
//	computeType - compute precision (e.g. CublasCompute32F)
func GemmEx(h *Handle, m, n, k int, alpha float32,
	a unsafe.Pointer, aType CudaDataType,
	b unsafe.Pointer, bType CudaDataType,
	beta float32,
	c unsafe.Pointer, cType CudaDataType,
	computeType CublasComputeType,
) error {
	cAlpha := C.float(alpha)
	cBeta := C.float(beta)

	// Row-major to column-major: swap A and B, swap m and n (same trick as Sgemm).
	status := C.cublasGemmEx(
		h.h,
		C.CUBLAS_OP_N,
		C.CUBLAS_OP_N,
		C.int(n),
		C.int(m),
		C.int(k),
		unsafe.Pointer(&cAlpha),
		b, C.cudaDataType(bType), C.int(n),
		a, C.cudaDataType(aType), C.int(k),
		unsafe.Pointer(&cBeta),
		c, C.cudaDataType(cType), C.int(n),
		C.cublasComputeType_t(computeType),
		C.CUBLAS_GEMM_DEFAULT,
	)

	if status != C.CUBLAS_STATUS_SUCCESS {
		return fmt.Errorf("cublasGemmEx failed with status %d", int(status))
	}

	return nil
}
