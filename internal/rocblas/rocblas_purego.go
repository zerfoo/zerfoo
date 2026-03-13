package rocblas

import (
	"fmt"
	"unsafe"

	"github.com/zerfoo/zerfoo/internal/cuda"
)

// rocblasStatusSuccess is the rocBLAS status code for success.
const rocblasStatusSuccess = 0

// rocblasOperationNone indicates no transpose.
const rocblasOperationNone = 111

func lib() *RocBLASLib {
	l := Lib()
	if l == nil {
		return nil
	}
	return l
}

// Handle wraps a rocBLAS handle.
type Handle struct {
	handle uintptr
}

// CreateHandle creates a new rocBLAS context handle.
func CreateHandle() (*Handle, error) {
	l := lib()
	if l == nil {
		return nil, fmt.Errorf("rocblas_create_handle failed: rocblas not available")
	}
	var h uintptr
	ret := cuda.Ccall(l.rocblasCreateHandle, uintptr(unsafe.Pointer(&h)))
	if ret != rocblasStatusSuccess {
		return nil, fmt.Errorf("rocblas_create_handle failed with status %d", int(ret))
	}
	return &Handle{handle: h}, nil
}

// Destroy releases the rocBLAS handle resources.
func (h *Handle) Destroy() error {
	l := lib()
	if l == nil {
		return fmt.Errorf("rocblas_destroy_handle failed: rocblas not available")
	}
	ret := cuda.Ccall(l.rocblasDestroyHandle, h.handle)
	if ret != rocblasStatusSuccess {
		return fmt.Errorf("rocblas_destroy_handle failed with status %d", int(ret))
	}
	return nil
}

// SetStream associates a HIP stream with this rocBLAS handle.
func (h *Handle) SetStream(streamPtr unsafe.Pointer) error {
	l := lib()
	if l == nil {
		return fmt.Errorf("rocblas_set_stream failed: rocblas not available")
	}
	ret := cuda.Ccall(l.rocblasSetStream, h.handle, uintptr(streamPtr))
	if ret != rocblasStatusSuccess {
		return fmt.Errorf("rocblas_set_stream failed with status %d", int(ret))
	}
	return nil
}

// Sgemm performs single-precision general matrix multiplication.
//
// This function handles the row-major to column-major conversion internally.
// rocBLAS uses column-major order, but Go uses row-major. The trick:
//
//	For row-major C = A * B (m x n = m x k * k x n):
//	Call rocblas_sgemm with B as first arg and A as second, swapping m/n,
//	because in column-major: B^T * A^T = (A * B)^T, and since rocBLAS reads
//	row-major data as the transpose of what it expects, this yields the
//	correct row-major result in C.
func Sgemm(h *Handle, m, n, k int, alpha float32,
	a unsafe.Pointer, b unsafe.Pointer,
	beta float32, c unsafe.Pointer,
) error {
	l := lib()
	if l == nil {
		return fmt.Errorf("rocblas_sgemm failed: rocblas not available")
	}

	// rocblas_sgemm(handle, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
	// Row-major to column-major conversion (same strategy as cuBLAS):
	// rocblas_sgemm(handle, transB, transA, n, m, k, alpha, B, n, A, k, beta, C, n)
	ret := cuda.Ccall(l.rocblasSgemm,
		h.handle,
		uintptr(rocblasOperationNone), // transB = no-transpose
		uintptr(rocblasOperationNone), // transA = no-transpose
		uintptr(n),                    // rows of op(B) and C (col-major)
		uintptr(m),                    // cols of op(A) and C (col-major)
		uintptr(k),                    // inner dimension
		uintptr(unsafe.Pointer(&alpha)),
		uintptr(b),  // B comes first
		uintptr(n),  // leading dimension of B
		uintptr(a),  // A comes second
		uintptr(k),  // leading dimension of A
		uintptr(unsafe.Pointer(&beta)),
		uintptr(c),
		uintptr(n), // leading dimension of C
	)
	if ret != rocblasStatusSuccess {
		return fmt.Errorf("rocblas_sgemm failed with status %d", int(ret))
	}
	return nil
}
