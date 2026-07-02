//go:build opencl

package clblast

/*
#cgo LDFLAGS: -lclblast -lOpenCL

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include <clblast_c.h>
*/
import "C"

import (
	"fmt"
	"unsafe"
)

// Handle wraps a CLBlast context for BLAS operations.
// CLBlast does not use explicit handles; it operates on OpenCL command queues.
type Handle struct {
	queue   C.cl_command_queue
	context C.cl_context
}

// NewHandle creates a CLBlast handle from an OpenCL command queue and context.
func NewHandle(queue, context unsafe.Pointer) *Handle {
	return &Handle{
		queue:   C.cl_command_queue(queue),
		context: C.cl_context(context),
	}
}

// Destroy releases the handle (no-op for CLBlast).
func (h *Handle) Destroy() error {
	return nil
}

// SetStream updates the command queue used for BLAS operations.
func (h *Handle) SetStream(queue unsafe.Pointer) error {
	h.queue = C.cl_command_queue(queue)
	return nil
}

// Sgemm performs single-precision general matrix multiplication.
// C = alpha * A * B + beta * C
// A is [m x k], B is [k x n], C is [m x n] in row-major order.
//
// CLBlast expects column-major layout. We use the standard trick:
// swap A and B and transpose the result dimensions to compute
// row-major matmul with column-major BLAS.
func (h *Handle) Sgemm(m, n, k int, alpha float32, a, b unsafe.Pointer, beta float32, c unsafe.Pointer) error {
	// Row-major -> column-major trick:
	// C_row = A_row * B_row is equivalent to:
	// C_col^T = B_col^T * A_col^T
	// So we call Sgemm with (n, m, k) and swap A <-> B.
	cAlpha := C.float(alpha)
	cBeta := C.float(beta)

	// a and b are cl_mem handles, not raw pointers.
	// CLBlast expects cl_mem buffers with offset = 0.
	status := C.CLBlastSgemm(
		C.CLBlastLayoutColMajor,
		C.CLBlastTransposeNo, C.CLBlastTransposeNo,
		C.size_t(n), C.size_t(m), C.size_t(k),
		cAlpha,
		C.cl_mem(b), 0, C.size_t(n),
		C.cl_mem(a), 0, C.size_t(k),
		cBeta,
		C.cl_mem(c), 0, C.size_t(n),
		&h.queue, nil,
	)
	if status != C.CLBlastSuccess {
		return fmt.Errorf("CLBlastSgemm: error %d", status)
	}

	// Wait for completion.
	C.clFinish(h.queue)
	return nil
}
