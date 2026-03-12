//go:build !cuda

package gpuapi

import (
	"unsafe"

	"github.com/zerfoo/zerfoo/internal/cublas"
)

// CUDABlas implements the BLAS interface using cuBLAS via purego.
type CUDABlas struct {
	handle *cublas.Handle
}

// NewCUDABlas creates a new cuBLAS adapter.
// The caller must call Destroy when done.
func NewCUDABlas() (*CUDABlas, error) {
	h, err := cublas.CreateHandle()
	if err != nil {
		return nil, err
	}
	return &CUDABlas{handle: h}, nil
}

// NewCUDABlasFromHandle wraps an existing cuBLAS handle.
// The caller retains ownership; Destroy on this adapter is a no-op.
func NewCUDABlasFromHandle(h *cublas.Handle) *CUDABlas {
	return &CUDABlas{handle: h}
}

func (b *CUDABlas) Sgemm(m, n, k int, alpha float32,
	a unsafe.Pointer, bPtr unsafe.Pointer,
	beta float32, c unsafe.Pointer,
) error {
	return cublas.Sgemm(b.handle, m, n, k, alpha, a, bPtr, beta, c)
}

func (b *CUDABlas) BFloat16Gemm(m, n, k int, alpha float32,
	a unsafe.Pointer, bPtr unsafe.Pointer,
	beta float32, c unsafe.Pointer,
) error {
	return cublas.GemmEx(b.handle, m, n, k, alpha,
		a, cublas.CudaR16BF,
		bPtr, cublas.CudaR16BF,
		beta,
		c, cublas.CudaR16BF,
		cublas.CublasCompute32F,
	)
}

// SgemmNT performs C = alpha * A * B^T + beta * C where A is [m, k] and
// B is [n, k] (row-major). This avoids an explicit Transpose of B.
func (b *CUDABlas) SgemmNT(m, n, k int, alpha float32,
	a unsafe.Pointer, bPtr unsafe.Pointer,
	beta float32, c unsafe.Pointer,
) error {
	return cublas.SgemmNT(b.handle, m, n, k, alpha, a, bPtr, beta, c)
}

func (b *CUDABlas) SetStream(stream Stream) error {
	var ptr unsafe.Pointer
	if stream != nil {
		ptr = stream.Ptr()
	}
	return b.handle.SetStream(ptr)
}

func (b *CUDABlas) Destroy() error {
	return b.handle.Destroy()
}

// Handle returns the underlying cuBLAS handle for backward compatibility.
func (b *CUDABlas) Handle() *cublas.Handle {
	return b.handle
}

func init() {
	BLASFactory = func() (BLAS, error) { return NewCUDABlas() }
}

// Compile-time interface assertion.
var _ BLAS = (*CUDABlas)(nil)
