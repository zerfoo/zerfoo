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

func (b *CUDABlas) Float16Gemm(m, n, k int, alpha float32,
	a unsafe.Pointer, bPtr unsafe.Pointer,
	beta float32, c unsafe.Pointer,
) error {
	return cublas.GemmEx(b.handle, m, n, k, alpha,
		a, cublas.CudaR16F,
		bPtr, cublas.CudaR16F,
		beta,
		c, cublas.CudaR16F,
		cublas.CublasCompute32F,
	)
}

func (b *CUDABlas) MixedFP16Gemm(m, n, k int, alpha float32,
	a unsafe.Pointer, bPtr unsafe.Pointer,
	beta float32, c unsafe.Pointer,
) error {
	return cublas.GemmEx(b.handle, m, n, k, alpha,
		a, cublas.CudaR16F,
		bPtr, cublas.CudaR16F,
		beta,
		c, cublas.CudaR32F,
		cublas.CublasCompute32F,
	)
}

func (b *CUDABlas) MixedBF16Gemm(m, n, k int, alpha float32,
	a unsafe.Pointer, bPtr unsafe.Pointer,
	beta float32, c unsafe.Pointer,
) error {
	return cublas.GemmEx(b.handle, m, n, k, alpha,
		a, cublas.CudaR16BF,
		bPtr, cublas.CudaR16BF,
		beta,
		c, cublas.CudaR32F,
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

// SgemmStridedBatched performs batched C = alpha * A * B + beta * C using
// cublasSgemmStridedBatched. All batch elements share the same m, n, k.
func (b *CUDABlas) SgemmStridedBatched(m, n, k int, alpha float32,
	a unsafe.Pointer, strideA int64,
	bPtr unsafe.Pointer, strideB int64,
	beta float32,
	c unsafe.Pointer, strideC int64,
	batch int,
) error {
	return cublas.SgemmStridedBatched(b.handle, m, n, k, alpha, a, strideA, bPtr, strideB, beta, c, strideC, batch)
}

// SgemmNTStridedBatched performs batched C = A * B^T using strided batched GEMM.
func (b *CUDABlas) SgemmNTStridedBatched(m, n, k int, alpha float32,
	a unsafe.Pointer, strideA int64,
	bPtr unsafe.Pointer, strideB int64,
	beta float32,
	c unsafe.Pointer, strideC int64,
	batch int,
) error {
	return cublas.SgemmNTStridedBatched(b.handle, m, n, k, alpha, a, strideA, bPtr, strideB, beta, c, strideC, batch)
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
	if cublas.Available() {
		BLASFactory = func() (BLAS, error) {
			b, err := NewCUDABlas()
			if err != nil {
				return nil, err
			}
			if cublasProfileEnabled {
				return WrapWithProfiler(b), nil
			}
			return b, nil
		}
	}
}

// Compile-time interface assertion.
var _ BLAS = (*CUDABlas)(nil)
