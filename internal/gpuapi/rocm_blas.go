package gpuapi

import (
	"fmt"
	"unsafe"

	"github.com/zerfoo/zerfoo/internal/rocblas"
)

// ROCmBlas implements the BLAS interface using rocBLAS.
type ROCmBlas struct {
	handle *rocblas.Handle
}

// NewROCmBlas creates a new rocBLAS adapter.
// The caller must call Destroy when done.
func NewROCmBlas() (*ROCmBlas, error) {
	h, err := rocblas.CreateHandle()
	if err != nil {
		return nil, err
	}
	return &ROCmBlas{handle: h}, nil
}

// NewROCmBlasFromHandle wraps an existing rocBLAS handle.
func NewROCmBlasFromHandle(h *rocblas.Handle) *ROCmBlas {
	return &ROCmBlas{handle: h}
}

func (b *ROCmBlas) Sgemm(m, n, k int, alpha float32,
	a unsafe.Pointer, bPtr unsafe.Pointer,
	beta float32, c unsafe.Pointer,
) error {
	return rocblas.Sgemm(b.handle, m, n, k, alpha, a, bPtr, beta, c)
}

func (b *ROCmBlas) BFloat16Gemm(_, _, _ int, _ float32,
	_, _ unsafe.Pointer, _ float32, _ unsafe.Pointer,
) error {
	return fmt.Errorf("BFloat16Gemm: not supported on ROCm backend")
}

func (b *ROCmBlas) Float16Gemm(_, _, _ int, _ float32,
	_, _ unsafe.Pointer, _ float32, _ unsafe.Pointer,
) error {
	return fmt.Errorf("Float16Gemm: not supported on ROCm backend")
}

func (b *ROCmBlas) MixedFP16Gemm(_, _, _ int, _ float32,
	_, _ unsafe.Pointer, _ float32, _ unsafe.Pointer,
) error {
	return fmt.Errorf("MixedFP16Gemm: not supported on ROCm backend")
}

func (b *ROCmBlas) MixedBF16Gemm(_, _, _ int, _ float32,
	_, _ unsafe.Pointer, _ float32, _ unsafe.Pointer,
) error {
	return fmt.Errorf("MixedBF16Gemm: not supported on ROCm backend")
}

func (b *ROCmBlas) SetStream(stream Stream) error {
	var ptr unsafe.Pointer
	if stream != nil {
		ptr = stream.Ptr()
	}
	return b.handle.SetStream(ptr)
}

func (b *ROCmBlas) Destroy() error {
	return b.handle.Destroy()
}

// Handle returns the underlying rocBLAS handle.
func (b *ROCmBlas) Handle() *rocblas.Handle {
	return b.handle
}

// Compile-time interface assertion.
var _ BLAS = (*ROCmBlas)(nil)
