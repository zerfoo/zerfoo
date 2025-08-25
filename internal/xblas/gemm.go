package xblas

import (
	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/gonum/blas/blas32"
	"gonum.org/v1/gonum/blas/blas64"

	float16 "github.com/zerfoo/float16"
	float8 "github.com/zerfoo/float8"
)

// GemmF32 computes C = A * B for row-major contiguous matrices.
// A has shape (m, k), B has shape (k, n), C has shape (m, n).
// Strides are assumed to be k for A and n for B and C.
func GemmF32(m, n, k int, a, b, c []float32) {
	alpha, beta := float32(1), float32(0)
	A := blas32.General{Rows: m, Cols: k, Data: a, Stride: k}
	B := blas32.General{Rows: k, Cols: n, Data: b, Stride: n}
	C := blas32.General{Rows: m, Cols: n, Data: c, Stride: n}
	blas32.Gemm(blas.NoTrans, blas.NoTrans, alpha, A, B, beta, C)
}

// GemmF64 computes C = A * B for row-major contiguous matrices.
func GemmF64(m, n, k int, a, b, c []float64) {
	alpha, beta := float64(1), float64(0)
	A := blas64.General{Rows: m, Cols: k, Data: a, Stride: k}
	B := blas64.General{Rows: k, Cols: n, Data: b, Stride: n}
	C := blas64.General{Rows: m, Cols: n, Data: c, Stride: n}
	blas64.Gemm(blas.NoTrans, blas.NoTrans, alpha, A, B, beta, C)
}

// GemmF16 computes C = A * B for Float16 by converting through float32 SGEMM.
func GemmF16(m, n, k int, a, b, c []float16.Float16) {
	// Convert inputs to float32
	a32 := make([]float32, len(a))
	for i := range a {
		a32[i] = a[i].ToFloat32()
	}
	b32 := make([]float32, len(b))
	for i := range b {
		b32[i] = b[i].ToFloat32()
	}
	c32 := make([]float32, m*n)

	// Compute SGEMM
	GemmF32(m, n, k, a32, b32, c32)

	// Convert result back to Float16 into c
	for i := 0; i < len(c); i++ {
		c[i] = float16.FromFloat32(c32[i])
	}
}

// GemmF8 computes C = A * B for Float8 by converting through float32 SGEMM.
func GemmF8(m, n, k int, a, b, c []float8.Float8) {
	// Convert inputs to float32
	a32 := make([]float32, len(a))
	for i := range a {
		a32[i] = a[i].ToFloat32()
	}
	b32 := make([]float32, len(b))
	for i := range b {
		b32[i] = b[i].ToFloat32()
	}
	c32 := make([]float32, m*n)

	// Compute SGEMM
	GemmF32(m, n, k, a32, b32, c32)

	// Convert result back to Float8 into c
	for i := 0; i < len(c); i++ {
		c[i] = float8.ToFloat8(c32[i])
	}
}
