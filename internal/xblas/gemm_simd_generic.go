//go:build !arm64 && !amd64

package xblas

import "unsafe"

// SgemmSimd falls back to a naive triple-loop GEMM on unsupported architectures.
func SgemmSimd(m, n, k int, a, b, c []float32) {
	for i := range m {
		for j := range n {
			var sum float32
			for p := range k {
				sum += a[i*k+p] * b[p*n+j]
			}
			c[i*n+j] = sum
		}
	}
}

// sgemmAccRow computes c[j] += aVal * b[j] for j=0..n-1 (scalar fallback).
func sgemmAccRow(cPtr, bPtr unsafe.Pointer, aVal float32, n int) {
	c := unsafe.Slice((*float32)(cPtr), n)
	b := unsafe.Slice((*float32)(bPtr), n)
	for j := range n {
		c[j] += aVal * b[j]
	}
}
