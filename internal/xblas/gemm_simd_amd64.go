//go:build amd64

package xblas

import (
	"unsafe"
)

// sgemmAccRow computes c[j] += aVal * b[j] for j = 0..n-1 using AVX2.
// Implemented in gemm_simd_amd64.s.
//
//go:noescape
func sgemmAccRow(c, b unsafe.Pointer, aVal float32, n int)

const tileK = 256

// SgemmSimd computes C = A*B using AVX2-accelerated operations.
// A is m×k, B is k×n, C is m×n. All row-major.
func SgemmSimd(m, n, k int, a, b, c []float32) {
	if m == 0 || n == 0 || k == 0 {
		return
	}

	// Tile along K to keep B panel in L2 cache.
	for p0 := 0; p0 < k; p0 += tileK {
		p1 := min(p0+tileK, k)

		for i := range m {
			cRow := unsafe.Pointer(&c[i*n])
			for p := p0; p < p1; p++ {
				if aVal := a[i*k+p]; aVal != 0 {
					sgemmAccRow(cRow, unsafe.Pointer(&b[p*n]), aVal, n)
				}
			}
		}
	}
}
