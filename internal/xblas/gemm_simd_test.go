package xblas

import (
	"math"
	"testing"
)

func TestSgemmSimd_Identity(t *testing.T) {
	a := []float32{1, 0, 0, 1}
	b := []float32{3, 4, 5, 6}
	c := make([]float32, 4)
	cRef := make([]float32, 4)

	SgemmSimd(2, 2, 2, a, b, c)
	GemmF32(2, 2, 2, a, b, cRef)

	for i := range c {
		if math.Abs(float64(c[i]-cRef[i])) > 1e-5 {
			t.Errorf("c[%d] = %v, want %v", i, c[i], cRef[i])
		}
	}
}

func TestSgemmSimd_KnownProduct(t *testing.T) {
	a := []float32{1, 2, 3, 4}
	b := []float32{5, 6, 7, 8}
	c := make([]float32, 4)
	cRef := make([]float32, 4)

	SgemmSimd(2, 2, 2, a, b, c)
	GemmF32(2, 2, 2, a, b, cRef)

	for i := range c {
		if math.Abs(float64(c[i]-cRef[i])) > 1e-5 {
			t.Errorf("c[%d] = %v, want %v", i, c[i], cRef[i])
		}
	}
}

func TestSgemmSimd_NonSquare(t *testing.T) {
	a := []float32{1, 2, 3}
	b := []float32{4, 5, 6}
	c := make([]float32, 1)
	cRef := make([]float32, 1)

	SgemmSimd(1, 1, 3, a, b, c)
	GemmF32(1, 1, 3, a, b, cRef)

	if math.Abs(float64(c[0]-cRef[0])) > 1e-5 {
		t.Errorf("c[0] = %v, want %v", c[0], cRef[0])
	}
}

func TestSgemmSimd_Zero(t *testing.T) {
	a := []float32{0, 0, 0, 0}
	b := []float32{1, 2, 3, 4}
	c := make([]float32, 4)

	SgemmSimd(2, 2, 2, a, b, c)

	for i, v := range c {
		if v != 0 {
			t.Errorf("c[%d] = %v, want 0", i, v)
		}
	}
}

func TestSgemmSimd_LargeMatrix(t *testing.T) {
	m, n, k := 64, 48, 128

	a := make([]float32, m*k)
	b := make([]float32, k*n)
	for i := range a {
		a[i] = float32(i%7-3) * 0.1
	}
	for i := range b {
		b[i] = float32(i%5-2) * 0.1
	}

	c := make([]float32, m*n)
	cRef := make([]float32, m*n)

	SgemmSimd(m, n, k, a, b, c)
	GemmF32(m, n, k, a, b, cRef)

	for i := range c {
		diff := math.Abs(float64(c[i] - cRef[i]))
		if diff > 1e-4 {
			t.Errorf("index %d: got %v, want %v (diff=%v)", i, c[i], cRef[i], diff)
		}
	}
}

func TestSgemmSimd_OddDimensions(t *testing.T) {
	tests := []struct {
		name    string
		m, n, k int
	}{
		{"1x1x1", 1, 1, 1},
		{"3x5x7", 3, 5, 7},
		{"1x1x33", 1, 1, 33},
		{"7x3x1", 7, 3, 1},
		{"5x5x17", 5, 5, 17},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			a := make([]float32, tt.m*tt.k)
			b := make([]float32, tt.k*tt.n)
			for i := range a {
				a[i] = float32(i%11-5) * 0.1
			}
			for i := range b {
				b[i] = float32(i%9-4) * 0.1
			}

			c := make([]float32, tt.m*tt.n)
			cRef := make([]float32, tt.m*tt.n)

			SgemmSimd(tt.m, tt.n, tt.k, a, b, c)
			GemmF32(tt.m, tt.n, tt.k, a, b, cRef)

			for i := range c {
				diff := math.Abs(float64(c[i] - cRef[i]))
				if diff > 1e-4 {
					t.Errorf("index %d: got %v, want %v (diff=%v)", i, c[i], cRef[i], diff)
				}
			}
		})
	}
}

func BenchmarkSgemm_1024(b *testing.B) {
	m, n, k := 1024, 1024, 1024
	a := make([]float32, m*k)
	bm := make([]float32, k*n)
	for i := range a {
		a[i] = float32(i%7-3) * 0.01
	}
	for i := range bm {
		bm[i] = float32(i%5-2) * 0.01
	}
	c := make([]float32, m*n)

	b.Run("gonum", func(b *testing.B) {
		for range b.N {
			GemmF32(m, n, k, a, bm, c)
		}
	})
	b.Run("simd", func(b *testing.B) {
		for range b.N {
			SgemmSimd(m, n, k, a, bm, c)
		}
	})
}

func BenchmarkSgemm_512(b *testing.B) {
	m, n, k := 512, 512, 512
	a := make([]float32, m*k)
	bm := make([]float32, k*n)
	for i := range a {
		a[i] = float32(i%7-3) * 0.01
	}
	for i := range bm {
		bm[i] = float32(i%5-2) * 0.01
	}
	c := make([]float32, m*n)

	b.Run("gonum", func(b *testing.B) {
		for range b.N {
			GemmF32(m, n, k, a, bm, c)
		}
	})
	b.Run("simd", func(b *testing.B) {
		for range b.N {
			SgemmSimd(m, n, k, a, bm, c)
		}
	})
}
