package xblas

import (
	"math"
	"testing"

	"github.com/zerfoo/ztensor/tensor"
)

type quantGemmTestCase struct {
	name    string
	m, n, k int
	maxErr  float32
}

func makeTestInputs(m, k, n int) ([]float32, []float32) {
	aF32 := make([]float32, m*k)
	for i := range aF32 {
		aF32[i] = float32(i%7-3) * 0.1
	}
	b := make([]float32, k*n)
	for i := range b {
		b[i] = float32(i%5-2) * 0.1
	}
	return aF32, b
}

func assertClose(t *testing.T, got, want []float32, maxErr float32) {
	t.Helper()
	for i := range got {
		diff := float32(math.Abs(float64(got[i] - want[i])))
		if diff > maxErr {
			t.Errorf("index %d: got %v, want %v (diff=%v)", i, got[i], want[i], diff)
		}
	}
}

var gemmSizes = []quantGemmTestCase{
	{"1x1x32", 1, 1, 32, 0},
	{"2x2x32", 2, 2, 32, 0},
	{"4x4x64", 4, 4, 64, 0},
	{"8x8x128", 8, 8, 128, 0},
}

func TestGemmQ4F32_Correctness(t *testing.T) {
	for _, tt := range gemmSizes {
		tt.maxErr = 0.15
		t.Run(tt.name, func(t *testing.T) {
			aF32, b := makeTestInputs(tt.m, tt.k, tt.n)
			aQ4 := tensor.QuantizeQ4(aF32)

			got := make([]float32, tt.m*tt.n)
			GemmQ4F32(tt.m, tt.n, tt.k, aQ4, b, got)

			// Reference: dequantize then float32 GEMM.
			af32Full := make([]float32, tt.m*tt.k)
			aQ4.Dequantize(af32Full)
			want := make([]float32, tt.m*tt.n)
			GemmF32(tt.m, tt.n, tt.k, af32Full, b, want)

			assertClose(t, got, want, tt.maxErr)
		})
	}
}

func TestGemmQ8F32_Correctness(t *testing.T) {
	for _, tt := range gemmSizes {
		tt.maxErr = 0.02
		t.Run(tt.name, func(t *testing.T) {
			aF32, b := makeTestInputs(tt.m, tt.k, tt.n)
			aQ8 := tensor.QuantizeQ8(aF32)

			got := make([]float32, tt.m*tt.n)
			GemmQ8F32(tt.m, tt.n, tt.k, aQ8, b, got)

			af32Full := make([]float32, tt.m*tt.k)
			aQ8.Dequantize(af32Full)
			want := make([]float32, tt.m*tt.n)
			GemmF32(tt.m, tt.n, tt.k, af32Full, b, want)

			assertClose(t, got, want, tt.maxErr)
		})
	}
}

func TestGemmQ4F32Fused_Correctness(t *testing.T) {
	for _, tt := range gemmSizes {
		tt.maxErr = 0.15
		t.Run(tt.name, func(t *testing.T) {
			aF32, b := makeTestInputs(tt.m, tt.k, tt.n)
			aQ4 := tensor.QuantizeQ4(aF32)

			got := make([]float32, tt.m*tt.n)
			GemmQ4F32Fused(tt.m, tt.n, tt.k, aQ4, b, got)

			// Reference: dequantize then float32 GEMM.
			af32Full := make([]float32, tt.m*tt.k)
			aQ4.Dequantize(af32Full)
			want := make([]float32, tt.m*tt.n)
			GemmF32(tt.m, tt.n, tt.k, af32Full, b, want)

			assertClose(t, got, want, tt.maxErr)
		})
	}
}

func TestGemmQ4F32Fused_GEMV(t *testing.T) {
	// M=1 is the critical decode path (memory-bound).
	m, n, k := 1, 64, 256
	aF32, b := makeTestInputs(m, k, n)
	aQ4 := tensor.QuantizeQ4(aF32)

	got := make([]float32, m*n)
	GemmQ4F32Fused(m, n, k, aQ4, b, got)

	af32Full := make([]float32, m*k)
	aQ4.Dequantize(af32Full)
	want := make([]float32, m*n)
	GemmF32(m, n, k, af32Full, b, want)

	assertClose(t, got, want, 0.15)
}

func TestGemmQ4F32Fused_LargeMatrix(t *testing.T) {
	m, n, k := 32, 64, 256
	aF32, b := makeTestInputs(m, k, n)
	aQ4 := tensor.QuantizeQ4(aF32)

	got := make([]float32, m*n)
	GemmQ4F32Fused(m, n, k, aQ4, b, got)

	af32Full := make([]float32, m*k)
	aQ4.Dequantize(af32Full)
	want := make([]float32, m*n)
	GemmF32(m, n, k, af32Full, b, want)

	assertClose(t, got, want, 0.15)
}

func BenchmarkGemmQ4F32(b *testing.B) {
	for _, size := range []int{512, 1024} {
		b.Run(benchLabel(size), func(b *testing.B) {
			aF32, bf32 := makeTestInputs(size, size, size)
			aQ4 := tensor.QuantizeQ4(aF32)
			c := make([]float32, size*size)

			b.ResetTimer()
			for range b.N {
				GemmQ4F32(size, size, size, aQ4, bf32, c)
			}
		})
	}
}

func BenchmarkGemmQ4F32Fused(b *testing.B) {
	for _, size := range []int{512, 1024} {
		b.Run(benchLabel(size), func(b *testing.B) {
			aF32, bf32 := makeTestInputs(size, size, size)
			aQ4 := tensor.QuantizeQ4(aF32)
			c := make([]float32, size*size)

			b.ResetTimer()
			for range b.N {
				GemmQ4F32Fused(size, size, size, aQ4, bf32, c)
			}
		})
	}
}

func BenchmarkGemmQ4F32_GEMV(b *testing.B) {
	// Decode-path: M=1, large K and N.
	k, n := 4096, 4096
	aF32, bf32 := makeTestInputs(1, k, n)
	aQ4 := tensor.QuantizeQ4(aF32)
	c := make([]float32, n)

	b.Run("dequant+sgemm", func(b *testing.B) {
		for range b.N {
			GemmQ4F32(1, n, k, aQ4, bf32, c)
		}
	})
	b.Run("fused", func(b *testing.B) {
		for range b.N {
			GemmQ4F32Fused(1, n, k, aQ4, bf32, c)
		}
	})
}

func BenchmarkGemmQ8F32(b *testing.B) {
	for _, size := range []int{512, 1024} {
		b.Run(benchLabel(size), func(b *testing.B) {
			aF32, bf32 := makeTestInputs(size, size, size)
			aQ8 := tensor.QuantizeQ8(aF32)
			c := make([]float32, size*size)

			b.ResetTimer()
			for range b.N {
				GemmQ8F32(size, size, size, aQ8, bf32, c)
			}
		})
	}
}

// --- GemmF32Q4NT tests: C = A * B^T where B is [N,K] in Q4 format ---

func TestGemmF32Q4NT_Correctness(t *testing.T) {
	cases := []struct {
		name    string
		m, n, k int
	}{
		{"1x1x32", 1, 1, 32},
		{"1x4x32", 1, 4, 32},
		{"1x4x64", 1, 4, 64},
		{"2x4x32", 2, 4, 32},
		{"4x8x64", 4, 8, 64},
		{"1x64x256", 1, 64, 256},
	}

	for _, tt := range cases {
		t.Run(tt.name, func(t *testing.T) {
			// A is [M, K], B_orig is [N, K] (to be quantized as Q4).
			a := make([]float32, tt.m*tt.k)
			for i := range a {
				a[i] = float32(i%7-3) * 0.1
			}
			bOrig := make([]float32, tt.n*tt.k)
			for i := range bOrig {
				bOrig[i] = float32(i%5-2) * 0.1
			}

			bQ4 := tensor.QuantizeQ4(bOrig)

			got := make([]float32, tt.m*tt.n)
			GemmF32Q4NT(tt.m, tt.n, tt.k, a, bQ4, got)

			// Reference: dequantize B, transpose, then float32 GEMM.
			bDeq := make([]float32, tt.n*tt.k)
			bQ4.Dequantize(bDeq)
			// Transpose bDeq from [N,K] to [K,N].
			bT := make([]float32, tt.k*tt.n)
			for r := range tt.n {
				for c := range tt.k {
					bT[c*tt.n+r] = bDeq[r*tt.k+c]
				}
			}
			want := make([]float32, tt.m*tt.n)
			GemmF32(tt.m, tt.n, tt.k, a, bT, want)

			assertClose(t, got, want, 0.15)
		})
	}
}

func TestGemmF32Q4NT_GEMV(t *testing.T) {
	// M=1 is the critical decode path.
	m, n, k := 1, 128, 256
	a := make([]float32, m*k)
	for i := range a {
		a[i] = float32(i%7-3) * 0.1
	}
	bOrig := make([]float32, n*k)
	for i := range bOrig {
		bOrig[i] = float32(i%5-2) * 0.1
	}
	bQ4 := tensor.QuantizeQ4(bOrig)

	got := make([]float32, m*n)
	GemmF32Q4NT(m, n, k, a, bQ4, got)

	bDeq := make([]float32, n*k)
	bQ4.Dequantize(bDeq)
	bT := make([]float32, k*n)
	for r := range n {
		for c := range k {
			bT[c*n+r] = bDeq[r*k+c]
		}
	}
	want := make([]float32, m*n)
	GemmF32(m, n, k, a, bT, want)

	assertClose(t, got, want, 0.15)
}

func BenchmarkGemmF32Q4NT_GEMV(b *testing.B) {
	// Simulates decode-path dimensions: M=1, large K and N.
	sizes := []struct {
		name string
		n, k int
	}{
		{"N1152_K1152", 1152, 1152},
		{"N6912_K1152", 6912, 1152},
		{"N1152_K6912", 1152, 6912},
	}
	for _, sz := range sizes {
		aF32 := make([]float32, sz.k)
		for i := range aF32 {
			aF32[i] = float32(i%7-3) * 0.1
		}
		bOrig := make([]float32, sz.n*sz.k)
		for i := range bOrig {
			bOrig[i] = float32(i%5-2) * 0.1
		}
		bQ4 := tensor.QuantizeQ4(bOrig)
		c := make([]float32, sz.n)

		b.Run(sz.name+"/q4nt", func(b *testing.B) {
			for range b.N {
				GemmF32Q4NT(1, sz.n, sz.k, aF32, bQ4, c)
			}
		})

		// Compare: dequant + transpose + SGEMM (current path).
		bDeq := make([]float32, sz.n*sz.k)
		bQ4.Dequantize(bDeq)
		bT := make([]float32, sz.k*sz.n)
		for r := range sz.n {
			for c := range sz.k {
				bT[c*sz.n+r] = bDeq[r*sz.k+c]
			}
		}
		b.Run(sz.name+"/dequant_sgemm", func(b *testing.B) {
			for range b.N {
				GemmF32(1, sz.n, sz.k, aF32, bT, c)
			}
		})
	}
}

func benchLabel(n int) string {
	switch {
	case n >= 1024:
		return "1k"
	default:
		return "512"
	}
}
