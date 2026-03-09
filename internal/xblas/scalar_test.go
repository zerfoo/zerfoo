package xblas

import (
	"math"
	"math/rand/v2"
	"testing"
)

func TestVmulScalarF32(t *testing.T) {
	a := []float32{1, 2, 3, 4, 5, 6, 7, 8}
	out := make([]float32, 8)
	VmulScalarF32(&out[0], &a[0], 3.0, 8)
	want := []float32{3, 6, 9, 12, 15, 18, 21, 24}
	for i := range out {
		if out[i] != want[i] {
			t.Errorf("VmulScalarF32[%d] = %v, want %v", i, out[i], want[i])
		}
	}
}

func TestVaddScalarF32(t *testing.T) {
	a := []float32{1, 2, 3, 4, 5, 6, 7, 8}
	out := make([]float32, 8)
	VaddScalarF32(&out[0], &a[0], 10.0, 8)
	want := []float32{11, 12, 13, 14, 15, 16, 17, 18}
	for i := range out {
		if out[i] != want[i] {
			t.Errorf("VaddScalarF32[%d] = %v, want %v", i, out[i], want[i])
		}
	}
}

func TestVdivScalarF32(t *testing.T) {
	a := []float32{10, 20, 30, 40, 50, 60, 70, 80}
	out := make([]float32, 8)
	VdivScalarF32(&out[0], &a[0], 10.0, 8)
	want := []float32{1, 2, 3, 4, 5, 6, 7, 8}
	for i := range out {
		if out[i] != want[i] {
			t.Errorf("VdivScalarF32[%d] = %v, want %v", i, out[i], want[i])
		}
	}
}

func TestScalar_TailHandling(t *testing.T) {
	for _, n := range []int{1, 3, 7} {
		a := make([]float32, n)
		out := make([]float32, n)
		for i := range n {
			a[i] = float32(i + 1)
		}

		t.Run("mulscalar_tail_"+string(rune('0'+n)), func(t *testing.T) {
			VmulScalarF32(&out[0], &a[0], 2.5, n)
			for i := range n {
				want := a[i] * 2.5
				if math.Abs(float64(out[i]-want)) > 1e-7 {
					t.Errorf("n=%d, i=%d: got %v, want %v", n, i, out[i], want)
				}
			}
		})
		t.Run("addscalar_tail_"+string(rune('0'+n)), func(t *testing.T) {
			VaddScalarF32(&out[0], &a[0], 100.0, n)
			for i := range n {
				want := a[i] + 100.0
				if out[i] != want {
					t.Errorf("n=%d, i=%d: got %v, want %v", n, i, out[i], want)
				}
			}
		})
		t.Run("divscalar_tail_"+string(rune('0'+n)), func(t *testing.T) {
			VdivScalarF32(&out[0], &a[0], 4.0, n)
			for i := range n {
				want := a[i] / 4.0
				if math.Abs(float64(out[i]-want)) > 1e-7 {
					t.Errorf("n=%d, i=%d: got %v, want %v", n, i, out[i], want)
				}
			}
		})
	}
}

func TestScalar_Random(t *testing.T) {
	const n = 2048
	rng := rand.New(rand.NewPCG(42, 0))
	a := make([]float32, n)
	out := make([]float32, n)
	for i := range n {
		a[i] = float32(rng.Float64()*200 - 100)
	}
	scalar := float32(3.14)

	t.Run("mulscalar_random", func(t *testing.T) {
		VmulScalarF32(&out[0], &a[0], scalar, n)
		for i := range n {
			want := a[i] * scalar
			if math.Abs(float64(out[i]-want)) > 1e-7*math.Max(math.Abs(float64(want)), 1) {
				t.Errorf("i=%d: got %v, want %v", i, out[i], want)
			}
		}
	})
	t.Run("addscalar_random", func(t *testing.T) {
		VaddScalarF32(&out[0], &a[0], scalar, n)
		for i := range n {
			want := a[i] + scalar
			if math.Abs(float64(out[i]-want)) > 1e-7 {
				t.Errorf("i=%d: got %v, want %v", i, out[i], want)
			}
		}
	})
	t.Run("divscalar_random", func(t *testing.T) {
		VdivScalarF32(&out[0], &a[0], scalar, n)
		for i := range n {
			want := a[i] / scalar
			if math.Abs(float64(out[i]-want)) > 1e-7*math.Max(math.Abs(float64(want)), 1) {
				t.Errorf("i=%d: got %v, want %v", i, out[i], want)
			}
		}
	})
}

func TestScalar_ZeroLength(t *testing.T) {
	VmulScalarF32(nil, nil, 1.0, 0)
	VaddScalarF32(nil, nil, 1.0, 0)
	VdivScalarF32(nil, nil, 1.0, 0)
}

func BenchmarkVmulScalarF32(b *testing.B) {
	const n = 2048
	a := make([]float32, n)
	out := make([]float32, n)
	rng := rand.New(rand.NewPCG(42, 0))
	for i := range n {
		a[i] = float32(rng.Float64())
	}
	b.SetBytes(int64(n * 4 * 2))
	b.ResetTimer()
	for range b.N {
		VmulScalarF32(&out[0], &a[0], 2.5, n)
	}
}

func BenchmarkVaddScalarF32(b *testing.B) {
	const n = 2048
	a := make([]float32, n)
	out := make([]float32, n)
	rng := rand.New(rand.NewPCG(42, 0))
	for i := range n {
		a[i] = float32(rng.Float64())
	}
	b.SetBytes(int64(n * 4 * 2))
	b.ResetTimer()
	for range b.N {
		VaddScalarF32(&out[0], &a[0], 10.0, n)
	}
}
