package xblas

import (
	"math"
	"math/rand/v2"
	"testing"
)

func TestVaddF32(t *testing.T) {
	a := []float32{1, 2, 3, 4, 5, 6, 7, 8}
	b := []float32{10, 20, 30, 40, 50, 60, 70, 80}
	out := make([]float32, 8)
	VaddF32(&out[0], &a[0], &b[0], 8)
	want := []float32{11, 22, 33, 44, 55, 66, 77, 88}
	for i := range out {
		if out[i] != want[i] {
			t.Errorf("VaddF32[%d] = %v, want %v", i, out[i], want[i])
		}
	}
}

func TestVmulF32(t *testing.T) {
	a := []float32{1, 2, 3, 4, 5, 6, 7, 8}
	b := []float32{2, 3, 4, 5, 6, 7, 8, 9}
	out := make([]float32, 8)
	VmulF32(&out[0], &a[0], &b[0], 8)
	want := []float32{2, 6, 12, 20, 30, 42, 56, 72}
	for i := range out {
		if out[i] != want[i] {
			t.Errorf("VmulF32[%d] = %v, want %v", i, out[i], want[i])
		}
	}
}

func TestVsubF32(t *testing.T) {
	a := []float32{10, 20, 30, 40, 50, 60, 70, 80}
	b := []float32{1, 2, 3, 4, 5, 6, 7, 8}
	out := make([]float32, 8)
	VsubF32(&out[0], &a[0], &b[0], 8)
	want := []float32{9, 18, 27, 36, 45, 54, 63, 72}
	for i := range out {
		if out[i] != want[i] {
			t.Errorf("VsubF32[%d] = %v, want %v", i, out[i], want[i])
		}
	}
}

func TestVdivF32(t *testing.T) {
	a := []float32{10, 20, 30, 40, 50, 60, 70, 80}
	b := []float32{2, 4, 5, 8, 10, 12, 14, 16}
	out := make([]float32, 8)
	VdivF32(&out[0], &a[0], &b[0], 8)
	want := []float32{5, 5, 6, 5, 5, 5, 5, 5}
	for i := range out {
		if out[i] != want[i] {
			t.Errorf("VdivF32[%d] = %v, want %v", i, out[i], want[i])
		}
	}
}

func TestElementwise_TailHandling(t *testing.T) {
	for _, n := range []int{1, 3, 7} {
		a := make([]float32, n)
		b := make([]float32, n)
		out := make([]float32, n)
		for i := range n {
			a[i] = float32(i + 1)
			b[i] = float32(i + 2)
		}

		t.Run("add_tail_"+string(rune('0'+n)), func(t *testing.T) {
			VaddF32(&out[0], &a[0], &b[0], n)
			for i := range n {
				if out[i] != a[i]+b[i] {
					t.Errorf("VaddF32 tail n=%d, i=%d: got %v, want %v", n, i, out[i], a[i]+b[i])
				}
			}
		})
		t.Run("mul_tail_"+string(rune('0'+n)), func(t *testing.T) {
			VmulF32(&out[0], &a[0], &b[0], n)
			for i := range n {
				if out[i] != a[i]*b[i] {
					t.Errorf("VmulF32 tail n=%d, i=%d: got %v, want %v", n, i, out[i], a[i]*b[i])
				}
			}
		})
		t.Run("sub_tail_"+string(rune('0'+n)), func(t *testing.T) {
			VsubF32(&out[0], &a[0], &b[0], n)
			for i := range n {
				if out[i] != a[i]-b[i] {
					t.Errorf("VsubF32 tail n=%d, i=%d: got %v, want %v", n, i, out[i], a[i]-b[i])
				}
			}
		})
		t.Run("div_tail_"+string(rune('0'+n)), func(t *testing.T) {
			VdivF32(&out[0], &a[0], &b[0], n)
			for i := range n {
				want := a[i] / b[i]
				if math.Abs(float64(out[i]-want)) > 1e-7 {
					t.Errorf("VdivF32 tail n=%d, i=%d: got %v, want %v", n, i, out[i], want)
				}
			}
		})
	}
}

func TestElementwise_Random(t *testing.T) {
	const n = 2048
	rng := rand.New(rand.NewPCG(42, 0))
	a := make([]float32, n)
	b := make([]float32, n)
	out := make([]float32, n)
	for i := range n {
		a[i] = float32(rng.Float64()*200 - 100)
		b[i] = float32(rng.Float64()*200-100) + 0.01 // avoid div by zero
	}

	t.Run("add_random", func(t *testing.T) {
		VaddF32(&out[0], &a[0], &b[0], n)
		for i := range n {
			want := a[i] + b[i]
			if math.Abs(float64(out[i]-want)) > 1e-7 {
				t.Errorf("i=%d: got %v, want %v", i, out[i], want)
			}
		}
	})
	t.Run("mul_random", func(t *testing.T) {
		VmulF32(&out[0], &a[0], &b[0], n)
		for i := range n {
			want := a[i] * b[i]
			if math.Abs(float64(out[i]-want)) > 1e-7*math.Abs(float64(want)) {
				t.Errorf("i=%d: got %v, want %v", i, out[i], want)
			}
		}
	})
	t.Run("sub_random", func(t *testing.T) {
		VsubF32(&out[0], &a[0], &b[0], n)
		for i := range n {
			want := a[i] - b[i]
			if math.Abs(float64(out[i]-want)) > 1e-7 {
				t.Errorf("i=%d: got %v, want %v", i, out[i], want)
			}
		}
	})
	t.Run("div_random", func(t *testing.T) {
		VdivF32(&out[0], &a[0], &b[0], n)
		for i := range n {
			want := a[i] / b[i]
			if math.Abs(float64(out[i]-want)) > 1e-7*math.Max(math.Abs(float64(want)), 1) {
				t.Errorf("i=%d: got %v, want %v", i, out[i], want)
			}
		}
	})
}

func TestElementwise_ZeroLength(t *testing.T) {
	VaddF32(nil, nil, nil, 0)
	VmulF32(nil, nil, nil, 0)
	VsubF32(nil, nil, nil, 0)
	VdivF32(nil, nil, nil, 0)
}

func BenchmarkVaddF32(b *testing.B) {
	const n = 2048
	a := make([]float32, n)
	bs := make([]float32, n)
	out := make([]float32, n)
	rng := rand.New(rand.NewPCG(42, 0))
	for i := range n {
		a[i] = float32(rng.Float64())
		bs[i] = float32(rng.Float64())
	}
	b.SetBytes(int64(n * 4 * 3)) // read a + b, write out
	b.ResetTimer()
	for range b.N {
		VaddF32(&out[0], &a[0], &bs[0], n)
	}
}

func BenchmarkVmulF32(b *testing.B) {
	const n = 2048
	a := make([]float32, n)
	bs := make([]float32, n)
	out := make([]float32, n)
	rng := rand.New(rand.NewPCG(42, 0))
	for i := range n {
		a[i] = float32(rng.Float64())
		bs[i] = float32(rng.Float64())
	}
	b.SetBytes(int64(n * 4 * 3))
	b.ResetTimer()
	for range b.N {
		VmulF32(&out[0], &a[0], &bs[0], n)
	}
}
