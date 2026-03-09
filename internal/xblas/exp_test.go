package xblas

import (
	"math"
	"math/rand/v2"
	"testing"
)

func TestVexpF32_Basic(t *testing.T) {
	tests := []struct {
		name string
		x    float32
		want float64
		tol  float64
	}{
		{"exp(0)=1", 0, 1.0, 1e-7},
		{"exp(1)~2.718", 1, math.E, 5e-6},
		{"exp(-1)~0.368", -1, 1.0 / math.E, 5e-6},
		{"exp(2)~7.389", 2, math.Exp(2), 5e-6},
		{"exp(-2)~0.135", -2, math.Exp(-2), 5e-6},
		{"exp(0.5)", 0.5, math.Exp(0.5), 5e-6},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var in, out float32
			in = tt.x
			VexpF32(&out, &in, 1)
			got := float64(out)
			relErr := math.Abs(got-tt.want) / math.Max(math.Abs(tt.want), 1e-30)
			if relErr > tt.tol {
				t.Errorf("VexpF32(%v) = %v, want %v (rel err %e > %e)", tt.x, got, tt.want, relErr, tt.tol)
			}
		})
	}
}

func TestVexpF32_Range(t *testing.T) {
	const n = 10000
	rng := rand.New(rand.NewPCG(42, 0))

	xs := make([]float32, n)
	outs := make([]float32, n)

	for i := range xs {
		xs[i] = float32(rng.Float64()*160 - 80) // [-80, 80] safe range
	}

	VexpF32(&outs[0], &xs[0], n)

	var maxRelErr float64
	for i := range n {
		want := math.Exp(float64(xs[i]))
		got := float64(outs[i])
		if want == 0 || math.IsInf(want, 0) {
			continue
		}
		relErr := math.Abs(got-want) / want
		if relErr > maxRelErr {
			maxRelErr = relErr
		}
	}

	// Degree-5 Taylor polynomial has ~2.5e-6 truncation error at boundary,
	// compounded by float32 rounding in the Horner evaluation.
	if maxRelErr > 1e-5 {
		t.Errorf("max relative error %e > 1e-5", maxRelErr)
	}
	t.Logf("max relative error over %d values: %e", n, maxRelErr)
}

func TestVexpF32_EdgeCases(t *testing.T) {
	t.Run("very_negative_underflow", func(t *testing.T) {
		x := float32(-100)
		var out float32
		VexpF32(&out, &x, 1)
		// Input is clamped to -87, so exp(-87) ~ 1.2e-38
		if out < 0 || out > 1e-30 {
			t.Errorf("exp(-100) = %v, expected small positive (clamped)", out)
		}
	})

	t.Run("large_positive", func(t *testing.T) {
		x := float32(80)
		var out float32
		VexpF32(&out, &x, 1)
		want := math.Exp(80)
		got := float64(out)
		relErr := math.Abs(got-want) / want
		if relErr > 5e-6 {
			t.Errorf("exp(80) = %v, want %v (rel err %e)", got, want, relErr)
		}
	})

	t.Run("zero_length", func(t *testing.T) {
		// Should not crash.
		VexpF32(nil, nil, 0)
	})
}

func TestVexpF32_TailHandling(t *testing.T) {
	for _, n := range []int{1, 2, 3, 5, 7} {
		t.Run("n="+string(rune('0'+n)), func(t *testing.T) {
			xs := make([]float32, n)
			outs := make([]float32, n)
			for i := range xs {
				xs[i] = float32(i) * 0.5
			}

			VexpF32(&outs[0], &xs[0], n)

			for i := range n {
				want := math.Exp(float64(xs[i]))
				got := float64(outs[i])
				relErr := math.Abs(got-want) / math.Max(want, 1e-30)
				if relErr > 5e-6 {
					t.Errorf("n=%d, i=%d: exp(%v) = %v, want %v (rel err %e)",
						n, i, xs[i], got, want, relErr)
				}
			}
		})
	}
}

func BenchmarkVexpF32(b *testing.B) {
	const n = 2048
	xs := make([]float32, n)
	outs := make([]float32, n)
	rng := rand.New(rand.NewPCG(42, 0))
	for i := range xs {
		xs[i] = float32(rng.Float64()*20 - 10) // [-10, 10]
	}

	b.SetBytes(int64(n * 4))
	b.ResetTimer()
	for range b.N {
		VexpF32(&outs[0], &xs[0], n)
	}
}
