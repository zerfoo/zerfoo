package xblas

import (
	"math"
	"math/rand/v2"
	"testing"
)

func referenceSiLU(x float64) float64 {
	return x / (1 + math.Exp(-x))
}

func TestSiLUF32_Basic(t *testing.T) {
	tests := []struct {
		name string
		x    float32
		want float64
		tol  float64
	}{
		{"silu(0)=0", 0, 0, 1e-7},
		{"silu(1)", 1, referenceSiLU(1), 1e-5},
		{"silu(-1)", -1, referenceSiLU(-1), 1e-5},
		{"silu(2)", 2, referenceSiLU(2), 1e-5},
		{"silu(-2)", -2, referenceSiLU(-2), 1e-5},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var in, out float32
			in = tt.x
			SiLUF32(&out, &in, 1)
			got := float64(out)
			diff := math.Abs(got - tt.want)
			denom := math.Max(math.Abs(tt.want), 1e-30)
			relErr := diff / denom
			if relErr > tt.tol && diff > 1e-7 {
				t.Errorf("SiLUF32(%v) = %v, want %v (rel err %e > %e)", tt.x, got, tt.want, relErr, tt.tol)
			}
		})
	}
}

func TestSiLUF32_Range(t *testing.T) {
	const n = 2048
	rng := rand.New(rand.NewPCG(42, 0))

	xs := make([]float32, n)
	outs := make([]float32, n)

	for i := range xs {
		xs[i] = float32(rng.Float64()*20 - 10) // [-10, 10]
	}

	SiLUF32(&outs[0], &xs[0], n)

	var maxRelErr float64
	for i := range n {
		want := referenceSiLU(float64(xs[i]))
		got := float64(outs[i])
		denom := math.Max(math.Abs(want), 1e-30)
		relErr := math.Abs(got-want) / denom
		if relErr > maxRelErr {
			maxRelErr = relErr
		}
	}

	if maxRelErr > 1e-5 {
		t.Errorf("max relative error %e > 1e-5", maxRelErr)
	}
	t.Logf("SiLUF32 max relative error over %d values: %e", n, maxRelErr)
}

func TestSiLUF32_Lengths(t *testing.T) {
	for _, n := range []int{1, 4, 7, 128, 2048} {
		t.Run("n="+itoa(n), func(t *testing.T) {
			xs := make([]float32, n)
			outs := make([]float32, n)
			for i := range xs {
				xs[i] = float32(i)*0.1 - float32(n)*0.05
			}

			SiLUF32(&outs[0], &xs[0], n)

			for i := range n {
				want := referenceSiLU(float64(xs[i]))
				got := float64(outs[i])
				denom := math.Max(math.Abs(want), 1e-30)
				relErr := math.Abs(got-want) / denom
				if relErr > 1e-5 && math.Abs(got-want) > 1e-7 {
					t.Errorf("n=%d, i=%d: SiLUF32(%v) = %v, want %v (rel err %e)",
						n, i, xs[i], got, want, relErr)
				}
			}
		})
	}
}

func TestSiLUGateF32_Basic(t *testing.T) {
	tests := []struct {
		name     string
		gate, up float32
		want     float64
		tol      float64
	}{
		{"zero_gate", 0, 1.0, 0, 1e-7},
		{"one_one", 1, 1.0, referenceSiLU(1), 1e-5},
		{"neg_one", -1, 2.0, referenceSiLU(-1) * 2.0, 1e-5},
		{"two_half", 2, 0.5, referenceSiLU(2) * 0.5, 1e-5},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var out float32
			g := tt.gate
			u := tt.up
			SiLUGateF32(&out, &g, &u, 1)
			got := float64(out)
			diff := math.Abs(got - tt.want)
			denom := math.Max(math.Abs(tt.want), 1e-30)
			relErr := diff / denom
			if relErr > tt.tol && diff > 1e-7 {
				t.Errorf("SiLUGateF32(gate=%v, up=%v) = %v, want %v (rel err %e > %e)",
					tt.gate, tt.up, got, tt.want, relErr, tt.tol)
			}
		})
	}
}

func TestSiLUGateF32_Range(t *testing.T) {
	const n = 2048
	rng := rand.New(rand.NewPCG(99, 0))

	gates := make([]float32, n)
	ups := make([]float32, n)
	outs := make([]float32, n)

	for i := range n {
		gates[i] = float32(rng.Float64()*20 - 10)
		ups[i] = float32(rng.Float64()*4 - 2)
	}

	SiLUGateF32(&outs[0], &gates[0], &ups[0], n)

	var maxRelErr float64
	for i := range n {
		want := referenceSiLU(float64(gates[i])) * float64(ups[i])
		got := float64(outs[i])
		denom := math.Max(math.Abs(want), 1e-30)
		relErr := math.Abs(got-want) / denom
		if relErr > maxRelErr {
			maxRelErr = relErr
		}
	}

	if maxRelErr > 1e-5 {
		t.Errorf("max relative error %e > 1e-5", maxRelErr)
	}
	t.Logf("SiLUGateF32 max relative error over %d values: %e", n, maxRelErr)
}

func BenchmarkSiLUF32(b *testing.B) {
	const n = 2048
	xs := make([]float32, n)
	outs := make([]float32, n)
	rng := rand.New(rand.NewPCG(42, 0))
	for i := range xs {
		xs[i] = float32(rng.Float64()*20 - 10)
	}

	b.SetBytes(int64(n * 4))
	b.ResetTimer()
	for range b.N {
		SiLUF32(&outs[0], &xs[0], n)
	}
}

func BenchmarkSiLUGateF32(b *testing.B) {
	const n = 2048
	gates := make([]float32, n)
	ups := make([]float32, n)
	outs := make([]float32, n)
	rng := rand.New(rand.NewPCG(42, 0))
	for i := range n {
		gates[i] = float32(rng.Float64()*20 - 10)
		ups[i] = float32(rng.Float64()*4 - 2)
	}

	b.SetBytes(int64(n * 4 * 2))
	b.ResetTimer()
	for range b.N {
		SiLUGateF32(&outs[0], &gates[0], &ups[0], n)
	}
}

