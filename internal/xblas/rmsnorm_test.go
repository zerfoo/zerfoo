package xblas

import (
	"math"
	"math/rand/v2"
	"testing"
)

func rmsnormRef(x, weight []float32, eps float32) ([]float32, float32) {
	D := len(x)
	var sumSq float32
	for _, v := range x {
		sumSq += v * v
	}
	scale := float32(1.0 / math.Sqrt(float64(sumSq/float32(D)+eps)))
	out := make([]float32, D)
	for i := range D {
		out[i] = x[i] * scale * weight[i]
	}
	return out, scale
}

func TestRMSNormF32_Basic(t *testing.T) {
	x := []float32{1, 2, 3, 4}
	w := []float32{1, 1, 1, 1}
	out := make([]float32, 4)
	eps := float32(1e-6)

	wantOut, wantScale := rmsnormRef(x, w, eps)
	var gotScale float32
	RMSNormF32(&out[0], &x[0], &w[0], 4, eps, &gotScale)

	for i := range 4 {
		relErr := math.Abs(float64(out[i]-wantOut[i])) / math.Max(math.Abs(float64(wantOut[i])), 1e-30)
		if relErr > 1e-5 {
			t.Errorf("out[%d] = %v, want %v (rel err %e)", i, out[i], wantOut[i], relErr)
		}
	}
	scaleErr := math.Abs(float64(gotScale-wantScale)) / math.Max(math.Abs(float64(wantScale)), 1e-30)
	if scaleErr > 1e-5 {
		t.Errorf("scale = %v, want %v (rel err %e)", gotScale, wantScale, scaleErr)
	}
}

func TestRMSNormF32_Range(t *testing.T) {
	const D = 2048
	rng := rand.New(rand.NewPCG(42, 0))
	x := make([]float32, D)
	w := make([]float32, D)
	out := make([]float32, D)
	for i := range D {
		x[i] = float32(rng.Float64()*4 - 2)
		w[i] = float32(rng.Float64()*2 - 1)
	}
	eps := float32(1e-6)

	wantOut, wantScale := rmsnormRef(x, w, eps)
	var gotScale float32
	RMSNormF32(&out[0], &x[0], &w[0], D, eps, &gotScale)

	var maxRelErr float64
	for i := range D {
		relErr := math.Abs(float64(out[i]-wantOut[i])) / math.Max(math.Abs(float64(wantOut[i])), 1e-30)
		if relErr > maxRelErr {
			maxRelErr = relErr
		}
	}
	if maxRelErr > 1e-5 {
		t.Errorf("max relative error %e > 1e-5", maxRelErr)
	}
	t.Logf("max relative error over %d values: %e", D, maxRelErr)

	scaleErr := math.Abs(float64(gotScale-wantScale)) / math.Max(math.Abs(float64(wantScale)), 1e-30)
	if scaleErr > 1e-5 {
		t.Errorf("scale = %v, want %v (rel err %e)", gotScale, wantScale, scaleErr)
	}
}

func TestRMSNormF32_Lengths(t *testing.T) {
	rng := rand.New(rand.NewPCG(99, 0))
	for _, D := range []int{1, 4, 7, 128, 2048} {
		t.Run("D="+itoa(D), func(t *testing.T) {
			x := make([]float32, D)
			w := make([]float32, D)
			out := make([]float32, D)
			for i := range D {
				x[i] = float32(rng.Float64()*4 - 2)
				w[i] = float32(rng.Float64()*2)
			}
			eps := float32(1e-6)
			wantOut, wantScale := rmsnormRef(x, w, eps)
			var gotScale float32
			RMSNormF32(&out[0], &x[0], &w[0], D, eps, &gotScale)

			var maxRelErr float64
			for i := range D {
				relErr := math.Abs(float64(out[i]-wantOut[i])) / math.Max(math.Abs(float64(wantOut[i])), 1e-30)
				if relErr > maxRelErr {
					maxRelErr = relErr
				}
			}
			if maxRelErr > 1e-5 {
				t.Errorf("D=%d: max rel err %e > 1e-5", D, maxRelErr)
			}
			scaleErr := math.Abs(float64(gotScale-wantScale)) / math.Max(math.Abs(float64(wantScale)), 1e-30)
			if scaleErr > 1e-5 {
				t.Errorf("D=%d: scale err %e > 1e-5", D, scaleErr)
			}
		})
	}
}

func TestRMSNormF32_UniformInput(t *testing.T) {
	const D = 128
	x := make([]float32, D)
	w := make([]float32, D)
	out := make([]float32, D)
	for i := range D {
		x[i] = 3.0
		w[i] = 1.0
	}
	eps := float32(1e-6)
	var gotScale float32
	RMSNormF32(&out[0], &x[0], &w[0], D, eps, &gotScale)

	// mean(x^2) = 9, rms = 3, scale = 1/3
	wantScale := float32(1.0 / math.Sqrt(float64(9.0+eps)))
	scaleErr := math.Abs(float64(gotScale-wantScale)) / float64(wantScale)
	if scaleErr > 1e-5 {
		t.Errorf("uniform scale = %v, want %v (rel err %e)", gotScale, wantScale, scaleErr)
	}

	// Each output should be 3 * (1/3) * 1 = 1.0
	for i := range D {
		if math.Abs(float64(out[i]-1.0)) > 1e-4 {
			t.Errorf("out[%d] = %v, want ~1.0", i, out[i])
			break
		}
	}
}

func TestRMSNormF32_ReturnScale(t *testing.T) {
	x := []float32{2, 2, 2, 2}
	w := []float32{1, 1, 1, 1}
	out := make([]float32, 4)
	eps := float32(1e-6)
	var gotScale float32
	RMSNormF32(&out[0], &x[0], &w[0], 4, eps, &gotScale)

	// mean(x^2) = 4, scale = 1/sqrt(4+eps) = ~0.5
	wantScale := float32(1.0 / math.Sqrt(float64(4.0+eps)))
	relErr := math.Abs(float64(gotScale-wantScale)) / float64(wantScale)
	if relErr > 1e-5 {
		t.Errorf("scale = %v, want %v (rel err %e)", gotScale, wantScale, relErr)
	}
}

func BenchmarkRMSNormF32(b *testing.B) {
	const D = 2048
	rng := rand.New(rand.NewPCG(42, 0))
	x := make([]float32, D)
	w := make([]float32, D)
	out := make([]float32, D)
	for i := range D {
		x[i] = float32(rng.Float64()*4 - 2)
		w[i] = float32(rng.Float64()*2 - 1)
	}
	eps := float32(1e-6)
	var scale float32

	b.SetBytes(int64(D * 4 * 3)) // read x + weight, write out
	b.ResetTimer()
	for range b.N {
		RMSNormF32(&out[0], &x[0], &w[0], D, eps, &scale)
	}
}

func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	buf := [20]byte{}
	i := len(buf) - 1
	for n > 0 {
		buf[i] = byte('0' + n%10)
		n /= 10
		i--
	}
	return string(buf[i+1:])
}
