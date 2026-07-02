package dsp

import (
	"math"
	"math/cmplx"
	"testing"
)

const eps = 1e-10 // tolerance for float comparison

// assertClose fails the test if |got - want| > eps.
func assertClose(t *testing.T, label string, got, want float64) {
	t.Helper()
	if math.Abs(got-want) > eps {
		t.Errorf("%s: got %v, want %v (diff %v)", label, got, want, got-want)
	}
}

// assertCloseComplex fails the test if |got - want| > eps.
func assertCloseComplex(t *testing.T, label string, got, want complex128) {
	t.Helper()
	if cmplx.Abs(got-want) > eps {
		t.Errorf("%s: got %v, want %v", label, got, want)
	}
}

// --- Known DFT outputs ---

func TestCoefficients_DCSignal(t *testing.T) {
	// Constant signal x[n] = 3.0 for N=4.
	// DFT: X[0] = 4*3 = 12, X[1] = X[2] = X[3] = 0.
	data := []float64{3, 3, 3, 3}
	c := Coefficients(data)
	if len(c) != 4 {
		t.Fatalf("len = %d, want 4", len(c))
	}
	assertCloseComplex(t, "X[0]", c[0], 12+0i)
	assertCloseComplex(t, "X[1]", c[1], 0+0i)
	assertCloseComplex(t, "X[2]", c[2], 0+0i)
	assertCloseComplex(t, "X[3]", c[3], 0+0i)
}

func TestCoefficients_Impulse(t *testing.T) {
	// Impulse x = {1, 0, 0, 0, 0, 0, 0, 0} (N=8).
	// DFT of impulse: X[k] = 1 for all k.
	data := []float64{1, 0, 0, 0, 0, 0, 0, 0}
	c := Coefficients(data)
	if len(c) != 8 {
		t.Fatalf("len = %d, want 8", len(c))
	}
	for k := 0; k < 8; k++ {
		assertCloseComplex(t, "X[k]", c[k], 1+0i)
	}
}

func TestCoefficients_SingleFrequencySinusoid(t *testing.T) {
	// x[n] = cos(2*pi*n/N) for N=8 is a pure cosine at bin k=1.
	// DFT: X[1] = N/2 = 4, X[N-1] = N/2 = 4, rest zero.
	const N = 8
	data := make([]float64, N)
	for n := 0; n < N; n++ {
		data[n] = math.Cos(2 * math.Pi * float64(n) / float64(N))
	}
	c := Coefficients(data)
	if len(c) != N {
		t.Fatalf("len = %d, want %d", len(c), N)
	}
	for k := 0; k < N; k++ {
		var want complex128
		switch k {
		case 1:
			want = 4 + 0i
		case N - 1:
			want = 4 + 0i
		default:
			want = 0 + 0i
		}
		assertCloseComplex(t, "X[k]", c[k], want)
	}
}

// --- Power-of-2 lengths ---

func TestCoefficients_PowerOfTwoLengths(t *testing.T) {
	for _, n := range []int{2, 4, 8, 16, 32, 64, 128} {
		t.Run("N="+itoa(n), func(t *testing.T) {
			// Impulse: DFT should be all ones.
			data := make([]float64, n)
			data[0] = 1
			c := Coefficients(data)
			if len(c) != n {
				t.Fatalf("len = %d, want %d", len(c), n)
			}
			for k := 0; k < n; k++ {
				assertCloseComplex(t, "X[k]", c[k], 1+0i)
			}
		})
	}
}

// --- Non-power-of-2 lengths (zero-padding) ---

func TestCoefficients_NonPowerOfTwo(t *testing.T) {
	for _, n := range []int{3, 5, 7, 10} {
		t.Run("N="+itoa(n), func(t *testing.T) {
			// DC signal x[i] = 2.0 of length n.
			// Zero-padded to m = nextPow2(n).
			// DFT of [2,2,...,2,0,...,0] (n twos, m-n zeros).
			// X[k] = 2 * sum_{j=0}^{n-1} exp(-i*2*pi*k*j/m).
			data := make([]float64, n)
			for i := range data {
				data[i] = 2
			}
			m := nextPow2(n)
			c := Coefficients(data)
			if len(c) != m {
				t.Fatalf("len = %d, want %d", len(c), m)
			}

			// Verify against a direct DFT computation.
			for k := 0; k < m; k++ {
				var want complex128
				for j := 0; j < n; j++ {
					angle := -2 * math.Pi * float64(k) * float64(j) / float64(m)
					want += complex(2*math.Cos(angle), 2*math.Sin(angle))
				}
				if cmplx.Abs(c[k]-want) > 1e-8 {
					t.Errorf("k=%d: got %v, want %v", k, c[k], want)
				}
			}
		})
	}
}

func TestCoefficients_Nil(t *testing.T) {
	c := Coefficients(nil)
	if c != nil {
		t.Errorf("expected nil, got %v", c)
	}
}

func TestCoefficients_Empty(t *testing.T) {
	c := Coefficients([]float64{})
	if c != nil {
		t.Errorf("expected nil, got %v", c)
	}
}

func TestCoefficients_SingleElement(t *testing.T) {
	c := Coefficients([]float64{5})
	if len(c) != 1 {
		t.Fatalf("len = %d, want 1", len(c))
	}
	assertCloseComplex(t, "X[0]", c[0], 5+0i)
}

// --- Benchmark ---

func BenchmarkCoefficients(b *testing.B) {
	for _, n := range []int{2, 4, 8, 16, 32, 64, 128} {
		data := make([]float64, n)
		for i := range data {
			data[i] = float64(i)
		}
		b.Run("N="+itoa(n), func(b *testing.B) {
			for b.Loop() {
				Coefficients(data)
			}
		})
	}
}

// itoa is a simple int-to-string for test names without importing strconv.
func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	buf := [20]byte{}
	i := len(buf)
	for n > 0 {
		i--
		buf[i] = byte('0' + n%10)
		n /= 10
	}
	return string(buf[i:])
}
