package xblas

import (
	"math"
	"math/rand/v2"
	"testing"
)

func referenceSoftmax(x []float32) []float32 {
	out := make([]float32, len(x))
	mx := float64(x[0])
	for _, v := range x[1:] {
		if float64(v) > mx {
			mx = float64(v)
		}
	}
	var sum float64
	for i, v := range x {
		out[i] = float32(math.Exp(float64(v) - mx))
		sum += float64(out[i])
	}
	for i := range out {
		out[i] = float32(float64(out[i]) / sum)
	}
	return out
}

func TestSoftmaxF32_Basic(t *testing.T) {
	input := []float32{1, 2, 3, 4}
	data := make([]float32, len(input))
	copy(data, input)

	SoftmaxF32(&data[0], len(data))

	ref := referenceSoftmax(input)

	var sum float32
	for i, got := range data {
		sum += got
		diff := math.Abs(float64(got - ref[i]))
		if diff > 1e-5 {
			t.Errorf("index %d: got %v, want %v (diff %e)", i, got, ref[i], diff)
		}
	}
	if math.Abs(float64(sum)-1.0) > 1e-5 {
		t.Errorf("sum = %v, want ~1.0", sum)
	}
}

func TestSoftmaxF32_Range(t *testing.T) {
	const n = 2048
	rng := rand.New(rand.NewPCG(42, 0))

	input := make([]float32, n)
	for i := range input {
		input[i] = float32(rng.Float64()*20 - 10) // [-10, 10]
	}

	data := make([]float32, n)
	copy(data, input)
	SoftmaxF32(&data[0], n)

	ref := referenceSoftmax(input)

	var maxRelErr float64
	for i := range n {
		if ref[i] == 0 {
			continue
		}
		relErr := math.Abs(float64(data[i]-ref[i])) / float64(ref[i])
		if relErr > maxRelErr {
			maxRelErr = relErr
		}
	}
	if maxRelErr > 1e-5 {
		t.Errorf("max relative error %e > 1e-5", maxRelErr)
	}
	t.Logf("max relative error over %d values: %e", n, maxRelErr)
}

func TestSoftmaxF32_Lengths(t *testing.T) {
	for _, n := range []int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 16, 128, 2048} {
		t.Run("n="+intToStr(n), func(t *testing.T) {
			rng := rand.New(rand.NewPCG(99, uint64(n)))
			input := make([]float32, n)
			for i := range input {
				input[i] = float32(rng.Float64()*10 - 5)
			}

			data := make([]float32, n)
			copy(data, input)
			SoftmaxF32(&data[0], n)

			ref := referenceSoftmax(input)

			var sum float32
			for i, got := range data {
				sum += got
				if ref[i] == 0 {
					continue
				}
				relErr := math.Abs(float64(got-ref[i])) / float64(ref[i])
				if relErr > 1e-5 {
					t.Errorf("i=%d: got %v, want %v (relErr %e)", i, got, ref[i], relErr)
				}
			}
			if math.Abs(float64(sum)-1.0) > 1e-4 {
				t.Errorf("sum = %v, want ~1.0", sum)
			}
		})
	}
}

func TestSoftmaxF32_EdgeCases(t *testing.T) {
	t.Run("all_same", func(t *testing.T) {
		data := []float32{5, 5, 5, 5}
		SoftmaxF32(&data[0], len(data))
		for i, v := range data {
			if math.Abs(float64(v)-0.25) > 1e-5 {
				t.Errorf("index %d: got %v, want 0.25", i, v)
			}
		}
	})

	t.Run("single_element", func(t *testing.T) {
		data := []float32{42}
		SoftmaxF32(&data[0], 1)
		if math.Abs(float64(data[0])-1.0) > 1e-6 {
			t.Errorf("got %v, want 1.0", data[0])
		}
	})
}

func TestSoftmaxF32_CausalMask(t *testing.T) {
	// Test softmax with -1e9 values (simulating causal mask).
	// Row 0 of an 8x8 causal mask: only position 0 visible.
	for _, n := range []int{4, 8, 16} {
		t.Run("n="+intToStr(n), func(t *testing.T) {
			for visibleCount := 1; visibleCount <= n; visibleCount++ {
				input := make([]float32, n)
				for i := range n {
					if i < visibleCount {
						input[i] = float32(i) * 0.5
					} else {
						input[i] = -1e9
					}
				}

				data := make([]float32, n)
				copy(data, input)
				SoftmaxF32(&data[0], n)

				ref := referenceSoftmax(input)

				for i, got := range data {
					diff := math.Abs(float64(got - ref[i]))
					if diff > 1e-5 {
						t.Errorf("visible=%d i=%d: got %v, want %v (diff %e)", visibleCount, i, got, ref[i], diff)
					}
				}
			}
		})
	}
}

func TestSoftmaxF32_MaxInFirstChunk(t *testing.T) {
	// Regression test: max value appears only in the first 4-element NEON
	// chunk while the remaining elements are -1e9 (causal mask). A prior
	// bug in the FMAXV encoding caused the horizontal max reduction to
	// read the last loaded vector (V0) instead of the accumulator (V30),
	// producing NaN when the true max lived in an earlier chunk.
	for _, maxVal := range []float32{0.5, 1.0, 1.73, 5.0, 10.0, 100.0} {
		t.Run("max="+intToStr(int(maxVal*100)), func(t *testing.T) {
			input := make([]float32, 8)
			input[0] = maxVal
			for i := 1; i < 8; i++ {
				input[i] = -1e9
			}
			data := make([]float32, 8)
			copy(data, input)
			SoftmaxF32(&data[0], 8)
			ref := referenceSoftmax(input)
			for i, got := range data {
				if math.IsNaN(float64(got)) {
					t.Fatalf("NaN at index %d: input=%v got=%v ref=%v", i, input, data, ref)
				}
				diff := math.Abs(float64(got - ref[i]))
				if diff > 1e-5 {
					t.Errorf("index %d: got %v, want %v (diff %e)", i, got, ref[i], diff)
				}
			}
		})
	}
}

func BenchmarkSoftmaxF32(b *testing.B) {
	const n = 2048
	data := make([]float32, n)
	rng := rand.New(rand.NewPCG(42, 0))
	for i := range data {
		data[i] = float32(rng.Float64()*20 - 10)
	}

	buf := make([]float32, n)
	b.SetBytes(int64(n * 4))
	b.ResetTimer()
	for range b.N {
		copy(buf, data)
		SoftmaxF32(&buf[0], n)
	}
}

func intToStr(n int) string {
	if n == 0 {
		return "0"
	}
	var buf [20]byte
	i := len(buf)
	for n > 0 {
		i--
		buf[i] = byte('0' + n%10)
		n /= 10
	}
	return string(buf[i:])
}
