package xblas

import (
	"math"
	"math/rand/v2"
	"testing"
)

func ropeRef(out, in, cos, sin []float32, halfDim, headDim int) {
	for i := 0; i < halfDim; i++ {
		out[i] = in[i]*cos[i] - in[i+halfDim]*sin[i]
		out[i+halfDim] = in[i+halfDim]*cos[i] + in[i]*sin[i]
	}
	for i := halfDim * 2; i < headDim; i++ {
		out[i] = in[i]
	}
}

func TestRoPEF32_Basic(t *testing.T) {
	halfDim := 4
	headDim := 8
	in := []float32{1, 2, 3, 4, 5, 6, 7, 8}
	cos := []float32{0.5, 0.6, 0.7, 0.8}
	sin := []float32{0.1, 0.2, 0.3, 0.4}

	want := make([]float32, headDim)
	ropeRef(want, in, cos, sin, halfDim, headDim)

	got := make([]float32, headDim)
	RoPEF32(&got[0], &in[0], &cos[0], &sin[0], halfDim, headDim)

	for i := range got {
		if diff := float32(math.Abs(float64(got[i] - want[i]))); diff > 1e-6 {
			t.Errorf("index %d: got %v, want %v (diff %v)", i, got[i], want[i], diff)
		}
	}
}

func TestRoPEF32_Identity(t *testing.T) {
	halfDim := 8
	headDim := 16
	in := make([]float32, headDim)
	for i := range in {
		in[i] = float32(i + 1)
	}
	cos := make([]float32, halfDim)
	sin := make([]float32, halfDim)
	for i := range cos {
		cos[i] = 1.0
		sin[i] = 0.0
	}

	got := make([]float32, headDim)
	RoPEF32(&got[0], &in[0], &cos[0], &sin[0], halfDim, headDim)

	for i := range got {
		if got[i] != in[i] {
			t.Errorf("identity: index %d: got %v, want %v", i, got[i], in[i])
		}
	}
}

func TestRoPEF32_Range(t *testing.T) {
	rng := rand.New(rand.NewPCG(42, 0))
	halfDim := 128
	headDim := 256
	in := make([]float32, headDim)
	cos := make([]float32, halfDim)
	sin := make([]float32, halfDim)
	for i := range in {
		in[i] = rng.Float32()*2 - 1
	}
	for i := range cos {
		cos[i] = rng.Float32()*2 - 1
		sin[i] = rng.Float32()*2 - 1
	}

	want := make([]float32, headDim)
	ropeRef(want, in, cos, sin, halfDim, headDim)

	got := make([]float32, headDim)
	RoPEF32(&got[0], &in[0], &cos[0], &sin[0], halfDim, headDim)

	for i := range got {
		if diff := float32(math.Abs(float64(got[i] - want[i]))); diff > 1e-6 {
			t.Errorf("index %d: got %v, want %v (diff %v)", i, got[i], want[i], diff)
		}
	}
}

func TestRoPEF32_Lengths(t *testing.T) {
	rng := rand.New(rand.NewPCG(99, 0))
	for _, halfDim := range []int{2, 4, 7, 64, 128} {
		headDim := halfDim * 2
		in := make([]float32, headDim)
		cos := make([]float32, halfDim)
		sin := make([]float32, halfDim)
		for i := range in {
			in[i] = rng.Float32()*2 - 1
		}
		for i := range cos {
			cos[i] = rng.Float32()*2 - 1
			sin[i] = rng.Float32()*2 - 1
		}

		want := make([]float32, headDim)
		ropeRef(want, in, cos, sin, halfDim, headDim)

		got := make([]float32, headDim)
		RoPEF32(&got[0], &in[0], &cos[0], &sin[0], halfDim, headDim)

		for i := range got {
			if diff := float32(math.Abs(float64(got[i] - want[i]))); diff > 1e-6 {
				t.Errorf("halfDim=%d index %d: got %v, want %v (diff %v)", halfDim, i, got[i], want[i], diff)
			}
		}
	}
}

func TestRoPEF32_Passthrough(t *testing.T) {
	halfDim := 4
	headDim := 12
	in := make([]float32, headDim)
	for i := range in {
		in[i] = float32(i + 1)
	}
	cos := []float32{1, 0, 1, 0}
	sin := []float32{0, 1, 0, 1}

	want := make([]float32, headDim)
	ropeRef(want, in, cos, sin, halfDim, headDim)

	got := make([]float32, headDim)
	RoPEF32(&got[0], &in[0], &cos[0], &sin[0], halfDim, headDim)

	for i := range got {
		if diff := float32(math.Abs(float64(got[i] - want[i]))); diff > 1e-6 {
			t.Errorf("index %d: got %v, want %v (diff %v)", i, got[i], want[i], diff)
		}
	}
	// Specifically check passthrough dimensions
	for i := halfDim * 2; i < headDim; i++ {
		if got[i] != in[i] {
			t.Errorf("passthrough index %d: got %v, want %v", i, got[i], in[i])
		}
	}
}

func BenchmarkRoPEF32(b *testing.B) {
	halfDim := 128
	headDim := 256
	in := make([]float32, headDim)
	out := make([]float32, headDim)
	cos := make([]float32, halfDim)
	sin := make([]float32, halfDim)
	rng := rand.New(rand.NewPCG(0, 0))
	for i := range in {
		in[i] = rng.Float32()
	}
	for i := range cos {
		cos[i] = rng.Float32()
		sin[i] = rng.Float32()
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		RoPEF32(&out[0], &in[0], &cos[0], &sin[0], halfDim, headDim)
	}
}
