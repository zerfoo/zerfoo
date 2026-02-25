package xblas

import (
	"math"
	"testing"

	float16 "github.com/zerfoo/float16"
	float8 "github.com/zerfoo/float8"
)

func TestGemmF32_Identity(t *testing.T) {
	// A = [[1, 0], [0, 1]] (identity 2x2)
	// B = [[3, 4], [5, 6]]
	// C = A * B = B
	a := []float32{1, 0, 0, 1}
	b := []float32{3, 4, 5, 6}
	c := make([]float32, 4)

	GemmF32(2, 2, 2, a, b, c)

	want := []float32{3, 4, 5, 6}
	for i, v := range c {
		if v != want[i] {
			t.Errorf("c[%d] = %v, want %v", i, v, want[i])
		}
	}
}

func TestGemmF32_KnownProduct(t *testing.T) {
	// A = [[1, 2], [3, 4]] (2x2)
	// B = [[5, 6], [7, 8]] (2x2)
	// C = [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]] = [[19, 22], [43, 50]]
	a := []float32{1, 2, 3, 4}
	b := []float32{5, 6, 7, 8}
	c := make([]float32, 4)

	GemmF32(2, 2, 2, a, b, c)

	want := []float32{19, 22, 43, 50}
	for i, v := range c {
		if v != want[i] {
			t.Errorf("c[%d] = %v, want %v", i, v, want[i])
		}
	}
}

func TestGemmF32_NonSquare(t *testing.T) {
	// A = [[1, 2, 3]] (1x3)
	// B = [[4], [5], [6]] (3x1)
	// C = [[1*4+2*5+3*6]] = [[32]]
	a := []float32{1, 2, 3}
	b := []float32{4, 5, 6}
	c := make([]float32, 1)

	GemmF32(1, 1, 3, a, b, c)

	if c[0] != 32 {
		t.Errorf("c[0] = %v, want 32", c[0])
	}
}

func TestGemmF32_ZeroMatrix(t *testing.T) {
	a := []float32{0, 0, 0, 0}
	b := []float32{1, 2, 3, 4}
	c := make([]float32, 4)

	GemmF32(2, 2, 2, a, b, c)

	for i, v := range c {
		if v != 0 {
			t.Errorf("c[%d] = %v, want 0", i, v)
		}
	}
}

func TestGemmF64_Identity(t *testing.T) {
	a := []float64{1, 0, 0, 1}
	b := []float64{3, 4, 5, 6}
	c := make([]float64, 4)

	GemmF64(2, 2, 2, a, b, c)

	want := []float64{3, 4, 5, 6}
	for i, v := range c {
		if v != want[i] {
			t.Errorf("c[%d] = %v, want %v", i, v, want[i])
		}
	}
}

func TestGemmF64_KnownProduct(t *testing.T) {
	a := []float64{1, 2, 3, 4}
	b := []float64{5, 6, 7, 8}
	c := make([]float64, 4)

	GemmF64(2, 2, 2, a, b, c)

	want := []float64{19, 22, 43, 50}
	for i, v := range c {
		if v != want[i] {
			t.Errorf("c[%d] = %v, want %v", i, v, want[i])
		}
	}
}

func TestGemmF64_ZeroMatrix(t *testing.T) {
	a := []float64{0, 0, 0, 0}
	b := []float64{1, 2, 3, 4}
	c := make([]float64, 4)

	GemmF64(2, 2, 2, a, b, c)

	for i, v := range c {
		if v != 0 {
			t.Errorf("c[%d] = %v, want 0", i, v)
		}
	}
}

func TestGemmF16_KnownProduct(t *testing.T) {
	// A = [[1, 2], [3, 4]]
	// B = [[5, 6], [7, 8]]
	// C = [[19, 22], [43, 50]]
	a := []float16.Float16{
		float16.FromFloat32(1), float16.FromFloat32(2),
		float16.FromFloat32(3), float16.FromFloat32(4),
	}
	b := []float16.Float16{
		float16.FromFloat32(5), float16.FromFloat32(6),
		float16.FromFloat32(7), float16.FromFloat32(8),
	}
	c := make([]float16.Float16, 4)

	GemmF16(2, 2, 2, a, b, c)

	want := []float32{19, 22, 43, 50}
	for i, v := range c {
		got := v.ToFloat32()
		if math.Abs(float64(got-want[i])) > 0.5 {
			t.Errorf("c[%d] = %v, want %v", i, got, want[i])
		}
	}
}

func TestGemmF16_Identity(t *testing.T) {
	a := []float16.Float16{
		float16.FromFloat32(1), float16.FromFloat32(0),
		float16.FromFloat32(0), float16.FromFloat32(1),
	}
	b := []float16.Float16{
		float16.FromFloat32(3), float16.FromFloat32(4),
		float16.FromFloat32(5), float16.FromFloat32(6),
	}
	c := make([]float16.Float16, 4)

	GemmF16(2, 2, 2, a, b, c)

	want := []float32{3, 4, 5, 6}
	for i, v := range c {
		got := v.ToFloat32()
		if math.Abs(float64(got-want[i])) > 0.5 {
			t.Errorf("c[%d] = %v, want %v", i, got, want[i])
		}
	}
}

func TestGemmF8_KnownProduct(t *testing.T) {
	// A = [[1, 2], [3, 4]]
	// B = [[5, 6], [7, 8]]
	// C = [[19, 22], [43, 50]]
	a := []float8.Float8{
		float8.ToFloat8(1), float8.ToFloat8(2),
		float8.ToFloat8(3), float8.ToFloat8(4),
	}
	b := []float8.Float8{
		float8.ToFloat8(5), float8.ToFloat8(6),
		float8.ToFloat8(7), float8.ToFloat8(8),
	}
	c := make([]float8.Float8, 4)

	GemmF8(2, 2, 2, a, b, c)

	want := []float32{19, 22, 43, 50}
	for i, v := range c {
		got := v.ToFloat32()
		// Float8 has very limited precision, allow generous tolerance
		if math.Abs(float64(got-want[i])) > 10 {
			t.Errorf("c[%d] = %v, want ~%v", i, got, want[i])
		}
	}
}

func TestGemmF8_Identity(t *testing.T) {
	a := []float8.Float8{
		float8.ToFloat8(1), float8.ToFloat8(0),
		float8.ToFloat8(0), float8.ToFloat8(1),
	}
	b := []float8.Float8{
		float8.ToFloat8(3), float8.ToFloat8(4),
		float8.ToFloat8(5), float8.ToFloat8(6),
	}
	c := make([]float8.Float8, 4)

	GemmF8(2, 2, 2, a, b, c)

	want := []float32{3, 4, 5, 6}
	for i, v := range c {
		got := v.ToFloat32()
		if math.Abs(float64(got-want[i])) > 2 {
			t.Errorf("c[%d] = %v, want ~%v", i, got, want[i])
		}
	}
}
