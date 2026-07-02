package functional

import (
	"context"
	"math"
	"testing"
)

// geluPrimeRef computes the analytical derivative of the tanh-approximated GELU.
func geluPrimeRef(x float64) float64 {
	a := math.Sqrt(2 / math.Pi)
	b := 0.044715
	u := a * (x + b*x*x*x)
	t := math.Tanh(u)
	dudx := a * (1 + 3*b*x*x)
	return 0.5*(1+t) + 0.5*x*(1-t*t)*dudx
}

func TestGELUBackward(t *testing.T) {
	ctx := context.Background()

	t.Run("numerical_gradient_check_float64", func(t *testing.T) {
		engine, ops := newF64Engine()
		eps := 1e-5
		tol := 1e-4

		vals := []float64{-3, -2, -1, -0.5, 0, 0.5, 1, 2, 3}
		input := makeTensor(t, []int{len(vals)}, vals)

		// dOutput = all ones for gradient check
		ones := make([]float64, len(vals))
		for i := range ones {
			ones[i] = 1.0
		}
		dOutput := makeTensor(t, []int{len(vals)}, ones)

		got, err := GELUBackward(ctx, engine, ops, dOutput, input)
		if err != nil {
			t.Fatalf("GELUBackward: %v", err)
		}

		for i, x := range vals {
			// Numerical gradient: (GELU(x+eps) - GELU(x-eps)) / (2*eps)
			numerical := (geluRef(x+eps) - geluRef(x-eps)) / (2 * eps)
			analytical := got.Data()[i]
			if math.Abs(analytical-numerical) > tol {
				t.Errorf("x=%.2f: analytical=%.8f, numerical=%.8f, diff=%.2e",
					x, analytical, numerical, math.Abs(analytical-numerical))
			}
		}
	})

	t.Run("known_values", func(t *testing.T) {
		engine, ops := newF64Engine()

		vals := []float64{0, 5, -5}
		input := makeTensor(t, []int{len(vals)}, vals)
		ones := make([]float64, len(vals))
		for i := range ones {
			ones[i] = 1.0
		}
		dOutput := makeTensor(t, []int{len(vals)}, ones)

		got, err := GELUBackward(ctx, engine, ops, dOutput, input)
		if err != nil {
			t.Fatalf("GELUBackward: %v", err)
		}

		data := got.Data()

		// At x=0, GELU'(0) = 0.5
		if math.Abs(data[0]-0.5) > 1e-10 {
			t.Errorf("GELU'(0) = %v, want 0.5", data[0])
		}

		// At large positive x, GELU'(x) ≈ 1.0
		if math.Abs(data[1]-1.0) > 1e-3 {
			t.Errorf("GELU'(5) = %v, want ≈1.0", data[1])
		}

		// At large negative x, GELU'(x) ≈ 0.0
		if math.Abs(data[2]) > 1e-3 {
			t.Errorf("GELU'(-5) = %v, want ≈0.0", data[2])
		}
	})

	t.Run("shape_4x8", func(t *testing.T) {
		engine, ops := newF64Engine()

		n := 4 * 8
		vals := make([]float64, n)
		ones := make([]float64, n)
		for i := range vals {
			vals[i] = float64(i-16) * 0.25
			ones[i] = 1.0
		}
		input := makeTensor(t, []int{4, 8}, vals)
		dOutput := makeTensor(t, []int{4, 8}, ones)

		got, err := GELUBackward(ctx, engine, ops, dOutput, input)
		if err != nil {
			t.Fatalf("GELUBackward: %v", err)
		}

		// Verify shape
		shape := got.Shape()
		if len(shape) != 2 || shape[0] != 4 || shape[1] != 8 {
			t.Fatalf("shape = %v, want [4, 8]", shape)
		}

		// Verify against reference
		for i, x := range vals {
			want := geluPrimeRef(x)
			if math.Abs(got.Data()[i]-want) > 1e-10 {
				t.Errorf("[%d] x=%.2f: got %v, want %v", i, x, got.Data()[i], want)
			}
		}
	})

	t.Run("float32", func(t *testing.T) {
		engine, ops := newF32Engine()

		vals := []float32{-3, -1, 0, 1, 3}
		input := makeTensor(t, []int{len(vals)}, vals)
		ones := make([]float32, len(vals))
		for i := range ones {
			ones[i] = 1.0
		}
		dOutput := makeTensor(t, []int{len(vals)}, ones)

		got, err := GELUBackward(ctx, engine, ops, dOutput, input)
		if err != nil {
			t.Fatalf("GELUBackward: %v", err)
		}

		for i, x := range vals {
			want := geluPrimeRef(float64(x))
			if math.Abs(float64(got.Data()[i])-want) > 1e-4 {
				t.Errorf("x=%.2f: got %v, want %v", x, got.Data()[i], want)
			}
		}
	})
}
