package functional_test

import (
	"context"
	"math"
	"math/rand/v2"
	"testing"

	"github.com/zerfoo/zerfoo/layers/functional"
	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
)

func TestLayerNormBackward_NumericalGradient(t *testing.T) {
	ctx := context.Background()
	engine := compute.NewCPUEngine[float64](&numeric.Float64Ops{})

	shapes := []struct {
		name string
		rows int
		cols int
	}{
		{"4x8", 4, 8},
		{"1x16", 1, 16},
		{"8x32", 8, 32},
	}

	eps := 1e-5
	delta := 1e-5
	tol := 1e-4

	rng := rand.New(rand.NewPCG(42, 0))

	for _, tc := range shapes {
		t.Run(tc.name, func(t *testing.T) {
			n := tc.rows * tc.cols

			inputData := make([]float64, n)
			scaleData := make([]float64, tc.cols)
			biasData := make([]float64, tc.cols)
			dOutData := make([]float64, n)

			for i := range inputData {
				inputData[i] = rng.NormFloat64()
			}
			for i := range scaleData {
				scaleData[i] = 0.5 + rng.Float64()
			}
			for i := range biasData {
				biasData[i] = rng.NormFloat64() * 0.1
			}
			for i := range dOutData {
				dOutData[i] = rng.NormFloat64()
			}

			input, _ := tensor.New[float64]([]int{tc.rows, tc.cols}, inputData)
			scale, _ := tensor.New[float64]([]int{tc.cols}, scaleData)
			bias, _ := tensor.New[float64]([]int{tc.cols}, biasData)
			dOut, _ := tensor.New[float64]([]int{tc.rows, tc.cols}, dOutData)

			// Analytic gradients
			dInput, dScale, dBias, err := functional.LayerNormBackward(ctx, engine, dOut, input, scale, eps)
			assertNoErr(t, err)

			// --- Numerical gradient for dInput ---
			t.Run("dInput", func(t *testing.T) {
				numGrad := make([]float64, n)
				for i := 0; i < n; i++ {
					pertPlus := make([]float64, n)
					pertMinus := make([]float64, n)
					copy(pertPlus, inputData)
					copy(pertMinus, inputData)
					pertPlus[i] += delta
					pertMinus[i] -= delta

					xp, _ := tensor.New[float64]([]int{tc.rows, tc.cols}, pertPlus)
					xm, _ := tensor.New[float64]([]int{tc.rows, tc.cols}, pertMinus)

					outP, err := functional.LayerNorm(ctx, engine, xp, scale, bias, eps)
					assertNoErr(t, err)
					outM, err := functional.LayerNorm(ctx, engine, xm, scale, bias, eps)
					assertNoErr(t, err)

					// loss = sum(dOut * out), so dLoss/dx_i = sum(dOut * dout/dx_i)
					dp := outP.Data()
					dm := outM.Data()
					var grad float64
					for j := 0; j < n; j++ {
						grad += dOutData[j] * (dp[j] - dm[j]) / (2 * delta)
					}
					numGrad[i] = grad
				}
				assertF64Close(t, numGrad, dInput.Data(), tol)
			})

			// --- Numerical gradient for dScale ---
			t.Run("dScale", func(t *testing.T) {
				numGrad := make([]float64, tc.cols)
				for i := 0; i < tc.cols; i++ {
					pertPlus := make([]float64, tc.cols)
					pertMinus := make([]float64, tc.cols)
					copy(pertPlus, scaleData)
					copy(pertMinus, scaleData)
					pertPlus[i] += delta
					pertMinus[i] -= delta

					sp, _ := tensor.New[float64]([]int{tc.cols}, pertPlus)
					sm, _ := tensor.New[float64]([]int{tc.cols}, pertMinus)

					outP, err := functional.LayerNorm(ctx, engine, input, sp, bias, eps)
					assertNoErr(t, err)
					outM, err := functional.LayerNorm(ctx, engine, input, sm, bias, eps)
					assertNoErr(t, err)

					dp := outP.Data()
					dm := outM.Data()
					var grad float64
					for j := 0; j < n; j++ {
						grad += dOutData[j] * (dp[j] - dm[j]) / (2 * delta)
					}
					numGrad[i] = grad
				}
				assertF64Close(t, numGrad, dScale.Data(), tol)
			})

			// --- Numerical gradient for dBias ---
			t.Run("dBias", func(t *testing.T) {
				numGrad := make([]float64, tc.cols)
				for i := 0; i < tc.cols; i++ {
					pertPlus := make([]float64, tc.cols)
					pertMinus := make([]float64, tc.cols)
					copy(pertPlus, biasData)
					copy(pertMinus, biasData)
					pertPlus[i] += delta
					pertMinus[i] -= delta

					bp, _ := tensor.New[float64]([]int{tc.cols}, pertPlus)
					bm, _ := tensor.New[float64]([]int{tc.cols}, pertMinus)

					outP, err := functional.LayerNorm(ctx, engine, input, scale, bp, eps)
					assertNoErr(t, err)
					outM, err := functional.LayerNorm(ctx, engine, input, scale, bm, eps)
					assertNoErr(t, err)

					dp := outP.Data()
					dm := outM.Data()
					var grad float64
					for j := 0; j < n; j++ {
						grad += dOutData[j] * (dp[j] - dm[j]) / (2 * delta)
					}
					numGrad[i] = grad
				}
				assertF64Close(t, numGrad, dBias.Data(), tol)
			})
		})
	}
}

func TestLayerNormBackward_DBiasIsSum(t *testing.T) {
	ctx := context.Background()
	engine := compute.NewCPUEngine[float64](&numeric.Float64Ops{})

	rows, cols := 4, 8
	n := rows * cols

	rng := rand.New(rand.NewPCG(99, 0))
	inputData := make([]float64, n)
	dOutData := make([]float64, n)
	scaleData := make([]float64, cols)
	for i := range inputData {
		inputData[i] = rng.NormFloat64()
	}
	for i := range dOutData {
		dOutData[i] = rng.NormFloat64()
	}
	for i := range scaleData {
		scaleData[i] = 1.0
	}

	input, _ := tensor.New[float64]([]int{rows, cols}, inputData)
	dOut, _ := tensor.New[float64]([]int{rows, cols}, dOutData)
	scale, _ := tensor.New[float64]([]int{cols}, scaleData)

	_, _, dBias, err := functional.LayerNormBackward(ctx, engine, dOut, input, scale, 1e-5)
	assertNoErr(t, err)

	// dBias should equal column-wise sum of dOutput
	expected := make([]float64, cols)
	for r := 0; r < rows; r++ {
		for c := 0; c < cols; c++ {
			expected[c] += dOutData[r*cols+c]
		}
	}

	got := dBias.Data()
	for i := range expected {
		d := math.Abs(expected[i] - got[i])
		if d > 1e-10 {
			t.Errorf("dBias[%d]: want %v, got %v (diff %v)", i, expected[i], got[i], d)
		}
	}
}

func TestLayerNormBackward_NilInputs(t *testing.T) {
	ctx := context.Background()
	engine := compute.NewCPUEngine[float64](&numeric.Float64Ops{})

	x, _ := tensor.New[float64]([]int{2, 4}, make([]float64, 8))
	s, _ := tensor.New[float64]([]int{4}, []float64{1, 1, 1, 1})

	tests := []struct {
		name   string
		dOut   *tensor.TensorNumeric[float64]
		input  *tensor.TensorNumeric[float64]
		scale  *tensor.TensorNumeric[float64]
	}{
		{"nil dOutput", nil, x, s},
		{"nil input", x, nil, s},
		{"nil scale", x, x, nil},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			_, _, _, err := functional.LayerNormBackward(ctx, engine, tc.dOut, tc.input, tc.scale, 1e-5)
			if err == nil {
				t.Fatal("expected error for nil input, got nil")
			}
		})
	}
}
