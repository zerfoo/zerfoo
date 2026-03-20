package core

import (
	"context"
	"math"
	"testing"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/ztensor/types"
)

func TestPowBackward(t *testing.T) {
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	ctx := context.Background()

	tests := []struct {
		name     string
		base     []float32
		exp      []float32
		dOut     []float32
		wantBase []float32 // expected gradient w.r.t. base
		wantExp  []float32 // expected gradient w.r.t. exponent
	}{
		{
			name:     "2^3",
			base:     []float32{2},
			exp:      []float32{3},
			dOut:     []float32{1},
			wantBase: []float32{12},                              // 3 * 2^2 = 12
			wantExp:  []float32{float32(8 * math.Log(2))},       // 2^3 * ln(2)
		},
		{
			name:     "3^2",
			base:     []float32{3},
			exp:      []float32{2},
			dOut:     []float32{1},
			wantBase: []float32{6},                               // 2 * 3^1 = 6
			wantExp:  []float32{float32(9 * math.Log(3))},       // 3^2 * ln(3)
		},
		{
			name:     "0.5^2",
			base:     []float32{0.5},
			exp:      []float32{2},
			dOut:     []float32{1},
			wantBase: []float32{1},                               // 2 * 0.5^1 = 1
			wantExp:  []float32{float32(0.25 * math.Log(0.5))},  // 0.5^2 * ln(0.5)
		},
		{
			name:     "base=1 (grad should be n)",
			base:     []float32{1},
			exp:      []float32{5},
			dOut:     []float32{1},
			wantBase: []float32{5},  // 5 * 1^4 = 5
			wantExp:  []float32{0},  // 1^5 * ln(1) = 0
		},
		{
			name:     "exponent=1 (grad should be 1)",
			base:     []float32{4},
			exp:      []float32{1},
			dOut:     []float32{1},
			wantBase: []float32{1},                               // 1 * 4^0 = 1
			wantExp:  []float32{float32(4 * math.Log(4))},       // 4^1 * ln(4)
		},
		{
			name:     "exponent=0 (grad should be 0)",
			base:     []float32{3},
			exp:      []float32{0},
			dOut:     []float32{1},
			wantBase: []float32{0},                               // 0 * 3^(-1) = 0
			wantExp:  []float32{float32(1 * math.Log(3))},       // 3^0 * ln(3) = ln(3)
		},
		{
			name:     "multi-element",
			base:     []float32{2, 3},
			exp:      []float32{3, 2},
			dOut:     []float32{1, 1},
			wantBase: []float32{12, 6},
			wantExp:  []float32{float32(8 * math.Log(2)), float32(9 * math.Log(3))},
		},
		{
			name:     "upstream gradient scaling",
			base:     []float32{2},
			exp:      []float32{3},
			dOut:     []float32{0.5},
			wantBase: []float32{6},                                    // 0.5 * 12 = 6
			wantExp:  []float32{float32(0.5 * 8 * math.Log(2))},     // 0.5 * 2^3 * ln(2)
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			shape := []int{len(tc.base)}

			baseTensor, err := tensor.New[float32](shape, tc.base)
			if err != nil {
				t.Fatalf("creating base tensor: %v", err)
			}

			expTensor, err := tensor.New[float32](shape, tc.exp)
			if err != nil {
				t.Fatalf("creating exp tensor: %v", err)
			}

			dOutTensor, err := tensor.New[float32](shape, tc.dOut)
			if err != nil {
				t.Fatalf("creating dOut tensor: %v", err)
			}

			pow := NewPow[float32](engine)

			grads, err := pow.Backward(ctx, types.FullBackprop, dOutTensor, baseTensor, expTensor)
			if err != nil {
				t.Fatalf("Backward returned error: %v", err)
			}

			if len(grads) != 2 {
				t.Fatalf("expected 2 gradients, got %d", len(grads))
			}

			const tol = 1e-5

			gradBase := grads[0].Data()
			for i, want := range tc.wantBase {
				if diff := math.Abs(float64(gradBase[i] - want)); diff > float64(tol) {
					t.Errorf("gradBase[%d] = %v, want %v (diff=%v)", i, gradBase[i], want, diff)
				}
			}

			gradExp := grads[1].Data()
			for i, want := range tc.wantExp {
				if diff := math.Abs(float64(gradExp[i] - want)); diff > float64(tol) {
					t.Errorf("gradExp[%d] = %v, want %v (diff=%v)", i, gradExp[i], want, diff)
				}
			}
		})
	}
}

func TestPowBackwardNumericalGradient(t *testing.T) {
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	ctx := context.Background()

	bases := []float32{1.5, 2.0, 3.0, 0.5}
	exps := []float32{2.0, 3.0, 0.5, 2.0}

	const h = 1e-4
	const tol = 1e-2

	for i := range bases {
		t.Run("numerical_check", func(t *testing.T) {
			b := bases[i]
			e := exps[i]

			shape := []int{1}
			ones := []float32{1}

			baseTensor, _ := tensor.New[float32](shape, []float32{b})
			expTensor, _ := tensor.New[float32](shape, []float32{e})
			dOutTensor, _ := tensor.New[float32](shape, ones)

			pow := NewPow[float32](engine)
			grads, err := pow.Backward(ctx, types.FullBackprop, dOutTensor, baseTensor, expTensor)
			if err != nil {
				t.Fatalf("Backward error: %v", err)
			}

			// Numerical gradient w.r.t. base: (f(b+h, e) - f(b-h, e)) / (2h)
			bPlusH, _ := tensor.New[float32](shape, []float32{b + h})
			bMinusH, _ := tensor.New[float32](shape, []float32{b - h})

			fPlus, _ := engine.Pow(ctx, bPlusH, expTensor)
			fMinus, _ := engine.Pow(ctx, bMinusH, expTensor)

			numericalGradBase := (fPlus.Data()[0] - fMinus.Data()[0]) / (2 * h)
			analyticalGradBase := grads[0].Data()[0]

			if diff := math.Abs(float64(analyticalGradBase - numericalGradBase)); diff > tol {
				t.Errorf("base grad: analytical=%v, numerical=%v, diff=%v (base=%v, exp=%v)",
					analyticalGradBase, numericalGradBase, diff, b, e)
			}

			// Numerical gradient w.r.t. exponent: (f(b, e+h) - f(b, e-h)) / (2h)
			ePlusH, _ := tensor.New[float32](shape, []float32{e + h})
			eMinusH, _ := tensor.New[float32](shape, []float32{e - h})

			fPlusE, _ := engine.Pow(ctx, baseTensor, ePlusH)
			fMinusE, _ := engine.Pow(ctx, baseTensor, eMinusH)

			numericalGradExp := (fPlusE.Data()[0] - fMinusE.Data()[0]) / (2 * h)
			analyticalGradExp := grads[1].Data()[0]

			if diff := math.Abs(float64(analyticalGradExp - numericalGradExp)); diff > tol {
				t.Errorf("exp grad: analytical=%v, numerical=%v, diff=%v (base=%v, exp=%v)",
					analyticalGradExp, numericalGradExp, diff, b, e)
			}
		})
	}
}

func TestPowBackwardInputValidation(t *testing.T) {
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	ctx := context.Background()

	pow := NewPow[float32](engine)
	dOut, _ := tensor.New[float32]([]int{1}, []float32{1})
	input, _ := tensor.New[float32]([]int{1}, []float32{2})

	_, err := pow.Backward(ctx, types.FullBackprop, dOut, input)
	if err == nil {
		t.Error("expected error with 1 input, got nil")
	}

	_, err = pow.Backward(ctx, types.FullBackprop, dOut, input, input, input)
	if err == nil {
		t.Error("expected error with 3 inputs, got nil")
	}
}
