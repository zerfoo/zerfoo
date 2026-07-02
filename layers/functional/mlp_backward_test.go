package functional

import (
	"context"
	"math"
	"math/rand"
	"testing"

	"github.com/zerfoo/ztensor/tensor"
)

// mlpForwardRef computes y = Linear2(activation(Linear1(x))) using raw slices for numerical gradient checking.
func mlpForwardRef(x, w1, b1, w2, b2 []float64, inF, hidF, outF int, activation string) []float64 {
	batch := len(x) / inF

	// Linear1: hidden = x @ w1^T + b1
	hidden := make([]float64, batch*hidF)
	for i := 0; i < batch; i++ {
		for j := 0; j < hidF; j++ {
			sum := b1[j]
			for k := 0; k < inF; k++ {
				sum += x[i*inF+k] * w1[j*inF+k]
			}
			hidden[i*hidF+j] = sum
		}
	}

	// Activation
	activated := make([]float64, batch*hidF)
	for i, v := range hidden {
		switch activation {
		case "relu":
			if v > 0 {
				activated[i] = v
			}
		case "gelu":
			activated[i] = geluRef(v)
		}
	}

	// Linear2: out = activated @ w2^T + b2
	out := make([]float64, batch*outF)
	for i := 0; i < batch; i++ {
		for j := 0; j < outF; j++ {
			sum := b2[j]
			for k := 0; k < hidF; k++ {
				sum += activated[i*hidF+k] * w2[j*hidF+k]
			}
			out[i*outF+j] = sum
		}
	}
	return out
}

// numericalGradient computes df/d(param[idx]) via central differences.
func numericalGradient(param []float64, idx int, f func() float64, eps float64) float64 {
	orig := param[idx]
	param[idx] = orig + eps
	fPlus := f()
	param[idx] = orig - eps
	fMinus := f()
	param[idx] = orig
	return (fPlus - fMinus) / (2 * eps)
}

func TestMLPBackward(t *testing.T) {
	ctx := context.Background()

	type testCase struct {
		name       string
		batch      int
		inF        int
		hidF       int
		outF       int
		activation string
	}

	cases := []testCase{
		{"relu_4x8_h16", 4, 8, 16, 8, "relu"},
		{"gelu_4x8_h16", 4, 8, 16, 8, "gelu"},
		{"relu_8x16_h32", 8, 16, 32, 16, "relu"},
		{"gelu_8x16_h32", 8, 16, 32, 16, "gelu"},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			engine, ops := newF64Engine()
			rng := rand.New(rand.NewSource(42))

			eps := 1e-5
			tol := 1e-4

			// Initialize parameters
			initSlice := func(n int) []float64 {
				s := make([]float64, n)
				for i := range s {
					s[i] = rng.NormFloat64() * 0.1
				}
				return s
			}

			xData := initSlice(tc.batch * tc.inF)
			w1Data := initSlice(tc.hidF * tc.inF)
			b1Data := initSlice(tc.hidF)
			w2Data := initSlice(tc.outF * tc.hidF)
			b2Data := initSlice(tc.outF)

			// Compute forward pass to get hidden and activated
			hidden := make([]float64, tc.batch*tc.hidF)
			for i := 0; i < tc.batch; i++ {
				for j := 0; j < tc.hidF; j++ {
					sum := b1Data[j]
					for k := 0; k < tc.inF; k++ {
						sum += xData[i*tc.inF+k] * w1Data[j*tc.inF+k]
					}
					hidden[i*tc.hidF+j] = sum
				}
			}

			activated := make([]float64, tc.batch*tc.hidF)
			for i, v := range hidden {
				switch tc.activation {
				case "relu":
					if v > 0 {
						activated[i] = v
					}
				case "gelu":
					activated[i] = geluRef(v)
				}
			}

			// dOutput = ones (so loss = sum of outputs)
			dOutData := make([]float64, tc.batch*tc.outF)
			for i := range dOutData {
				dOutData[i] = 1.0
			}

			// Build tensors
			mkT := func(shape []int, data []float64) *tensor.TensorNumeric[float64] {
				return makeTensor(t, shape, data)
			}

			dOutput := mkT([]int{tc.batch, tc.outF}, dOutData)
			input := mkT([]int{tc.batch, tc.inF}, xData)
			weight1 := mkT([]int{tc.hidF, tc.inF}, w1Data)
			bias1 := mkT([]int{tc.hidF}, b1Data)
			weight2 := mkT([]int{tc.outF, tc.hidF}, w2Data)
			bias2 := mkT([]int{tc.outF}, b2Data)
			hiddenT := mkT([]int{tc.batch, tc.hidF}, hidden)
			activatedT := mkT([]int{tc.batch, tc.hidF}, activated)

			// Compute analytical gradients
			dInput, dWeight1, dBias1, dWeight2, dBias2, err := MLPBackward(ctx, engine, ops,
				dOutput, input, weight1, bias1, weight2, bias2, hiddenT, activatedT, tc.activation)
			if err != nil {
				t.Fatalf("MLPBackward: %v", err)
			}

			// Loss = sum of MLP outputs
			loss := func() float64 {
				out := mlpForwardRef(xData, w1Data, b1Data, w2Data, b2Data,
					tc.inF, tc.hidF, tc.outF, tc.activation)
				sum := 0.0
				for _, v := range out {
					sum += v
				}
				return sum
			}

			// Check dInput
			checkGrad := func(name string, param, grad []float64) {
				t.Helper()
				for i := range param {
					num := numericalGradient(param, i, loss, eps)
					diff := math.Abs(grad[i] - num)
					denom := math.Max(math.Abs(grad[i])+math.Abs(num), 1e-8)
					relErr := diff / denom
					if relErr > tol && diff > 1e-7 {
						t.Errorf("%s[%d]: analytical=%.8f, numerical=%.8f, relErr=%.2e",
							name, i, grad[i], num, relErr)
					}
				}
			}

			checkGrad("dInput", xData, dInput.Data())
			checkGrad("dWeight1", w1Data, dWeight1.Data())
			checkGrad("dBias1", b1Data, dBias1.Data())
			checkGrad("dWeight2", w2Data, dWeight2.Data())
			checkGrad("dBias2", b2Data, dBias2.Data())
		})
	}
}

func TestMLPBackward_Shapes(t *testing.T) {
	ctx := context.Background()
	engine, ops := newF64Engine()

	batch, inF, hidF, outF := 4, 8, 16, 8

	zeros := func(n int) []float64 {
		return make([]float64, n)
	}

	mkT := func(shape []int, data []float64) *tensor.TensorNumeric[float64] {
		return makeTensor(t, shape, data)
	}

	dOutput := mkT([]int{batch, outF}, zeros(batch*outF))
	input := mkT([]int{batch, inF}, zeros(batch*inF))
	weight1 := mkT([]int{hidF, inF}, zeros(hidF*inF))
	bias1 := mkT([]int{hidF}, zeros(hidF))
	weight2 := mkT([]int{outF, hidF}, zeros(outF*hidF))
	bias2 := mkT([]int{outF}, zeros(outF))
	hidden := mkT([]int{batch, hidF}, zeros(batch*hidF))
	activated := mkT([]int{batch, hidF}, zeros(batch*hidF))

	dInput, dWeight1, dBias1, dWeight2, dBias2, err := MLPBackward(ctx, engine, ops,
		dOutput, input, weight1, bias1, weight2, bias2, hidden, activated, "relu")
	if err != nil {
		t.Fatalf("MLPBackward: %v", err)
	}

	assertShape := func(name string, got *tensor.TensorNumeric[float64], want []int) {
		t.Helper()
		s := got.Shape()
		if len(s) != len(want) {
			t.Fatalf("%s shape = %v, want %v", name, s, want)
		}
		for i := range want {
			if s[i] != want[i] {
				t.Fatalf("%s shape = %v, want %v", name, s, want)
			}
		}
	}

	assertShape("dInput", dInput, []int{batch, inF})
	assertShape("dWeight1", dWeight1, []int{hidF, inF})
	assertShape("dBias1", dBias1, []int{hidF})
	assertShape("dWeight2", dWeight2, []int{outF, hidF})
	assertShape("dBias2", dBias2, []int{outF})
}

func TestMLPBackward_UnsupportedActivation(t *testing.T) {
	ctx := context.Background()
	engine, ops := newF64Engine()

	mkT := func(shape []int) *tensor.TensorNumeric[float64] {
		return makeTensor(t, shape, make([]float64, 1))
	}

	_, _, _, _, _, err := MLPBackward(ctx, engine, ops,
		mkT([]int{1, 1}), mkT([]int{1, 1}), mkT([]int{1, 1}), mkT([]int{1}),
		mkT([]int{1, 1}), mkT([]int{1}), mkT([]int{1, 1}), mkT([]int{1, 1}),
		"tanh")
	if err == nil {
		t.Fatal("expected error for unsupported activation")
	}
}

func TestMLPBackward_NilInputs(t *testing.T) {
	ctx := context.Background()
	engine, ops := newF64Engine()

	mkT := func(shape []int) *tensor.TensorNumeric[float64] {
		return makeTensor(t, shape, make([]float64, 1))
	}
	dummy := mkT([]int{1, 1})
	dummyBias := mkT([]int{1})

	cases := []struct {
		name string
		args [8]*tensor.TensorNumeric[float64]
	}{
		{"nil_dOutput", [8]*tensor.TensorNumeric[float64]{nil, dummy, dummy, dummyBias, dummy, dummyBias, dummy, dummy}},
		{"nil_input", [8]*tensor.TensorNumeric[float64]{dummy, nil, dummy, dummyBias, dummy, dummyBias, dummy, dummy}},
		{"nil_weight1", [8]*tensor.TensorNumeric[float64]{dummy, dummy, nil, dummyBias, dummy, dummyBias, dummy, dummy}},
		{"nil_weight2", [8]*tensor.TensorNumeric[float64]{dummy, dummy, dummy, dummyBias, nil, dummyBias, dummy, dummy}},
		{"nil_hidden", [8]*tensor.TensorNumeric[float64]{dummy, dummy, dummy, dummyBias, dummy, dummyBias, nil, dummy}},
		{"nil_activated", [8]*tensor.TensorNumeric[float64]{dummy, dummy, dummy, dummyBias, dummy, dummyBias, dummy, nil}},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, _, _, _, _, err := MLPBackward(ctx, engine, ops,
				tc.args[0], tc.args[1], tc.args[2], tc.args[3],
				tc.args[4], tc.args[5], tc.args[6], tc.args[7],
				"relu")
			if err == nil {
				t.Fatal("expected error for nil input")
			}
		})
	}
}
