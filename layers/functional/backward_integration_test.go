package functional

import (
	"context"
	"math"
	"testing"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
)

// TestLinearGELUChainBackward verifies that composing LinearBackward and
// GELUBackward produces gradients matching a numerical gradient check over
// the composed forward: f(x) = GELU(Linear(x)).
func TestLinearGELUChainBackward(t *testing.T) {
	ctx := context.Background()
	engine, ops := newF64Engine()

	batch, in, out := 4, 6, 4
	inputData := deterministicFill(batch*in, 7, 3)
	weightData := deterministicFill(out*in, 11, 5)
	biasData := deterministicFill(out, 5, 2)
	dOutputData := deterministicFill(batch*out, 9, 4)

	input := mustTensor(t, []int{batch, in}, inputData)
	weight := mustTensor(t, []int{out, in}, weightData)
	bias := mustTensor(t, []int{out}, biasData)
	dOutput := mustTensor(t, []int{batch, out}, dOutputData)

	// Forward: linear then GELU
	linear, err := Linear(ctx, engine, input, weight, bias)
	if err != nil {
		t.Fatal(err)
	}

	// Backward: GELU then linear
	dLinear, err := GELUBackward(ctx, engine, ops, dOutput, linear)
	if err != nil {
		t.Fatal(err)
	}
	dInput, dWeight, dBias, err := LinearBackward(ctx, engine, dLinear, input, weight)
	if err != nil {
		t.Fatal(err)
	}

	// Numerical gradient check for dInput
	eps := 1e-5
	tol := 1e-4

	t.Run("dInput", func(t *testing.T) {
		for i := range inputData {
			ng := numericalGradLinearGELU(ctx, engine, ops,
				inputData, weightData, biasData, dOutputData,
				[]int{batch, in}, []int{out, in}, []int{out},
				"input", i, eps)
			if math.Abs(dInput.Data()[i]-ng) > tol {
				t.Errorf("dInput[%d]: analytical=%f, numerical=%f, diff=%.2e",
					i, dInput.Data()[i], ng, math.Abs(dInput.Data()[i]-ng))
			}
		}
	})

	t.Run("dWeight", func(t *testing.T) {
		for i := range weightData {
			ng := numericalGradLinearGELU(ctx, engine, ops,
				inputData, weightData, biasData, dOutputData,
				[]int{batch, in}, []int{out, in}, []int{out},
				"weight", i, eps)
			if math.Abs(dWeight.Data()[i]-ng) > tol {
				t.Errorf("dWeight[%d]: analytical=%f, numerical=%f, diff=%.2e",
					i, dWeight.Data()[i], ng, math.Abs(dWeight.Data()[i]-ng))
			}
		}
	})

	t.Run("dBias", func(t *testing.T) {
		for i := range biasData {
			ng := numericalGradLinearGELU(ctx, engine, ops,
				inputData, weightData, biasData, dOutputData,
				[]int{batch, in}, []int{out, in}, []int{out},
				"bias", i, eps)
			if math.Abs(dBias.Data()[i]-ng) > tol {
				t.Errorf("dBias[%d]: analytical=%f, numerical=%f, diff=%.2e",
					i, dBias.Data()[i], ng, math.Abs(dBias.Data()[i]-ng))
			}
		}
	})
}

// TestLinearGELUChainBackwardFloat32 verifies generics work with float32.
func TestLinearGELUChainBackwardFloat32(t *testing.T) {
	ctx := context.Background()
	engine, ops := newF32Engine()

	batch, in, out := 2, 4, 3
	inputData := deterministicFillF32(batch*in, 7, 3)
	weightData := deterministicFillF32(out*in, 11, 5)
	biasData := deterministicFillF32(out, 5, 2)
	dOutputData := deterministicFillF32(batch*out, 9, 4)

	input := mustTensor(t, []int{batch, in}, inputData)
	weight := mustTensor(t, []int{out, in}, weightData)
	bias := mustTensor(t, []int{out}, biasData)
	dOutput := mustTensor(t, []int{batch, out}, dOutputData)

	// Forward
	linear, err := Linear(ctx, engine, input, weight, bias)
	if err != nil {
		t.Fatal(err)
	}

	// Backward
	dLinear, err := GELUBackward(ctx, engine, ops, dOutput, linear)
	if err != nil {
		t.Fatal(err)
	}
	dInput, _, _, err := LinearBackward(ctx, engine, dLinear, input, weight)
	if err != nil {
		t.Fatal(err)
	}

	// Smoke check: shape and no NaN
	assertShapeGeneric(t, "dInput", dInput, []int{batch, in})
	for i, v := range dInput.Data() {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Errorf("dInput[%d] = %v (NaN or Inf)", i, v)
		}
	}
}

// TestEncoderLayerBackward simulates a full encoder layer backward:
// Forward: LayerNorm1 -> Linear(Qkv) -> (skip MHA details, use identity) -> residual
//          -> LayerNorm2 -> Linear1 -> GELU -> Linear2 -> residual
// Then backward through the reverse chain and verify dInput via numerical gradient.
func TestEncoderLayerBackward(t *testing.T) {
	ctx := context.Background()
	engine, ops := newF64Engine()

	batch, features := 2, 4
	hidden := 8

	inputData := deterministicFill(batch*features, 13, 3)
	// LayerNorm1 params
	ln1ScaleData := deterministicFill(features, 3, 1)
	ln1BiasData := deterministicFill(features, 7, 2)
	// FFN weights
	w1Data := deterministicFill(hidden*features, 11, 5)
	b1Data := deterministicFill(hidden, 5, 2)
	w2Data := deterministicFill(features*hidden, 7, 3)
	b2Data := deterministicFill(features, 3, 1)
	// LayerNorm2 params
	ln2ScaleData := deterministicFill(features, 9, 4)
	ln2BiasData := deterministicFill(features, 11, 5)
	// Upstream gradient
	dOutputData := deterministicFill(batch*features, 17, 7)

	epsVal := 1e-5

	// Composed forward function for numerical gradient
	composedForward := func(inD []float64) float64 {
		in, _ := tensor.New[float64]([]int{batch, features}, inD)
		ln1Scale, _ := tensor.New[float64]([]int{features}, ln1ScaleData)
		ln1Bias, _ := tensor.New[float64]([]int{features}, ln1BiasData)
		w1, _ := tensor.New[float64]([]int{hidden, features}, w1Data)
		b1, _ := tensor.New[float64]([]int{hidden}, b1Data)
		w2, _ := tensor.New[float64]([]int{features, hidden}, w2Data)
		b2, _ := tensor.New[float64]([]int{features}, b2Data)
		ln2Scale, _ := tensor.New[float64]([]int{features}, ln2ScaleData)
		ln2Bias, _ := tensor.New[float64]([]int{features}, ln2BiasData)

		// LayerNorm1
		normed1, _ := LayerNorm(ctx, engine, in, ln1Scale, ln1Bias, epsVal)
		// Skip MHA (identity) + residual = normed1 + input
		res1, _ := engine.Add(ctx, normed1, in)
		// LayerNorm2
		normed2, _ := LayerNorm(ctx, engine, res1, ln2Scale, ln2Bias, epsVal)
		// FFN: Linear1 -> GELU -> Linear2
		h, _ := Linear(ctx, engine, normed2, w1, b1)
		act, _ := GELU(ctx, engine, ops, h)
		ffnOut, _ := Linear(ctx, engine, act, w2, b2)
		// Residual
		res2, _ := engine.Add(ctx, ffnOut, res1)

		// L = sum(dOutput * res2)
		var loss float64
		for i, v := range res2.Data() {
			loss += dOutputData[i] * v
		}
		return loss
	}

	// Analytical backward
	input := mustTensor(t, []int{batch, features}, inputData)
	ln1Scale := mustTensor(t, []int{features}, ln1ScaleData)
	ln1Bias := mustTensor(t, []int{features}, ln1BiasData)
	w1 := mustTensor(t, []int{hidden, features}, w1Data)
	w2 := mustTensor(t, []int{features, hidden}, w2Data)
	ln2Scale := mustTensor(t, []int{features}, ln2ScaleData)
	ln2Bias := mustTensor(t, []int{features}, ln2BiasData)

	// Forward pass (save intermediates)
	normed1, err := LayerNorm(ctx, engine, input, ln1Scale, ln1Bias, epsVal)
	if err != nil {
		t.Fatal(err)
	}
	res1, err := engine.Add(ctx, normed1, input)
	if err != nil {
		t.Fatal(err)
	}
	normed2, err := LayerNorm(ctx, engine, res1, ln2Scale, ln2Bias, epsVal)
	if err != nil {
		t.Fatal(err)
	}
	h, err := Linear(ctx, engine, normed2, w1, mustTensor(t, []int{hidden}, b1Data))
	if err != nil {
		t.Fatal(err)
	}
	act, err := GELU(ctx, engine, ops, h)
	if err != nil {
		t.Fatal(err)
	}

	dOutput := mustTensor(t, []int{batch, features}, dOutputData)

	// Backward through residual2: dRes2 = dOutput, dFFN = dOutput, dRes1_from_res2 = dOutput
	// Backward through Linear2
	dAct, _, _, err := LinearBackward(ctx, engine, dOutput, act, w2)
	if err != nil {
		t.Fatal(err)
	}
	// Backward through GELU
	dH, err := GELUBackward(ctx, engine, ops, dAct, h)
	if err != nil {
		t.Fatal(err)
	}
	// Backward through Linear1
	dNormed2, _, _, err := LinearBackward(ctx, engine, dH, normed2, w1)
	if err != nil {
		t.Fatal(err)
	}
	// Backward through LayerNorm2
	dRes1FromLN2, _, _, err := LayerNormBackward(ctx, engine, dNormed2, res1, ln2Scale, epsVal)
	if err != nil {
		t.Fatal(err)
	}
	// Combine residual gradients: dRes1 = dRes1_from_res2 + dRes1_from_LN2
	dRes1, err := engine.Add(ctx, dOutput, dRes1FromLN2)
	if err != nil {
		t.Fatal(err)
	}
	// Backward through residual1: dNormed1 = dRes1, dInput_from_res1 = dRes1
	// Backward through LayerNorm1
	dInputFromLN1, _, _, err := LayerNormBackward(ctx, engine, dRes1, input, ln1Scale, epsVal)
	if err != nil {
		t.Fatal(err)
	}
	// Final dInput = dInput_from_res1 + dInput_from_LN1
	dInput, err := engine.Add(ctx, dRes1, dInputFromLN1)
	if err != nil {
		t.Fatal(err)
	}

	// Numerical gradient check for dInput
	eps := 1e-5
	tol := 1e-3

	for i := range inputData {
		plus := make([]float64, len(inputData))
		minus := make([]float64, len(inputData))
		copy(plus, inputData)
		copy(minus, inputData)
		plus[i] += eps
		minus[i] -= eps
		ng := (composedForward(plus) - composedForward(minus)) / (2 * eps)
		got := dInput.Data()[i]
		if math.Abs(got-ng) > tol {
			t.Errorf("dInput[%d]: analytical=%f, numerical=%f, diff=%.2e",
				i, got, ng, math.Abs(got-ng))
		}
	}
}

// TestCompositionEquivalence verifies that composing LinearBackward + GELUBackward
// gives the same result as a single numerical gradient check over the composed
// forward f(x) = GELU(Linear(x)).
func TestCompositionEquivalence(t *testing.T) {
	ctx := context.Background()
	engine, ops := newF64Engine()

	batch, in, out := 2, 3, 2
	inputData := deterministicFill(batch*in, 7, 3)
	weightData := deterministicFill(out*in, 11, 5)
	biasData := deterministicFill(out, 5, 2)

	input := mustTensor(t, []int{batch, in}, inputData)
	weight := mustTensor(t, []int{out, in}, weightData)
	bias := mustTensor(t, []int{out}, biasData)

	// Forward
	linear, err := Linear(ctx, engine, input, weight, bias)
	if err != nil {
		t.Fatal(err)
	}

	// Use identity upstream gradient (ones)
	onesData := make([]float64, batch*out)
	for i := range onesData {
		onesData[i] = 1.0
	}
	dOutput := mustTensor(t, []int{batch, out}, onesData)

	// Composed analytical backward
	dLinear, err := GELUBackward(ctx, engine, ops, dOutput, linear)
	if err != nil {
		t.Fatal(err)
	}
	dInputAnalytical, _, _, err := LinearBackward(ctx, engine, dLinear, input, weight)
	if err != nil {
		t.Fatal(err)
	}

	// Single numerical gradient over composed forward: sum(GELU(Linear(x)))
	eps := 1e-5
	tol := 1e-4

	composedForward := func(inD []float64) float64 {
		x, _ := tensor.New[float64]([]int{batch, in}, inD)
		w, _ := tensor.New[float64]([]int{out, in}, weightData)
		b, _ := tensor.New[float64]([]int{out}, biasData)
		lin, _ := Linear(ctx, engine, x, w, b)
		g, _ := GELU(ctx, engine, ops, lin)
		var sum float64
		for _, v := range g.Data() {
			sum += v
		}
		return sum
	}

	for i := range inputData {
		plus := make([]float64, len(inputData))
		minus := make([]float64, len(inputData))
		copy(plus, inputData)
		copy(minus, inputData)
		plus[i] += eps
		minus[i] -= eps
		ng := (composedForward(plus) - composedForward(minus)) / (2 * eps)
		got := dInputAnalytical.Data()[i]
		if math.Abs(got-ng) > tol {
			t.Errorf("dInput[%d]: composed_analytical=%f, numerical=%f, diff=%.2e",
				i, got, ng, math.Abs(got-ng))
		}
	}
}

// --- helpers ---

// deterministicFill produces a reproducible float64 slice using modular arithmetic.
func deterministicFill(n, mod, offset int) []float64 {
	data := make([]float64, n)
	for i := range data {
		data[i] = float64(i%mod-offset) * 0.1
	}
	return data
}

// deterministicFillF32 produces a reproducible float32 slice.
func deterministicFillF32(n, mod, offset int) []float32 {
	data := make([]float32, n)
	for i := range data {
		data[i] = float32(i%mod-offset) * 0.1
	}
	return data
}

func mustTensor[T tensor.Numeric](t *testing.T, shape []int, data []T) *tensor.TensorNumeric[T] {
	t.Helper()
	out, err := tensor.New[T](shape, data)
	if err != nil {
		t.Fatalf("mustTensor: %v", err)
	}
	return out
}

func assertShapeGeneric[T tensor.Numeric](t *testing.T, name string, got *tensor.TensorNumeric[T], want []int) {
	t.Helper()
	s := got.Shape()
	if len(s) != len(want) {
		t.Fatalf("%s: shape rank %d, want %d", name, len(s), len(want))
	}
	for i := range s {
		if s[i] != want[i] {
			t.Fatalf("%s: shape %v, want %v", name, s, want)
		}
	}
}

// numericalGradLinearGELU computes the numerical gradient of L = sum(dOutput * GELU(Linear(x)))
// with respect to the named parameter at the given index.
func numericalGradLinearGELU(ctx context.Context, engine compute.Engine[float64], ops numeric.Arithmetic[float64],
	inputData, weightData, biasData, dOutputData []float64,
	inputShape, weightShape, biasShape []int,
	param string, idx int, eps float64) float64 {

	perturb := func(data []float64, i int, delta float64) []float64 {
		cp := make([]float64, len(data))
		copy(cp, data)
		cp[i] += delta
		return cp
	}

	forward := func(inD, wD, bD []float64) float64 {
		in, _ := tensor.New[float64](inputShape, inD)
		w, _ := tensor.New[float64](weightShape, wD)
		b, _ := tensor.New[float64](biasShape, bD)
		lin, err := Linear(ctx, engine, in, w, b)
		if err != nil {
			panic(err)
		}
		g, err := GELU(ctx, engine, ops, lin)
		if err != nil {
			panic(err)
		}
		var loss float64
		for i, v := range g.Data() {
			loss += dOutputData[i] * v
		}
		return loss
	}

	var lPlus, lMinus float64
	switch param {
	case "input":
		lPlus = forward(perturb(inputData, idx, eps), weightData, biasData)
		lMinus = forward(perturb(inputData, idx, -eps), weightData, biasData)
	case "weight":
		lPlus = forward(inputData, perturb(weightData, idx, eps), biasData)
		lMinus = forward(inputData, perturb(weightData, idx, -eps), biasData)
	case "bias":
		lPlus = forward(inputData, weightData, perturb(biasData, idx, eps))
		lMinus = forward(inputData, weightData, perturb(biasData, idx, -eps))
	}

	return (lPlus - lMinus) / (2 * eps)
}
