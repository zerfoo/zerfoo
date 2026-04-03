package functional

import (
	"context"
	"math"
	"testing"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
)

func TestLinearBackward(t *testing.T) {
	engine := compute.NewCPUEngine[float64](numeric.Float64Ops{})
	ctx := context.Background()

	// y = x @ weight^T + bias
	// x: [2, 3], weight: [4, 3], bias: [4]
	input, err := tensor.New[float64]([]int{2, 3}, []float64{
		1, 2, 3,
		4, 5, 6,
	})
	if err != nil {
		t.Fatal(err)
	}

	weight, err := tensor.New[float64]([]int{4, 3}, []float64{
		1, 0, 0,
		0, 1, 0,
		0, 0, 1,
		1, 1, 1,
	})
	if err != nil {
		t.Fatal(err)
	}

	// dOutput: [2, 4] — upstream gradient
	dOutput, err := tensor.New[float64]([]int{2, 4}, []float64{
		1, 0, 0, 1,
		0, 1, 1, 0,
	})
	if err != nil {
		t.Fatal(err)
	}

	dInput, dWeight, dBias, err := LinearBackward(ctx, engine, dOutput, input, weight)
	if err != nil {
		t.Fatal(err)
	}

	// dInput = dOutput @ weight
	// [1,0,0,1] @ [[1,0,0],[0,1,0],[0,0,1],[1,1,1]] = [1+1, 0+1, 0+1] = [2, 1, 1]
	// [0,1,1,0] @ [[1,0,0],[0,1,0],[0,0,1],[1,1,1]] = [0, 1, 1]
	expectedDInput := []float64{2, 1, 1, 0, 1, 1}
	assertShape(t, "dInput", dInput, []int{2, 3})
	assertData(t, "dInput", dInput.Data(), expectedDInput, 1e-10)

	// dWeight = dOutput^T @ input
	// dOutput^T: [4, 2], input: [2, 3]
	// row0: [1,0] @ [[1,2,3],[4,5,6]] = [1, 2, 3]
	// row1: [0,1] @ [[1,2,3],[4,5,6]] = [4, 5, 6]
	// row2: [0,1] @ [[1,2,3],[4,5,6]] = [4, 5, 6]
	// row3: [1,0] @ [[1,2,3],[4,5,6]] = [1, 2, 3]
	expectedDWeight := []float64{1, 2, 3, 4, 5, 6, 4, 5, 6, 1, 2, 3}
	assertShape(t, "dWeight", dWeight, []int{4, 3})
	assertData(t, "dWeight", dWeight.Data(), expectedDWeight, 1e-10)

	// dBias = sum(dOutput, axis=0)
	// [1+0, 0+1, 0+1, 1+0] = [1, 1, 1, 1]
	expectedDBias := []float64{1, 1, 1, 1}
	assertShape(t, "dBias", dBias, []int{4})
	assertData(t, "dBias", dBias.Data(), expectedDBias, 1e-10)
}

func TestLinearBackwardNumericalGradient(t *testing.T) {
	engine := compute.NewCPUEngine[float64](numeric.Float64Ops{})
	ctx := context.Background()

	shapes := []struct {
		batch, in, out int
	}{
		{1, 1, 1},
		{4, 8, 4},
		{16, 32, 16},
	}

	for _, s := range shapes {
		t.Run(
			"batch"+itoa(s.batch)+"_in"+itoa(s.in)+"_out"+itoa(s.out),
			func(t *testing.T) {
				inputData := make([]float64, s.batch*s.in)
				weightData := make([]float64, s.out*s.in)
				biasData := make([]float64, s.out)
				dOutputData := make([]float64, s.batch*s.out)

				// Deterministic pseudo-random fill
				for i := range inputData {
					inputData[i] = float64(i%7-3) * 0.1
				}
				for i := range weightData {
					weightData[i] = float64(i%11-5) * 0.1
				}
				for i := range biasData {
					biasData[i] = float64(i%5-2) * 0.1
				}
				for i := range dOutputData {
					dOutputData[i] = float64(i%9-4) * 0.1
				}

				input, _ := tensor.New[float64]([]int{s.batch, s.in}, inputData)
				weight, _ := tensor.New[float64]([]int{s.out, s.in}, weightData)
				dOutput, _ := tensor.New[float64]([]int{s.batch, s.out}, dOutputData)

				// Analytical gradients
				dInput, dWeight, dBias, err := LinearBackward(ctx, engine, dOutput, input, weight)
				if err != nil {
					t.Fatal(err)
				}

				eps := 1e-5
				tol := 1e-4

				// Numerical gradient for input
				t.Run("dInput", func(t *testing.T) {
					for i := range inputData {
						numGrad := numericalGradLinear(ctx, engine, inputData, weightData, biasData, dOutputData,
							[]int{s.batch, s.in}, []int{s.out, s.in}, []int{s.out},
							"input", i, eps)
						got := dInput.Data()[i]
						if math.Abs(got-numGrad) > tol {
							t.Errorf("dInput[%d]: analytical=%f, numerical=%f", i, got, numGrad)
						}
					}
				})

				// Numerical gradient for weight
				t.Run("dWeight", func(t *testing.T) {
					for i := range weightData {
						numGrad := numericalGradLinear(ctx, engine, inputData, weightData, biasData, dOutputData,
							[]int{s.batch, s.in}, []int{s.out, s.in}, []int{s.out},
							"weight", i, eps)
						got := dWeight.Data()[i]
						if math.Abs(got-numGrad) > tol {
							t.Errorf("dWeight[%d]: analytical=%f, numerical=%f", i, got, numGrad)
						}
					}
				})

				// Numerical gradient for bias
				t.Run("dBias", func(t *testing.T) {
					for i := range biasData {
						numGrad := numericalGradLinear(ctx, engine, inputData, weightData, biasData, dOutputData,
							[]int{s.batch, s.in}, []int{s.out, s.in}, []int{s.out},
							"bias", i, eps)
						got := dBias.Data()[i]
						if math.Abs(got-numGrad) > tol {
							t.Errorf("dBias[%d]: analytical=%f, numerical=%f", i, got, numGrad)
						}
					}
				})
			},
		)
	}
}

// numericalGradLinear computes the numerical gradient of the loss L = sum(dOutput * y)
// with respect to the parameter at the given index, using central differences.
func numericalGradLinear(ctx context.Context, engine compute.Engine[float64],
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
		y, err := Linear(ctx, engine, in, w, b)
		if err != nil {
			panic(err)
		}
		// L = sum(dOutput * y)
		yData := y.Data()
		var loss float64
		for i, v := range yData {
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

func assertShape(t *testing.T, name string, got *tensor.TensorNumeric[float64], want []int) {
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

func assertData(t *testing.T, name string, got, want []float64, tol float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: length %d, want %d", name, len(got), len(want))
	}
	for i := range got {
		if math.Abs(got[i]-want[i]) > tol {
			t.Errorf("%s[%d] = %f, want %f", name, i, got[i], want[i])
		}
	}
}

func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	buf := make([]byte, 0, 10)
	for n > 0 {
		buf = append(buf, byte('0'+n%10))
		n /= 10
	}
	for i, j := 0, len(buf)-1; i < j; i, j = i+1, j-1 {
		buf[i], buf[j] = buf[j], buf[i]
	}
	return string(buf)
}
