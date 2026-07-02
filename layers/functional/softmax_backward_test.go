package functional

import (
	"context"
	"math"
	"testing"


)

// softmaxRef computes softmax over a flat row.
func softmaxRef(x []float64) []float64 {
	max := x[0]
	for _, v := range x[1:] {
		if v > max {
			max = v
		}
	}
	out := make([]float64, len(x))
	var sum float64
	for i, v := range x {
		out[i] = math.Exp(v - max)
		sum += out[i]
	}
	for i := range out {
		out[i] /= sum
	}
	return out
}

// softmaxBackwardRef computes the reference softmax backward for one row.
func softmaxBackwardRef(dOut, s []float64) []float64 {
	n := len(s)
	dot := 0.0
	for j := 0; j < n; j++ {
		dot += dOut[j] * s[j]
	}
	dx := make([]float64, n)
	for i := 0; i < n; i++ {
		dx[i] = s[i] * (dOut[i] - dot)
	}
	return dx
}

func TestSoftmaxBackward_NumericalGradient(t *testing.T) {
	ctx := context.Background()
	eps := 1e-5
	tol := 1e-4

	shapes := [][]int{{4, 8}, {1, 3}}
	for _, shape := range shapes {
		t.Run(shapeStr(shape), func(t *testing.T) {
			engine, ops := newF64Engine()
			rows := shape[0]
			cols := shape[1]
			n := rows * cols

			// Create input data.
			inputData := make([]float64, n)
			for i := range inputData {
				inputData[i] = float64(i-n/2) * 0.5
			}

			// Compute softmax forward.
			input := makeTensor(t, shape, inputData)
			softOut, err := engine.Softmax(ctx, input, -1)
			if err != nil {
				t.Fatalf("Softmax forward: %v", err)
			}

			// Create upstream gradient (dOutput).
			dOutData := make([]float64, n)
			for i := range dOutData {
				dOutData[i] = float64(i+1) * 0.1
			}
			dOut := makeTensor(t, shape, dOutData)

			// Compute analytical gradient.
			dInput, err := SoftmaxBackward(ctx, engine, ops, dOut, softOut)
			if err != nil {
				t.Fatalf("SoftmaxBackward: %v", err)
			}
			analytical := dInput.Data()

			// Compute numerical gradient for each element.
			// We use a scalar loss = sum(dOutput * softmax(input)) so that
			// dLoss/dInput_i is what SoftmaxBackward should return.
			for i := 0; i < n; i++ {
				// f(x+eps)
				plusData := make([]float64, n)
				copy(plusData, inputData)
				plusData[i] += eps
				plusInput := makeTensor(t, shape, plusData)
				plusSoft, err := engine.Softmax(ctx, plusInput, -1)
				if err != nil {
					t.Fatalf("Softmax plus: %v", err)
				}
				plusLoss := dot64(dOutData, plusSoft.Data())

				// f(x-eps)
				minusData := make([]float64, n)
				copy(minusData, inputData)
				minusData[i] -= eps
				minusInput := makeTensor(t, shape, minusData)
				minusSoft, err := engine.Softmax(ctx, minusInput, -1)
				if err != nil {
					t.Fatalf("Softmax minus: %v", err)
				}
				minusLoss := dot64(dOutData, minusSoft.Data())

				numerical := (plusLoss - minusLoss) / (2 * eps)
				if math.Abs(analytical[i]-numerical) > tol {
					t.Errorf("element [%d]: analytical=%v, numerical=%v, diff=%v",
						i, analytical[i], numerical, math.Abs(analytical[i]-numerical))
				}
			}
		})
	}
}

func TestSoftmaxBackward_UniformInput(t *testing.T) {
	ctx := context.Background()
	engine, ops := newF64Engine()

	// Uniform input -> softmax output is 1/n for all elements.
	n := 4
	inputData := make([]float64, n)
	for i := range inputData {
		inputData[i] = 1.0
	}
	input := makeTensor(t, []int{1, n}, inputData)
	softOut, err := engine.Softmax(ctx, input, -1)
	if err != nil {
		t.Fatalf("Softmax forward: %v", err)
	}

	// Verify uniform softmax output.
	expected := 1.0 / float64(n)
	for i, v := range softOut.Data() {
		if math.Abs(v-expected) > 1e-10 {
			t.Fatalf("softmax[%d] = %v, want %v", i, v, expected)
		}
	}

	// Upstream gradient: [1, 0, 0, 0]
	dOutData := []float64{1, 0, 0, 0}
	dOut := makeTensor(t, []int{1, n}, dOutData)

	dInput, err := SoftmaxBackward(ctx, engine, ops, dOut, softOut)
	if err != nil {
		t.Fatalf("SoftmaxBackward: %v", err)
	}

	// Reference: dot = sum(dOut * s) = 1/n
	// dInput_0 = s_0 * (1 - 1/n) = (1/n) * (n-1)/n = (n-1)/n^2
	// dInput_j = s_j * (0 - 1/n) = -1/n^2 for j != 0
	wantFirst := float64(n-1) / float64(n*n)
	wantRest := -1.0 / float64(n*n)

	data := dInput.Data()
	if math.Abs(data[0]-wantFirst) > 1e-10 {
		t.Errorf("dInput[0] = %v, want %v", data[0], wantFirst)
	}
	for i := 1; i < n; i++ {
		if math.Abs(data[i]-wantRest) > 1e-10 {
			t.Errorf("dInput[%d] = %v, want %v", i, data[i], wantRest)
		}
	}
}

func TestSoftmaxBackward_MultiRow(t *testing.T) {
	ctx := context.Background()
	engine, ops := newF64Engine()

	// 4x8 shape: verify against reference implementation row by row.
	rows, cols := 4, 8
	n := rows * cols
	inputData := make([]float64, n)
	for i := range inputData {
		inputData[i] = float64(i)*0.3 - 4.0
	}
	dOutData := make([]float64, n)
	for i := range dOutData {
		dOutData[i] = float64(n-i) * 0.05
	}

	input := makeTensor(t, []int{rows, cols}, inputData)
	softOut, err := engine.Softmax(ctx, input, -1)
	if err != nil {
		t.Fatalf("Softmax forward: %v", err)
	}
	dOut := makeTensor(t, []int{rows, cols}, dOutData)

	dInput, err := SoftmaxBackward(ctx, engine, ops, dOut, softOut)
	if err != nil {
		t.Fatalf("SoftmaxBackward: %v", err)
	}

	got := dInput.Data()
	sData := softOut.Data()

	for r := 0; r < rows; r++ {
		rowS := sData[r*cols : (r+1)*cols]
		rowDOut := dOutData[r*cols : (r+1)*cols]
		ref := softmaxBackwardRef(rowDOut, rowS)
		for c := 0; c < cols; c++ {
			idx := r*cols + c
			if math.Abs(got[idx]-ref[c]) > 1e-10 {
				t.Errorf("row %d col %d: got %v, want %v", r, c, got[idx], ref[c])
			}
		}
	}
}

func TestSoftmaxBackward_Float32(t *testing.T) {
	ctx := context.Background()
	engine, ops := newF32Engine()

	input := makeTensor(t, []int{1, 3}, []float32{1, 2, 3})
	softOut, err := engine.Softmax(ctx, input, -1)
	if err != nil {
		t.Fatalf("Softmax forward: %v", err)
	}

	dOut := makeTensor(t, []int{1, 3}, []float32{0.1, 0.2, 0.3})
	dInput, err := SoftmaxBackward(ctx, engine, ops, dOut, softOut)
	if err != nil {
		t.Fatalf("SoftmaxBackward: %v", err)
	}

	// Verify against f64 reference.
	s64 := softmaxRef([]float64{1, 2, 3})
	ref := softmaxBackwardRef([]float64{0.1, 0.2, 0.3}, s64)

	for i, v := range dInput.Data() {
		if math.Abs(float64(v)-ref[i]) > 1e-5 {
			t.Errorf("dInput[%d]: got %v, want %v", i, v, ref[i])
		}
	}
}

// dot64 computes the dot product of two float64 slices.
func dot64(a, b []float64) float64 {
	var sum float64
	for i := range a {
		sum += a[i] * b[i]
	}
	return sum
}

func shapeStr(shape []int) string {
	s := "["
	for i, d := range shape {
		if i > 0 {
			s += "x"
		}
		s += fmtInt(d)
	}
	return s + "]"
}

func fmtInt(n int) string {
	if n < 0 {
		return "-" + fmtInt(-n)
	}
	if n < 10 {
		return string(rune('0' + n))
	}
	return fmtInt(n/10) + string(rune('0'+n%10))
}
