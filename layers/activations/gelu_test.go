package activations

import (
	"context"
	"math"
	"testing"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/ztensor/types"
)

// geluScalar computes the GELU activation for a single float64 value.
// y = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
func geluScalar(x float64) float64 {
	inner := math.Sqrt(2/math.Pi) * (x + 0.044715*x*x*x)
	return 0.5 * x * (1 + math.Tanh(inner))
}

func TestGeluBackward(t *testing.T) {
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	ops := numeric.Float32Ops{}
	ctx := context.Background()

	// Table-driven: test backward (analytical gradient) against numerical gradient
	// using central difference: (gelu(x+h) - gelu(x-h)) / (2h)
	tests := []struct {
		name string
		x    float64
	}{
		{"x=0", 0.0},
		{"x=1", 1.0},
		{"x=-1", -1.0},
		{"x=3", 3.0},
		{"x=-3", -3.0},
		{"x=0.5", 0.5},
		{"x=-0.5", -0.5},
		{"x=2", 2.0},
		{"x=-2", -2.0},
	}

	h := 1e-4
	tol := 1e-3 // tolerance for float32 engine vs float64 numerical grad

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			x := float32(tt.x)

			// Compute analytical gradient via Backward
			gelu := NewGelu(engine, ops)
			input, err := tensor.New[float32]([]int{1}, []float32{x})
			if err != nil {
				t.Fatalf("create input: %v", err)
			}

			_, err = gelu.Forward(ctx, input)
			if err != nil {
				t.Fatalf("Forward: %v", err)
			}

			onesGrad, err := tensor.New[float32]([]int{1}, []float32{1.0})
			if err != nil {
				t.Fatalf("create grad: %v", err)
			}

			grads, err := gelu.Backward(ctx, types.FullBackprop, onesGrad, input)
			if err != nil {
				t.Fatalf("Backward: %v", err)
			}
			if len(grads) != 1 || len(grads[0].Data()) != 1 {
				t.Fatalf("unexpected gradient shape: %d tensors", len(grads))
			}
			analyticalGrad := float64(grads[0].Data()[0])

			// Compute numerical gradient via central difference in float64
			numericalGrad := (geluScalar(tt.x+h) - geluScalar(tt.x-h)) / (2 * h)

			if math.Abs(analyticalGrad-numericalGrad) > tol {
				t.Errorf("x=%.1f: analytical=%v, numerical=%v, diff=%v",
					tt.x, analyticalGrad, numericalGrad,
					math.Abs(analyticalGrad-numericalGrad))
			}
		})
	}
}

func TestGeluBackwardBatch(t *testing.T) {
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	ops := numeric.Float32Ops{}
	ctx := context.Background()

	// Test backward with a batch of values at once
	inputData := []float32{-3, -1, 0, 0.5, 1, 3}
	gelu := NewGelu(engine, ops)

	input, err := tensor.New[float32]([]int{1, len(inputData)}, inputData)
	if err != nil {
		t.Fatalf("create input: %v", err)
	}

	_, err = gelu.Forward(ctx, input)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}

	onesGrad, err := tensor.New[float32]([]int{1, len(inputData)}, make([]float32, len(inputData)))
	if err != nil {
		t.Fatalf("create grad: %v", err)
	}
	for i := range onesGrad.Data() {
		onesGrad.Data()[i] = 1.0
	}

	grads, err := gelu.Backward(ctx, types.FullBackprop, onesGrad, input)
	if err != nil {
		t.Fatalf("Backward: %v", err)
	}
	if len(grads) != 1 {
		t.Fatalf("expected 1 gradient tensor, got %d", len(grads))
	}

	gradData := grads[0].Data()
	if len(gradData) != len(inputData) {
		t.Fatalf("gradient length = %d, want %d", len(gradData), len(inputData))
	}

	h := 1e-4
	tol := 1e-3

	for i, x := range inputData {
		numerical := (geluScalar(float64(x)+h) - geluScalar(float64(x)-h)) / (2 * h)
		analytical := float64(gradData[i])
		if math.Abs(analytical-numerical) > tol {
			t.Errorf("index %d (x=%.1f): analytical=%v, numerical=%v, diff=%v",
				i, x, analytical, numerical, math.Abs(analytical-numerical))
		}
	}
}

func TestGeluBackwardWithOutputGradient(t *testing.T) {
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	ops := numeric.Float32Ops{}
	ctx := context.Background()

	// Backward should multiply the derivative by the output gradient (chain rule)
	gelu := NewGelu(engine, ops)
	input, err := tensor.New[float32]([]int{1, 2}, []float32{1.0, -1.0})
	if err != nil {
		t.Fatalf("create input: %v", err)
	}

	_, err = gelu.Forward(ctx, input)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}

	// Use output gradient of [2.0, 3.0]
	outGrad, err := tensor.New[float32]([]int{1, 2}, []float32{2.0, 3.0})
	if err != nil {
		t.Fatalf("create outGrad: %v", err)
	}

	grads, err := gelu.Backward(ctx, types.FullBackprop, outGrad, input)
	if err != nil {
		t.Fatalf("Backward: %v", err)
	}

	// Also compute with unit gradient for reference
	gelu2 := NewGelu(engine, ops)
	input2, _ := tensor.New[float32]([]int{1, 2}, []float32{1.0, -1.0})
	_, _ = gelu2.Forward(ctx, input2)
	unitGrad, _ := tensor.New[float32]([]int{1, 2}, []float32{1.0, 1.0})
	unitGrads, _ := gelu2.Backward(ctx, types.FullBackprop, unitGrad, input2)

	// grads[0] should equal unitGrads[0] * outGrad element-wise
	tol := float32(1e-5)
	for i := range grads[0].Data() {
		got := grads[0].Data()[i]
		want := unitGrads[0].Data()[i] * outGrad.Data()[i]
		if diff := got - want; diff < -tol || diff > tol {
			t.Errorf("index %d: got %v, want %v (diff %v)", i, got, want, diff)
		}
	}
}

func TestGeluForwardInputCountError(t *testing.T) {
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	ops := numeric.Float32Ops{}
	ctx := context.Background()
	gelu := NewGelu(engine, ops)

	// Zero inputs
	_, err := gelu.Forward(ctx)
	if err == nil {
		t.Error("expected error for 0 inputs")
	}

	// Two inputs
	a := makeTensor(t, []int{2}, []float32{1, 2})
	b := makeTensor(t, []int{2}, []float32{3, 4})
	_, err = gelu.Forward(ctx, a, b)
	if err == nil {
		t.Error("expected error for 2 inputs")
	}
}

func TestGeluForwardKnownValues(t *testing.T) {
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	ops := numeric.Float32Ops{}
	ctx := context.Background()

	tests := []struct {
		name string
		x    float32
		want float64
	}{
		{"zero", 0, 0},
		{"positive", 1.0, geluScalar(1.0)},
		{"negative", -1.0, geluScalar(-1.0)},
		{"large_positive", 3.0, geluScalar(3.0)},
		{"large_negative", -3.0, geluScalar(-3.0)},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gelu := NewGelu(engine, ops)
			input, _ := tensor.New[float32]([]int{1}, []float32{tt.x})
			out, err := gelu.Forward(ctx, input)
			if err != nil {
				t.Fatalf("Forward: %v", err)
			}
			got := float64(out.Data()[0])
			if math.Abs(got-tt.want) > 1e-5 {
				t.Errorf("GELU(%v) = %v, want %v", tt.x, got, tt.want)
			}
		})
	}
}

func TestGeluBackwardEngineErrors(t *testing.T) {
	ctx := context.Background()
	ops := makeOps()
	input := makeTensor(t, []int{2}, []float32{0.5, 1.0})

	// Backward calls: Mul(x2), Mul(x3), MulScalar(0.044715*x3), Add(x+...),
	// MulScalar(sqrt(2/pi)*...), Tanh, Mul(tanh^2), MulScalar(-1), AddScalar(1),
	// MulScalar(3*0.044715), AddScalar(1+...), MulScalar(sqrt(2/pi)),
	// AddScalar(1+tanh), MulScalar(0.5), Mul(x*sechSq), Mul(xSechSq*dudx),
	// MulScalar(0.5), Add(derivative), Mul(outputGrad*derivative)
	//
	// Test that errors in backward propagate correctly
	tests := []struct {
		name   string
		failOn map[string]int
	}{
		// Forward uses: Mul(1,2), MulScalar(1,2), Add(1), Tanh(1), AddScalar(1), Mul(3), MulScalar(3)
		// Backward uses: Mul(4,5), MulScalar(4,5), Add(2), Tanh(2), etc.
		// Fail on 2nd Tanh call (backward)
		{"Tanh_backward", map[string]int{"Tanh": 2}},
		// Fail on a late Mul in backward
		{"Mul_backward", map[string]int{"Mul": 7}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			eng := newErrEngine(tt.failOn)
			gelu := NewGelu[float32](eng, ops)
			out, err := gelu.Forward(ctx, input)
			if err != nil {
				t.Skipf("Forward failed (expected for some error injections): %v", err)
			}
			grad := makeTensor(t, out.Shape(), []float32{1, 1})
			_, err = gelu.Backward(ctx, types.FullBackprop, grad, input)
			if err == nil {
				t.Error("expected error")
			}
		})
	}
}
