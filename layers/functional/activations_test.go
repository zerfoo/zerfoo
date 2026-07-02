package functional

import (
	"context"
	"math"
	"testing"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
)

// --- reference scalars ---

func geluRef(x float64) float64 {
	inner := math.Sqrt(2/math.Pi) * (x + 0.044715*x*x*x)
	return 0.5 * x * (1 + math.Tanh(inner))
}

func sigmoidRef(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

func siluRef(x float64) float64 {
	return x * sigmoidRef(x)
}

// --- helpers ---

func newF32Engine() (compute.Engine[float32], numeric.Arithmetic[float32]) {
	ops := numeric.Float32Ops{}
	return compute.NewCPUEngine[float32](ops), ops
}

func newF64Engine() (compute.Engine[float64], numeric.Arithmetic[float64]) {
	ops := numeric.Float64Ops{}
	return compute.NewCPUEngine[float64](ops), ops
}

func makeTensor[T tensor.Numeric](t *testing.T, shape []int, data []T) *tensor.TensorNumeric[T] {
	t.Helper()
	out, err := tensor.New[T](shape, data)
	if err != nil {
		t.Fatalf("makeTensor: %v", err)
	}
	return out
}

// --- GELU ---

func TestGELU(t *testing.T) {
	ctx := context.Background()

	t.Run("float32", func(t *testing.T) {
		engine, ops := newF32Engine()
		input := makeTensor(t, []int{5}, []float32{-3, -1, 0, 1, 3})
		out, err := GELU(ctx, engine, ops, input)
		if err != nil {
			t.Fatalf("GELU: %v", err)
		}
		for i, v := range out.Data() {
			want := geluRef(float64(input.Data()[i]))
			if math.Abs(float64(v)-want) > 1e-5 {
				t.Errorf("GELU[%d]: got %v, want %v", i, v, want)
			}
		}
	})

	t.Run("float64", func(t *testing.T) {
		engine, ops := newF64Engine()
		input := makeTensor(t, []int{5}, []float64{-3, -1, 0, 1, 3})
		out, err := GELU(ctx, engine, ops, input)
		if err != nil {
			t.Fatalf("GELU: %v", err)
		}
		for i, v := range out.Data() {
			want := geluRef(input.Data()[i])
			if math.Abs(v-want) > 1e-10 {
				t.Errorf("GELU[%d]: got %v, want %v", i, v, want)
			}
		}
	})
}

// --- Softmax ---

func TestSoftmax(t *testing.T) {
	ctx := context.Background()

	t.Run("float32_sum_to_1", func(t *testing.T) {
		engine, _ := newF32Engine()
		input := makeTensor(t, []int{1, 4}, []float32{1, 2, 3, 4})
		out, err := Softmax(ctx, engine, input, -1)
		if err != nil {
			t.Fatalf("Softmax: %v", err)
		}
		var sum float64
		for _, v := range out.Data() {
			if v < 0 || v > 1 {
				t.Errorf("Softmax value out of [0,1]: %v", v)
			}
			sum += float64(v)
		}
		if math.Abs(sum-1.0) > 1e-5 {
			t.Errorf("Softmax sum = %v, want 1.0", sum)
		}
	})

	t.Run("float64_sum_to_1", func(t *testing.T) {
		engine, _ := newF64Engine()
		input := makeTensor(t, []int{1, 4}, []float64{1, 2, 3, 4})
		out, err := Softmax(ctx, engine, input, -1)
		if err != nil {
			t.Fatalf("Softmax: %v", err)
		}
		var sum float64
		for _, v := range out.Data() {
			sum += v
		}
		if math.Abs(sum-1.0) > 1e-10 {
			t.Errorf("Softmax sum = %v, want 1.0", sum)
		}
	})

	t.Run("float32_monotonic", func(t *testing.T) {
		engine, _ := newF32Engine()
		input := makeTensor(t, []int{4}, []float32{1, 2, 3, 4})
		out, err := Softmax(ctx, engine, input, 0)
		if err != nil {
			t.Fatalf("Softmax: %v", err)
		}
		data := out.Data()
		for i := 1; i < len(data); i++ {
			if data[i] <= data[i-1] {
				t.Errorf("Softmax not monotonic: data[%d]=%v <= data[%d]=%v", i, data[i], i-1, data[i-1])
			}
		}
	})
}

// --- ReLU ---

func TestReLU(t *testing.T) {
	ctx := context.Background()

	t.Run("float32", func(t *testing.T) {
		engine, ops := newF32Engine()
		input := makeTensor(t, []int{6}, []float32{-3, -1, -0.5, 0, 1, 3})
		out, err := ReLU(ctx, engine, ops, input)
		if err != nil {
			t.Fatalf("ReLU: %v", err)
		}
		for i, v := range out.Data() {
			x := input.Data()[i]
			var want float32
			if x > 0 {
				want = x
			}
			if v != want {
				t.Errorf("ReLU[%d]: got %v, want %v", i, v, want)
			}
		}
	})

	t.Run("float64", func(t *testing.T) {
		engine, ops := newF64Engine()
		input := makeTensor(t, []int{6}, []float64{-3, -1, -0.5, 0, 1, 3})
		out, err := ReLU(ctx, engine, ops, input)
		if err != nil {
			t.Fatalf("ReLU: %v", err)
		}
		for i, v := range out.Data() {
			x := input.Data()[i]
			var want float64
			if x > 0 {
				want = x
			}
			if v != want {
				t.Errorf("ReLU[%d]: got %v, want %v", i, v, want)
			}
		}
	})
}

// --- SiLU ---

func TestSiLU(t *testing.T) {
	ctx := context.Background()

	t.Run("float32", func(t *testing.T) {
		engine, ops := newF32Engine()
		input := makeTensor(t, []int{5}, []float32{-3, -1, 0, 1, 3})
		out, err := SiLU(ctx, engine, ops, input)
		if err != nil {
			t.Fatalf("SiLU: %v", err)
		}
		for i, v := range out.Data() {
			want := siluRef(float64(input.Data()[i]))
			if math.Abs(float64(v)-want) > 1e-5 {
				t.Errorf("SiLU[%d]: got %v, want %v", i, v, want)
			}
		}
	})

	t.Run("float64", func(t *testing.T) {
		engine, ops := newF64Engine()
		input := makeTensor(t, []int{5}, []float64{-3, -1, 0, 1, 3})
		out, err := SiLU(ctx, engine, ops, input)
		if err != nil {
			t.Fatalf("SiLU: %v", err)
		}
		for i, v := range out.Data() {
			want := siluRef(input.Data()[i])
			if math.Abs(v-want) > 1e-10 {
				t.Errorf("SiLU[%d]: got %v, want %v", i, v, want)
			}
		}
	})
}

// --- Sigmoid ---

func TestSigmoid(t *testing.T) {
	ctx := context.Background()

	t.Run("float32_range", func(t *testing.T) {
		engine, ops := newF32Engine()
		input := makeTensor(t, []int{7}, []float32{-10, -3, -1, 0, 1, 3, 10})
		out, err := Sigmoid(ctx, engine, ops, input)
		if err != nil {
			t.Fatalf("Sigmoid: %v", err)
		}
		for i, v := range out.Data() {
			if v <= 0 || v >= 1 {
				t.Errorf("Sigmoid[%d]: value %v not in (0,1)", i, v)
			}
			want := sigmoidRef(float64(input.Data()[i]))
			if math.Abs(float64(v)-want) > 1e-5 {
				t.Errorf("Sigmoid[%d]: got %v, want %v", i, v, want)
			}
		}
	})

	t.Run("float64_range", func(t *testing.T) {
		engine, ops := newF64Engine()
		input := makeTensor(t, []int{7}, []float64{-10, -3, -1, 0, 1, 3, 10})
		out, err := Sigmoid(ctx, engine, ops, input)
		if err != nil {
			t.Fatalf("Sigmoid: %v", err)
		}
		for i, v := range out.Data() {
			if v <= 0 || v >= 1 {
				t.Errorf("Sigmoid[%d]: value %v not in (0,1)", i, v)
			}
			want := sigmoidRef(input.Data()[i])
			if math.Abs(v-want) > 1e-10 {
				t.Errorf("Sigmoid[%d]: got %v, want %v", i, v, want)
			}
		}
	})

	t.Run("float32_symmetry", func(t *testing.T) {
		engine, ops := newF32Engine()
		input := makeTensor(t, []int{1}, []float32{0})
		out, err := Sigmoid(ctx, engine, ops, input)
		if err != nil {
			t.Fatalf("Sigmoid: %v", err)
		}
		if math.Abs(float64(out.Data()[0])-0.5) > 1e-6 {
			t.Errorf("Sigmoid(0) = %v, want 0.5", out.Data()[0])
		}
	})
}
