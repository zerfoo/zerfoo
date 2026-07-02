package loss

import (
	"context"
	"math"
	"testing"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/ztensor/types"
)

func TestBCELoss_Forward(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)
	bce := NewBCELoss[float32](engine, ops)

	tests := []struct {
		name     string
		yData    []float32
		pData    []float32
		shape    []int
		expected float32
	}{
		{
			name:  "basic binary targets",
			yData: []float32{1, 0, 1, 0},
			pData: []float32{0.9, 0.1, 0.8, 0.2},
			shape: []int{4},
			// -[1*log(0.9) + 0*log(0.1) + 1*log(0.8) + 0*log(0.8)] / 4
			expected: float32(-(math.Log(0.9) + math.Log(0.9) + math.Log(0.8) + math.Log(0.8)) / 4),
		},
		{
			name:     "perfect predictions",
			yData:    []float32{1, 0},
			pData:    []float32{1.0, 0.0}, // will be clamped
			shape:    []int{2},
			expected: float32(-0.5 * (math.Log(1-1e-7) + math.Log(1-1e-7))),
		},
		{
			name:     "all ones target",
			yData:    []float32{1, 1, 1},
			pData:    []float32{0.5, 0.5, 0.5},
			shape:    []int{3},
			expected: float32(-math.Log(0.5)),
		},
		{
			name:     "all zeros target",
			yData:    []float32{0, 0, 0},
			pData:    []float32{0.5, 0.5, 0.5},
			shape:    []int{3},
			expected: float32(-math.Log(0.5)),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			predictions, err := tensor.New[float32](tt.shape, tt.pData)
			if err != nil {
				t.Fatalf("failed to create predictions: %v", err)
			}
			targets, err := tensor.New[float32](tt.shape, tt.yData)
			if err != nil {
				t.Fatalf("failed to create targets: %v", err)
			}

			loss, err := bce.Forward(context.Background(), predictions, targets)
			if err != nil {
				t.Fatalf("Forward failed: %v", err)
			}

			got := loss.Data()[0]
			if math.Abs(float64(got-tt.expected)) > 1e-5 {
				t.Errorf("expected %f, got %f", tt.expected, got)
			}
		})
	}
}

func TestBCELoss_Backward(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)
	bce := NewBCELoss[float32](engine, ops)

	y := []float32{1, 0, 1, 0}
	p := []float32{0.9, 0.1, 0.8, 0.2}
	shape := []int{4}
	n := float64(len(y))

	predictions, _ := tensor.New[float32](shape, p)
	targets, _ := tensor.New[float32](shape, y)
	dOut, _ := tensor.New[float32]([]int{1}, []float32{1.0})

	grads, err := bce.Backward(context.Background(), types.FullBackprop, dOut, predictions, targets)
	if err != nil {
		t.Fatalf("Backward failed: %v", err)
	}

	// Expected gradient for each element: ((1-y)/(1-p) - y/p) / N
	for i := 0; i < len(y); i++ {
		yi := float64(y[i])
		pi := float64(p[i])
		expected := float32(((1-yi)/(1-pi) - yi/pi) / n)
		got := grads[0].Data()[i]
		if math.Abs(float64(got-expected)) > 1e-5 {
			t.Errorf("grad[%d]: expected %f, got %f", i, expected, got)
		}
	}

	// Targets gradient should be zero
	for i, v := range grads[1].Data() {
		if v != 0 {
			t.Errorf("target grad[%d]: expected 0, got %f", i, v)
		}
	}
}

func TestBCELoss_Backward_NumericalGradient(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	y := []float32{1, 0, 1, 0}
	p := []float32{0.7, 0.3, 0.6, 0.4}
	shape := []int{4}
	eps := float32(1e-4)

	targets, _ := tensor.New[float32](shape, y)
	dOut, _ := tensor.New[float32]([]int{1}, []float32{1.0})

	// Compute analytical gradient
	bce := NewBCELoss[float32](engine, ops)
	predictions, _ := tensor.New[float32](shape, p)
	grads, err := bce.Backward(context.Background(), types.FullBackprop, dOut, predictions, targets)
	if err != nil {
		t.Fatalf("Backward failed: %v", err)
	}

	// Compute numerical gradient for each element
	for i := 0; i < len(p); i++ {
		pPlus := make([]float32, len(p))
		pMinus := make([]float32, len(p))
		copy(pPlus, p)
		copy(pMinus, p)
		pPlus[i] += eps
		pMinus[i] -= eps

		bcePlus := NewBCELoss[float32](engine, ops)
		predPlus, _ := tensor.New[float32](shape, pPlus)
		lossPlus, _ := bcePlus.Forward(context.Background(), predPlus, targets)

		bceMinus := NewBCELoss[float32](engine, ops)
		predMinus, _ := tensor.New[float32](shape, pMinus)
		lossMinus, _ := bceMinus.Forward(context.Background(), predMinus, targets)

		numerical := (lossPlus.Data()[0] - lossMinus.Data()[0]) / (2 * eps)
		analytical := grads[0].Data()[i]

		if math.Abs(float64(numerical-analytical)) > 1e-3 {
			t.Errorf("grad[%d]: numerical=%f, analytical=%f", i, numerical, analytical)
		}
	}
}

func TestBCELoss_EdgeCases(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)
	bce := NewBCELoss[float32](engine, ops)

	t.Run("p near zero", func(t *testing.T) {
		predictions, _ := tensor.New[float32]([]int{2}, []float32{0.0, 1e-10})
		targets, _ := tensor.New[float32]([]int{2}, []float32{0, 0})

		loss, err := bce.Forward(context.Background(), predictions, targets)
		if err != nil {
			t.Fatalf("Forward failed: %v", err)
		}
		if math.IsNaN(float64(loss.Data()[0])) || math.IsInf(float64(loss.Data()[0]), 0) {
			t.Errorf("loss should not be NaN or Inf, got %f", loss.Data()[0])
		}
	})

	t.Run("p near one", func(t *testing.T) {
		predictions, _ := tensor.New[float32]([]int{2}, []float32{1.0, 1 - 1e-10})
		targets, _ := tensor.New[float32]([]int{2}, []float32{1, 1})

		loss, err := bce.Forward(context.Background(), predictions, targets)
		if err != nil {
			t.Fatalf("Forward failed: %v", err)
		}
		if math.IsNaN(float64(loss.Data()[0])) || math.IsInf(float64(loss.Data()[0]), 0) {
			t.Errorf("loss should not be NaN or Inf, got %f", loss.Data()[0])
		}
	})

	t.Run("backward p near zero", func(t *testing.T) {
		bce2 := NewBCELoss[float32](engine, ops)
		predictions, _ := tensor.New[float32]([]int{2}, []float32{0.0, 1e-10})
		targets, _ := tensor.New[float32]([]int{2}, []float32{0, 0})
		dOut, _ := tensor.New[float32]([]int{1}, []float32{1.0})

		grads, err := bce2.Backward(context.Background(), types.FullBackprop, dOut, predictions, targets)
		if err != nil {
			t.Fatalf("Backward failed: %v", err)
		}
		for i, v := range grads[0].Data() {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				t.Errorf("grad[%d] should not be NaN or Inf, got %f", i, v)
			}
		}
	})

	t.Run("backward p near one", func(t *testing.T) {
		bce2 := NewBCELoss[float32](engine, ops)
		predictions, _ := tensor.New[float32]([]int{2}, []float32{1.0, 1 - 1e-10})
		targets, _ := tensor.New[float32]([]int{2}, []float32{1, 1})
		dOut, _ := tensor.New[float32]([]int{1}, []float32{1.0})

		grads, err := bce2.Backward(context.Background(), types.FullBackprop, dOut, predictions, targets)
		if err != nil {
			t.Fatalf("Backward failed: %v", err)
		}
		for i, v := range grads[0].Data() {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				t.Errorf("grad[%d] should not be NaN or Inf, got %f", i, v)
			}
		}
	})
}

func TestBCELoss_WrongInputCount(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)
	bce := NewBCELoss[float32](engine, ops)

	t.Run("zero inputs", func(t *testing.T) {
		_, err := bce.Forward(context.Background())
		if err == nil {
			t.Error("expected error with 0 inputs")
		}
	})

	t.Run("one input", func(t *testing.T) {
		predictions, _ := tensor.New[float32]([]int{2}, []float32{0.5, 0.5})
		_, err := bce.Forward(context.Background(), predictions)
		if err == nil {
			t.Error("expected error with 1 input")
		}
	})

	t.Run("three inputs", func(t *testing.T) {
		a, _ := tensor.New[float32]([]int{2}, []float32{0.5, 0.5})
		b, _ := tensor.New[float32]([]int{2}, []float32{1, 0})
		c, _ := tensor.New[float32]([]int{2}, []float32{0.1, 0.2})
		_, err := bce.Forward(context.Background(), a, b, c)
		if err == nil {
			t.Error("expected error with 3 inputs")
		}
	})
}
