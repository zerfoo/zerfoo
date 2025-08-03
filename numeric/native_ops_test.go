package numeric

import (
	"math"
	"testing"
)

func TestFloat32Ops_Add(t *testing.T) {
	ops := Float32Ops{}
	tests := []struct {
		name           string
		a, b, expected float32
	}{
		{"positive numbers", 1.0, 2.0, 3.0},
		{"negative numbers", -1.0, -2.0, -3.0},
		{"mixed numbers", 1.0, -2.0, -1.0},
		{"zero", 0.0, 0.0, 0.0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := ops.Add(tt.a, tt.b)
			if result != tt.expected {
				t.Errorf("Add(%v, %v): expected %v, got %v", tt.a, tt.b, tt.expected, result)
			}
		})
	}
}

func TestFloat32Ops_Sub(t *testing.T) {
	ops := Float32Ops{}
	tests := []struct {
		name           string
		a, b, expected float32
	}{
		{"positive numbers", 3.0, 1.0, 2.0},
		{"negative numbers", -1.0, -2.0, 1.0},
		{"mixed numbers", 1.0, -2.0, 3.0},
		{"zero", 0.0, 0.0, 0.0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := ops.Sub(tt.a, tt.b)
			if result != tt.expected {
				t.Errorf("Sub(%v, %v): expected %v, got %v", tt.a, tt.b, tt.expected, result)
			}
		})
	}
}

func TestFloat32Ops_Mul(t *testing.T) {
	ops := Float32Ops{}
	tests := []struct {
		name           string
		a, b, expected float32
	}{
		{"positive numbers", 2.0, 3.0, 6.0},
		{"negative numbers", -2.0, -3.0, 6.0},
		{"mixed numbers", 2.0, -3.0, -6.0},
		{"zero", 0.0, 5.0, 0.0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := ops.Mul(tt.a, tt.b)
			if result != tt.expected {
				t.Errorf("Mul(%v, %v): expected %v, got %v", tt.a, tt.b, tt.expected, result)
			}
		})
	}
}

func TestFloat32Ops_Div(t *testing.T) {
	ops := Float32Ops{}
	tests := []struct {
		name           string
		a, b, expected float32
	}{
		{"positive numbers", 6.0, 3.0, 2.0},
		{"negative numbers", -6.0, -3.0, 2.0},
		{"mixed numbers", 6.0, -3.0, -2.0},
		{"divide by one", 5.0, 1.0, 5.0},
		{"zero dividend", 0.0, 5.0, 0.0},
		{"divide by zero", 5.0, 0.0, 0.0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := ops.Div(tt.a, tt.b)
			if result != tt.expected {
				t.Errorf("Div(%v, %v): expected %v, got %v", tt.a, tt.b, tt.expected, result)
			}
		})
	}
}

func TestFloat32Ops_Tanh(t *testing.T) {
	ops := Float32Ops{}
	tests := []struct {
		name        string
		x, expected float32
	}{
		{"zero", 0.0, 0.0},
		{"positive", 1.0, float32(math.Tanh(1.0))},
		{"negative", -1.0, float32(math.Tanh(-1.0))},
		{"large positive", 100.0, float32(math.Tanh(100.0))},
		{"large negative", -100.0, float32(math.Tanh(-100.0))},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := ops.Tanh(tt.x)
			if result != tt.expected {
				t.Errorf("Tanh(%v): expected %v, got %v", tt.x, tt.expected, result)
			}
		})
	}
}

func TestFloat32Ops_Sigmoid(t *testing.T) {
	ops := Float32Ops{}
	tests := []struct {
		name        string
		x, expected float32
	}{
		{"zero", 0.0, 0.5},
		{"positive", 1.0, 1.0 / (1.0 + float32(math.Exp(-1.0)))},
		{"negative", -1.0, 1.0 / (1.0 + float32(math.Exp(1.0)))},
		{"large positive", 100.0, 1.0},
		{"large negative", -100.0, 0.0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := ops.Sigmoid(tt.x)
			if result != tt.expected {
				t.Errorf("Sigmoid(%v): expected %v, got %v", tt.x, tt.expected, result)
			}
		})
	}
}

func TestFloat32Ops_TanhGrad(t *testing.T) {
	ops := Float32Ops{}
	tests := []struct {
		name        string
		x, expected float32
	}{
		{"zero", 0.0, 1.0},
		{"positive", 1.0, 1.0 - float32(math.Pow(math.Tanh(1.0), 2))},
		{"negative", -1.0, 1.0 - float32(math.Pow(math.Tanh(-1.0), 2))},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := ops.TanhGrad(tt.x)
			if result != tt.expected {
				t.Errorf("TanhGrad(%v): expected %v, got %v", tt.x, tt.expected, result)
			}
		})
	}
}

func TestFloat32Ops_SigmoidGrad(t *testing.T) {
	ops := Float32Ops{}
	tests := []struct {
		name        string
		x, expected float32
	}{
		{"zero", 0.0, 0.25},
		{"positive", 1.0, ops.Sigmoid(1.0) * (1.0 - ops.Sigmoid(1.0))},
		{"negative", -1.0, ops.Sigmoid(-1.0) * (1.0 - ops.Sigmoid(-1.0))},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := ops.SigmoidGrad(tt.x)
			if result != tt.expected {
				t.Errorf("SigmoidGrad(%v): expected %v, got %v", tt.x, tt.expected, result)
			}
		})
	}
}

func TestFloat32Ops_FromFloat32(t *testing.T) {
	ops := Float32Ops{}
	tests := []struct {
		name        string
		f, expected float32
	}{
		{"zero", 0.0, 0.0},
		{"positive", 1.0, 1.0},
		{"negative", -1.0, -1.0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := ops.FromFloat32(tt.f)
			if result != tt.expected {
				t.Errorf("FromFloat32(%v): expected %v, got %v", tt.f, tt.expected, result)
			}
		})
	}
}

func TestFloat32Ops_IsZero(t *testing.T) {
	ops := Float32Ops{}
	tests := []struct {
		name     string
		v        float32
		expected bool
	}{
		{"zero", 0.0, true},
		{"non-zero", 1.0, false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := ops.IsZero(tt.v)
			if result != tt.expected {
				t.Errorf("IsZero(%v): expected %v, got %v", tt.v, tt.expected, result)
			}
		})
	}
}

func TestFloat32Ops_Exp(t *testing.T) {
	ops := Float32Ops{}
	tests := []struct {
		name        string
		x, expected float32
	}{
		{"zero", 0.0, float32(math.Exp(0.0))},
		{"positive", 1.0, float32(math.Exp(1.0))},
		{"negative", -1.0, float32(math.Exp(-1.0))},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := ops.Exp(tt.x)
			if result != tt.expected {
				t.Errorf("Exp(%v): expected %v, got %v", tt.x, tt.expected, result)
			}
		})
	}
}

func TestFloat32Ops_Log(t *testing.T) {
	ops := Float32Ops{}
	tests := []struct {
		name        string
		x, expected float32
	}{
		{"one", 1.0, float32(math.Log(1.0))},
		{"positive", 2.0, float32(math.Log(2.0))},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := ops.Log(tt.x)
			if result != tt.expected {
				t.Errorf("Log(%v): expected %v, got %v", tt.x, tt.expected, result)
			}
		})
	}
}

func TestFloat32Ops_Pow(t *testing.T) {
	ops := Float32Ops{}
	tests := []struct {
		name                     string
		base, exponent, expected float32
	}{
		{"base 2 exp 3", 2.0, 3.0, 8.0},
		{"base 5 exp 0", 5.0, 0.0, 1.0},
		{"base 4 exp 0.5", 4.0, 0.5, 2.0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := ops.Pow(tt.base, tt.exponent)
			if result != tt.expected {
				t.Errorf("Pow(%v, %v): expected %v, got %v", tt.base, tt.exponent, tt.expected, result)
			}
		})
	}
}

func TestFloat32Ops_Abs(t *testing.T) {
	ops := Float32Ops{}
	tests := []struct {
		name        string
		x, expected float32
	}{
		{"positive", 1.0, 1.0},
		{"negative", -1.0, 1.0},
		{"zero", 0.0, 0.0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := ops.Abs(tt.x)
			if result != tt.expected {
				t.Errorf("Abs(%v): expected %v, got %v", tt.x, tt.expected, result)
			}
		})
	}
}

func TestFloat32Ops_Sum(t *testing.T) {
	ops := Float32Ops{}
	tests := []struct {
		name     string
		s        []float32
		expected float32
	}{
		{"positive", []float32{1.0, 2.0, 3.0}, 6.0},
		{"negative", []float32{-1.0, -2.0, -3.0}, -6.0},
		{"mixed", []float32{1.0, -2.0, 3.0}, 2.0},
		{"empty", []float32{}, 0.0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := ops.Sum(tt.s)
			if result != tt.expected {
				t.Errorf("Sum(%v): expected %v, got %v", tt.s, tt.expected, result)
			}
		})
	}
}

func TestFloat32Ops_ReLU(t *testing.T) {
	ops := Float32Ops{}
	tests := []struct {
		name     string
		x        float32
		expected float32
	}{
		{"positive", 2.5, 2.5},
		{"negative", -1.5, 0.0},
		{"zero", 0.0, 0.0},
		{"small positive", 0.1, 0.1},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := ops.ReLU(tt.x)
			if result != tt.expected {
				t.Errorf("ReLU(%v): expected %v, got %v", tt.x, tt.expected, result)
			}
		})
	}
}

func TestFloat32Ops_LeakyReLU(t *testing.T) {
	ops := Float32Ops{}
	tests := []struct {
		name     string
		x        float32
		alpha    float64
		expected float32
		epsilon  float32
	}{
		{"positive", 2.0, 0.1, 2.0, 0.001},
		{"negative", -2.0, 0.1, -0.2, 0.001},
		{"zero", 0.0, 0.1, 0.0, 0.001},
		{"negative with different alpha", -1.0, 0.2, -0.2, 0.001},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := ops.LeakyReLU(tt.x, tt.alpha)
			if math.Abs(float64(result-tt.expected)) > float64(tt.epsilon) {
				t.Errorf("LeakyReLU(%v, %v): expected %v, got %v", tt.x, tt.alpha, tt.expected, result)
			}
		})
	}
}

func TestFloat32Ops_ReLUGrad(t *testing.T) {
	ops := Float32Ops{}
	tests := []struct {
		name     string
		x        float32
		expected float32
	}{
		{"positive", 2.5, 1.0},
		{"negative", -1.5, 0.0},
		{"zero", 0.0, 0.0},
		{"small positive", 0.1, 1.0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := ops.ReLUGrad(tt.x)
			if result != tt.expected {
				t.Errorf("ReLUGrad(%v): expected %v, got %v", tt.x, tt.expected, result)
			}
		})
	}
}

func TestFloat32Ops_LeakyReLUGrad(t *testing.T) {
	ops := Float32Ops{}
	tests := []struct {
		name     string
		x        float32
		alpha    float64
		expected float32
		epsilon  float32
	}{
		{"positive", 2.0, 0.1, 1.0, 0.001},
		{"negative", -2.0, 0.1, 0.1, 0.001},
		{"zero", 0.0, 0.1, 0.1, 0.001},
		{"negative with different alpha", -1.0, 0.2, 0.2, 0.001},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := ops.LeakyReLUGrad(tt.x, tt.alpha)
			if math.Abs(float64(result-tt.expected)) > float64(tt.epsilon) {
				t.Errorf("LeakyReLUGrad(%v, %v): expected %v, got %v", tt.x, tt.alpha, tt.expected, result)
			}
		})
	}
}

func TestFloat32Ops_ToFloat32(t *testing.T) {
	ops := Float32Ops{}
	tests := []struct {
		name     string
		x        float32
		expected float32
	}{
		{"positive", 2.5, 2.5},
		{"negative", -1.5, -1.5},
		{"zero", 0.0, 0.0},
		{"small", 0.1, 0.1},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := ops.ToFloat32(tt.x)
			if result != tt.expected {
				t.Errorf("ToFloat32(%v): expected %v, got %v", tt.x, tt.expected, result)
			}
		})
	}
}
