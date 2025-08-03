package numeric

import (
	"math"
	"testing"

	"github.com/zerfoo/float8"
)

func TestFloat8Ops_Add(t *testing.T) {
	ops := Float8Ops{}
	tests := []struct {
		name           string
		a, b, expected float8.Float8
	}{
		{"positive numbers", float8.ToFloat8(1.0), float8.ToFloat8(2.0), float8.ToFloat8(3.0)},
		{"negative numbers", float8.ToFloat8(-1.0), float8.ToFloat8(-2.0), float8.ToFloat8(-3.0)},
		{"mixed numbers", float8.ToFloat8(1.0), float8.ToFloat8(-2.0), float8.ToFloat8(-1.0)},
		{"zero", float8.ToFloat8(0.0), float8.ToFloat8(0.0), float8.ToFloat8(0.0)},
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

func TestFloat8Ops_Sub(t *testing.T) {
	ops := Float8Ops{}
	tests := []struct {
		name           string
		a, b, expected float8.Float8
	}{
		{"positive numbers", float8.ToFloat8(3.0), float8.ToFloat8(1.0), float8.ToFloat8(2.0)},
		{"negative numbers", float8.ToFloat8(-1.0), float8.ToFloat8(-2.0), float8.ToFloat8(1.0)},
		{"mixed numbers", float8.ToFloat8(1.0), float8.ToFloat8(-2.0), float8.ToFloat8(3.0)},
		{"zero", float8.ToFloat8(0.0), float8.ToFloat8(0.0), float8.ToFloat8(0.0)},
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

func TestFloat8Ops_Mul(t *testing.T) {
	ops := Float8Ops{}
	tests := []struct {
		name           string
		a, b, expected float8.Float8
	}{
		{"positive numbers", float8.ToFloat8(2.0), float8.ToFloat8(3.0), float8.ToFloat8(6.0)},
		{"negative numbers", float8.ToFloat8(-2.0), float8.ToFloat8(-3.0), float8.ToFloat8(6.0)},
		{"mixed numbers", float8.ToFloat8(2.0), float8.ToFloat8(-3.0), float8.ToFloat8(-6.0)},
		{"zero", float8.ToFloat8(0.0), float8.ToFloat8(5.0), float8.ToFloat8(0.0)},
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

func TestFloat8Ops_Div(t *testing.T) {
	ops := Float8Ops{}
	tests := []struct {
		name           string
		a, b, expected float8.Float8
	}{
		{"positive numbers", float8.ToFloat8(6.0), float8.ToFloat8(3.0), float8.ToFloat8(2.0)},
		{"negative numbers", float8.ToFloat8(-6.0), float8.ToFloat8(-3.0), float8.ToFloat8(2.0)},
		{"mixed numbers", float8.ToFloat8(6.0), float8.ToFloat8(-3.0), float8.ToFloat8(-2.0)},
		{"divide by one", float8.ToFloat8(5.0), float8.ToFloat8(1.0), float8.ToFloat8(5.0)},
		{"zero dividend", float8.ToFloat8(0.0), float8.ToFloat8(5.0), float8.ToFloat8(0.0)},
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

func TestFloat8Ops_Tanh(t *testing.T) {
	ops := Float8Ops{}
	tests := []struct {
		name     string
		x        float8.Float8
		expected float8.Float8
	}{
		{"zero", float8.ToFloat8(0.0), float8.ToFloat8(0.0)},
		{"positive", float8.ToFloat8(1.0), float8.ToFloat8(float32(math.Tanh(1.0)))},
		{"negative", float8.ToFloat8(-1.0), float8.ToFloat8(float32(math.Tanh(-1.0)))},
		{"large positive", float8.ToFloat8(100.0), float8.ToFloat8(float32(math.Tanh(100.0)))},
		{"large negative", float8.ToFloat8(-100.0), float8.ToFloat8(float32(math.Tanh(-100.0)))},
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

func TestFloat8Ops_Sigmoid(t *testing.T) {
	ops := Float8Ops{}
	tests := []struct {
		name     string
		x        float8.Float8
		expected float8.Float8
	}{
		{"zero", float8.ToFloat8(0.0), float8.ToFloat8(0.5)},
		{"positive", float8.ToFloat8(1.0), float8.ToFloat8(1.0 / (1.0 + float32(math.Exp(-1.0))))},
		{"negative", float8.ToFloat8(-1.0), float8.ToFloat8(1.0 / (1.0 + float32(math.Exp(1.0))))},
		{"large positive", float8.ToFloat8(100.0), float8.ToFloat8(1.0)},
		{"large negative", float8.ToFloat8(-100.0), float8.ToFloat8(0.0)},
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

func TestFloat8Ops_TanhGrad(t *testing.T) {
	ops := Float8Ops{}
	tests := []struct {
		name     string
		x        float8.Float8
		expected float8.Float8
	}{
		{"zero", float8.ToFloat8(0.0), float8.ToFloat8(1.0)},
		{"positive", float8.ToFloat8(1.0), ops.Sub(float8.ToFloat8(1.0), ops.Mul(ops.Tanh(float8.ToFloat8(1.0)), ops.Tanh(float8.ToFloat8(1.0))))},
		{"negative", float8.ToFloat8(-1.0), ops.Sub(float8.ToFloat8(1.0), ops.Mul(ops.Tanh(float8.ToFloat8(-1.0)), ops.Tanh(float8.ToFloat8(-1.0))))},
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

func TestFloat8Ops_SigmoidGrad(t *testing.T) {
	ops := Float8Ops{}
	tests := []struct {
		name     string
		x        float8.Float8
		expected float8.Float8
	}{
		{"zero", float8.ToFloat8(0.0), float8.ToFloat8(0.25)},
		{"positive", float8.ToFloat8(1.0), ops.Mul(ops.Sigmoid(float8.ToFloat8(1.0)), ops.Sub(float8.ToFloat8(1.0), ops.Sigmoid(float8.ToFloat8(1.0))))},
		{"negative", float8.ToFloat8(-1.0), ops.Mul(ops.Sigmoid(float8.ToFloat8(-1.0)), ops.Sub(float8.ToFloat8(1.0), ops.Sigmoid(float8.ToFloat8(-1.0))))},
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

func TestFloat8Ops_FromFloat32(t *testing.T) {
	ops := Float8Ops{}
	tests := []struct {
		name     string
		f        float32
		expected float8.Float8
	}{
		{"zero", 0.0, float8.ToFloat8(0.0)},
		{"positive", 1.0, float8.ToFloat8(1.0)},
		{"negative", -1.0, float8.ToFloat8(-1.0)},
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

func TestFloat8Ops_IsZero(t *testing.T) {
	ops := Float8Ops{}
	tests := []struct {
		name     string
		v        float8.Float8
		expected bool
	}{
		{"zero", float8.ToFloat8(0.0), true},
		{"non-zero", float8.ToFloat8(1.0), false},
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

func TestFloat8Ops_Exp(t *testing.T) {
	ops := Float8Ops{}
	tests := []struct {
		name     string
		x        float8.Float8
		expected float8.Float8
	}{
		{"zero", float8.ToFloat8(0.0), float8.ToFloat8(float32(math.Exp(0.0)))},
		{"positive", float8.ToFloat8(1.0), float8.ToFloat8(float32(math.Exp(1.0)))},
		{"negative", float8.ToFloat8(-1.0), float8.ToFloat8(float32(math.Exp(-1.0)))},
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

func TestFloat8Ops_Log(t *testing.T) {
	ops := Float8Ops{}
	tests := []struct {
		name     string
		x        float8.Float8
		expected float8.Float8
	}{
		{"one", float8.ToFloat8(1.0), float8.ToFloat8(float32(math.Log(1.0)))},
		{"positive", float8.ToFloat8(2.0), float8.ToFloat8(float32(math.Log(2.0)))},
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

func TestFloat8Ops_Pow(t *testing.T) {
	ops := Float8Ops{}
	tests := []struct {
		name           string
		base, exponent float8.Float8
		expected       float8.Float8
	}{
		{"base 2 exp 3", float8.ToFloat8(2.0), float8.ToFloat8(3.0), float8.ToFloat8(8.0)},
		{"base 5 exp 0", float8.ToFloat8(5.0), float8.ToFloat8(0.0), float8.ToFloat8(1.0)},
		{"base 4 exp 0.5", float8.ToFloat8(4.0), float8.ToFloat8(0.5), float8.ToFloat8(2.0)},
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

func TestFloat8Ops_ReLU(t *testing.T) {
	ops := Float8Ops{}
	tests := []struct {
		name     string
		x        float8.Float8
		expected float8.Float8
	}{
		{"positive", float8.ToFloat8(2.5), float8.ToFloat8(2.5)},
		{"negative", float8.ToFloat8(-1.5), float8.ToFloat8(0.0)},
		{"zero", float8.ToFloat8(0.0), float8.ToFloat8(0.0)},
		{"small positive", float8.ToFloat8(0.1), float8.ToFloat8(0.1)},
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

func TestFloat8Ops_LeakyReLU(t *testing.T) {
	ops := Float8Ops{}
	tests := []struct {
		name     string
		x        float8.Float8
		alpha    float64
		expected float32
		epsilon  float32
	}{
		{"positive", float8.ToFloat8(2.0), 0.1, 2.0, 0.1},
		{"negative", float8.ToFloat8(-2.0), 0.1, -0.2, 0.1},
		{"zero", float8.ToFloat8(0.0), 0.1, 0.0, 0.1},
		{"negative with different alpha", float8.ToFloat8(-1.0), 0.2, -0.2, 0.1},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := ops.LeakyReLU(tt.x, tt.alpha)
			resultFloat := result.ToFloat32()
			if math.Abs(float64(resultFloat-tt.expected)) > float64(tt.epsilon) {
				t.Errorf("LeakyReLU(%v, %v): expected %v, got %v", tt.x, tt.alpha, tt.expected, resultFloat)
			}
		})
	}
}

func TestFloat8Ops_ReLUGrad(t *testing.T) {
	ops := Float8Ops{}
	tests := []struct {
		name     string
		x        float8.Float8
		expected float8.Float8
	}{
		{"positive", float8.ToFloat8(2.5), float8.ToFloat8(1.0)},
		{"negative", float8.ToFloat8(-1.5), float8.ToFloat8(0.0)},
		{"zero", float8.ToFloat8(0.0), float8.ToFloat8(0.0)},
		{"small positive", float8.ToFloat8(0.1), float8.ToFloat8(1.0)},
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

func TestFloat8Ops_LeakyReLUGrad(t *testing.T) {
	ops := Float8Ops{}
	tests := []struct {
		name     string
		x        float8.Float8
		alpha    float64
		expected float32
		epsilon  float32
	}{
		{"positive", float8.ToFloat8(2.0), 0.1, 1.0, 0.1},
		{"negative", float8.ToFloat8(-2.0), 0.1, 0.1, 0.1},
		{"zero", float8.ToFloat8(0.0), 0.1, 0.1, 0.1},
		{"negative with different alpha", float8.ToFloat8(-1.0), 0.2, 0.2, 0.1},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := ops.LeakyReLUGrad(tt.x, tt.alpha)
			resultFloat := result.ToFloat32()
			if math.Abs(float64(resultFloat-tt.expected)) > float64(tt.epsilon) {
				t.Errorf("LeakyReLUGrad(%v, %v): expected %v, got %v", tt.x, tt.alpha, tt.expected, resultFloat)
			}
		})
	}
}

func TestFloat8Ops_ToFloat32(t *testing.T) {
	ops := Float8Ops{}
	tests := []struct {
		name     string
		x        float8.Float8
		expected float32
		epsilon  float32
	}{
		{"positive", float8.ToFloat8(2.5), 2.5, 0.1},
		{"negative", float8.ToFloat8(-1.5), -1.5, 0.1},
		{"zero", float8.ToFloat8(0.0), 0.0, 0.1},
		{"small", float8.ToFloat8(0.1), 0.1, 0.1},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := ops.ToFloat32(tt.x)
			if math.Abs(float64(result-tt.expected)) > float64(tt.epsilon) {
				t.Errorf("ToFloat32(%v): expected %v, got %v", tt.x, tt.expected, result)
			}
		})
	}
}

func TestFloat8Ops_Abs(t *testing.T) {
	ops := Float8Ops{}
	tests := []struct {
		name     string
		x        float8.Float8
		expected float8.Float8
	}{
		{"positive", float8.ToFloat8(2.5), float8.ToFloat8(2.5)},
		{"negative", float8.ToFloat8(-1.5), float8.ToFloat8(1.5)},
		{"zero", float8.ToFloat8(0.0), float8.ToFloat8(0.0)},
		{"small negative", float8.ToFloat8(-0.1), float8.ToFloat8(0.1)},
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

func TestFloat8Ops_Sum(t *testing.T) {
	ops := Float8Ops{}
	tests := []struct {
		name     string
		s        []float8.Float8
		expected float32
		epsilon  float32
	}{
		{"empty slice", []float8.Float8{}, 0.0, 0.1},
		{"single element", []float8.Float8{float8.ToFloat8(2.5)}, 2.5, 0.1},
		{"multiple positive", []float8.Float8{float8.ToFloat8(1.0), float8.ToFloat8(2.0), float8.ToFloat8(3.0)}, 6.0, 0.1},
		{"mixed signs", []float8.Float8{float8.ToFloat8(1.0), float8.ToFloat8(-2.0), float8.ToFloat8(3.0)}, 2.0, 0.1},
		{"all zeros", []float8.Float8{float8.ToFloat8(0.0), float8.ToFloat8(0.0), float8.ToFloat8(0.0)}, 0.0, 0.1},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := ops.Sum(tt.s)
			resultFloat := result.ToFloat32()
			if math.Abs(float64(resultFloat-tt.expected)) > float64(tt.epsilon) {
				t.Errorf("Sum(%v): expected %v, got %v", tt.s, tt.expected, resultFloat)
			}
		})
	}
}
