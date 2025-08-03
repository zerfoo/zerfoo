package numeric

import (
	"math"
	"testing"

	"github.com/zerfoo/float16"
)

func TestFloat16Ops_Add(t *testing.T) {
	ops := Float16Ops{}
	tests := []struct {
		name           string
		a, b, expected float16.Float16
	}{
		{"positive numbers", float16.FromFloat32(1.0), float16.FromFloat32(2.0), float16.FromFloat32(3.0)},
		{"negative numbers", float16.FromFloat32(-1.0), float16.FromFloat32(-2.0), float16.FromFloat32(-3.0)},
		{"mixed numbers", float16.FromFloat32(1.0), float16.FromFloat32(-2.0), float16.FromFloat32(-1.0)},
		{"zero", float16.FromFloat32(0.0), float16.FromFloat32(0.0), float16.FromFloat32(0.0)},
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

func TestFloat16Ops_Sub(t *testing.T) {
	ops := Float16Ops{}
	tests := []struct {
		name           string
		a, b, expected float16.Float16
	}{
		{"positive numbers", float16.FromFloat32(3.0), float16.FromFloat32(1.0), float16.FromFloat32(2.0)},
		{"negative numbers", float16.FromFloat32(-1.0), float16.FromFloat32(-2.0), float16.FromFloat32(1.0)},
		{"mixed numbers", float16.FromFloat32(1.0), float16.FromFloat32(-2.0), float16.FromFloat32(3.0)},
		{"zero", float16.FromFloat32(0.0), float16.FromFloat32(0.0), float16.FromFloat32(0.0)},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := ops.Sub(tt.a, tt.b)
			if !ops.IsZero(result) && result != tt.expected {
				t.Errorf("Sub(%v, %v): expected %v, got %v", tt.a, tt.b, tt.expected, result)
			}
		})
	}
}

func TestFloat16Ops_Mul(t *testing.T) {
	ops := Float16Ops{}
	tests := []struct {
		name           string
		a, b, expected float16.Float16
	}{
		{"positive numbers", float16.FromFloat32(2.0), float16.FromFloat32(3.0), float16.FromFloat32(6.0)},
		{"negative numbers", float16.FromFloat32(-2.0), float16.FromFloat32(-3.0), float16.FromFloat32(6.0)},
		{"mixed numbers", float16.FromFloat32(2.0), float16.FromFloat32(-3.0), float16.FromFloat32(-6.0)},
		{"zero", float16.FromFloat32(0.0), float16.FromFloat32(5.0), float16.FromFloat32(0.0)},
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

func TestFloat16Ops_Div(t *testing.T) {
	ops := Float16Ops{}
	tests := []struct {
		name           string
		a, b, expected float16.Float16
	}{
		{"positive numbers", float16.FromFloat32(6.0), float16.FromFloat32(3.0), float16.FromFloat32(2.0)},
		{"negative numbers", float16.FromFloat32(-6.0), float16.FromFloat32(-3.0), float16.FromFloat32(2.0)},
		{"mixed numbers", float16.FromFloat32(6.0), float16.FromFloat32(-3.0), float16.FromFloat32(-2.0)},
		{"divide by one", float16.FromFloat32(5.0), float16.FromFloat32(1.0), float16.FromFloat32(5.0)},
		{"zero dividend", float16.FromFloat32(0.0), float16.FromFloat32(5.0), float16.FromFloat32(0.0)},
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

func TestFloat16Ops_Tanh(t *testing.T) {
	ops := Float16Ops{}
	tests := []struct {
		name     string
		x        float16.Float16
		expected float16.Float16
	}{
		{"zero", float16.FromFloat32(0.0), float16.FromFloat32(0.0)},
		{"positive", float16.FromFloat32(1.0), float16.FromFloat32(float32(math.Tanh(1.0)))},
		{"negative", float16.FromFloat32(-1.0), float16.FromFloat32(float32(math.Tanh(-1.0)))},
		{"large positive", float16.FromFloat32(100.0), float16.FromFloat32(float32(math.Tanh(100.0)))},
		{"large negative", float16.FromFloat32(-100.0), float16.FromFloat32(float32(math.Tanh(-100.0)))},
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

func TestFloat16Ops_Sigmoid(t *testing.T) {
	ops := Float16Ops{}
	tests := []struct {
		name     string
		x        float16.Float16
		expected float16.Float16
	}{
		{"zero", float16.FromFloat32(0.0), float16.FromFloat32(0.5)},
		{"positive", float16.FromFloat32(1.0), float16.FromFloat32(1.0 / (1.0 + float32(math.Exp(-1.0))))},
		{"negative", float16.FromFloat32(-1.0), float16.FromFloat32(1.0 / (1.0 + float32(math.Exp(1.0))))},
		{"large positive", float16.FromFloat32(100.0), float16.FromFloat32(1.0)},
		{"large negative", float16.FromFloat32(-100.0), float16.FromFloat32(0.0)},
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

func TestFloat16Ops_TanhGrad(t *testing.T) {
	ops := Float16Ops{}
	tests := []struct {
		name     string
		x        float16.Float16
		expected float16.Float16
	}{
		{"zero", float16.FromFloat32(0.0), float16.FromFloat32(1.0)},
		{"positive", float16.FromFloat32(1.0), float16.FromFloat32(1.0 - float32(math.Pow(math.Tanh(1.0), 2)))},
		{"negative", float16.FromFloat32(-1.0), float16.FromFloat32(1.0 - float32(math.Pow(math.Tanh(-1.0), 2)))},
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

func TestFloat16Ops_SigmoidGrad(t *testing.T) {
	ops := Float16Ops{}
	tests := []struct {
		name     string
		x        float16.Float16
		expected float16.Float16
	}{
		{"zero", float16.FromFloat32(0.0), float16.FromFloat32(0.25)},
		{"positive", float16.FromFloat32(1.0), float16.FromFloat32(ops.Sigmoid(float16.FromFloat32(1.0)).ToFloat32() * (1.0 - ops.Sigmoid(float16.FromFloat32(1.0)).ToFloat32()))},
		{"negative", float16.FromFloat32(-1.0), float16.FromFloat32(ops.Sigmoid(float16.FromFloat32(-1.0)).ToFloat32() * (1.0 - ops.Sigmoid(float16.FromFloat32(-1.0)).ToFloat32()))},
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

func TestFloat16Ops_FromFloat32(t *testing.T) {
	ops := Float16Ops{}
	tests := []struct {
		name     string
		f        float32
		expected float16.Float16
	}{
		{"zero", 0.0, float16.FromFloat32(0.0)},
		{"positive", 1.0, float16.FromFloat32(1.0)},
		{"negative", -1.0, float16.FromFloat32(-1.0)},
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

func TestFloat16Ops_IsZero(t *testing.T) {
	ops := Float16Ops{}
	tests := []struct {
		name     string
		v        float16.Float16
		expected bool
	}{
		{"zero", float16.FromFloat32(0.0), true},
		{"non-zero", float16.FromFloat32(1.0), false},
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

func TestFloat16Ops_Exp(t *testing.T) {
	ops := Float16Ops{}
	tests := []struct {
		name     string
		x        float16.Float16
		expected float16.Float16
	}{
		{"zero", float16.FromFloat32(0.0), float16.FromFloat32(float32(math.Exp(0.0)))},
		{"positive", float16.FromFloat32(1.0), float16.FromFloat32(float32(math.Exp(1.0)))},
		{"negative", float16.FromFloat32(-1.0), float16.FromFloat32(float32(math.Exp(-1.0)))},
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

func TestFloat16Ops_Log(t *testing.T) {
	ops := Float16Ops{}
	tests := []struct {
		name     string
		x        float16.Float16
		expected float16.Float16
	}{
		{"one", float16.FromFloat32(1.0), float16.FromFloat32(float32(math.Log(1.0)))},
		{"positive", float16.FromFloat32(2.0), float16.FromFloat32(float32(math.Log(2.0)))},
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

func TestFloat16Ops_Pow(t *testing.T) {
	ops := Float16Ops{}
	tests := []struct {
		name           string
		base, exponent float16.Float16
		expected       float32
		epsilon        float32
	}{
		{"2^3", float16.FromFloat32(2.0), float16.FromFloat32(3.0), 8.0, 0.1},
		{"3^2", float16.FromFloat32(3.0), float16.FromFloat32(2.0), 9.0, 0.1},
		{"1^5", float16.FromFloat32(1.0), float16.FromFloat32(5.0), 1.0, 0.1},
		{"0^2", float16.FromFloat32(0.0), float16.FromFloat32(2.0), 0.0, 0.1},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := ops.Pow(tt.base, tt.exponent)
			resultFloat := result.ToFloat32()
			if math.Abs(float64(resultFloat-tt.expected)) > float64(tt.epsilon) {
				t.Errorf("Pow(%v, %v): expected %v, got %v", tt.base, tt.exponent, tt.expected, resultFloat)
			}
		})
	}
}

func TestFloat16Ops_ReLU(t *testing.T) {
	ops := Float16Ops{}
	tests := []struct {
		name     string
		x        float16.Float16
		expected float16.Float16
	}{
		{"positive", float16.FromFloat32(2.5), float16.FromFloat32(2.5)},
		{"negative", float16.FromFloat32(-1.5), float16.FromInt(0)},
		{"zero", float16.FromInt(0), float16.FromInt(0)},
		{"small positive", float16.FromFloat32(0.1), float16.FromFloat32(0.1)},
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

func TestFloat16Ops_LeakyReLU(t *testing.T) {
	ops := Float16Ops{}
	tests := []struct {
		name     string
		x        float16.Float16
		alpha    float64
		expected float32
		epsilon  float32
	}{
		{"positive", float16.FromFloat32(2.0), 0.1, 2.0, 0.01},
		{"negative", float16.FromFloat32(-2.0), 0.1, -0.2, 0.01},
		{"zero", float16.FromInt(0), 0.1, 0.0, 0.01},
		{"negative with different alpha", float16.FromFloat32(-1.0), 0.2, -0.2, 0.01},
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

func TestFloat16Ops_ReLUGrad(t *testing.T) {
	ops := Float16Ops{}
	tests := []struct {
		name     string
		x        float16.Float16
		expected float16.Float16
	}{
		{"positive", float16.FromFloat32(2.5), float16.FromInt(1)},
		{"negative", float16.FromFloat32(-1.5), float16.FromInt(0)},
		{"zero", float16.FromInt(0), float16.FromInt(0)},
		{"small positive", float16.FromFloat32(0.1), float16.FromInt(1)},
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

func TestFloat16Ops_LeakyReLUGrad(t *testing.T) {
	ops := Float16Ops{}
	tests := []struct {
		name     string
		x        float16.Float16
		alpha    float64
		expected float32
		epsilon  float32
	}{
		{"positive", float16.FromFloat32(2.0), 0.1, 1.0, 0.01},
		{"negative", float16.FromFloat32(-2.0), 0.1, 0.1, 0.01},
		{"zero", float16.FromInt(0), 0.1, 0.1, 0.01},
		{"negative with different alpha", float16.FromFloat32(-1.0), 0.2, 0.2, 0.01},
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

func TestFloat16Ops_ToFloat32(t *testing.T) {
	ops := Float16Ops{}
	tests := []struct {
		name     string
		x        float16.Float16
		expected float32
		epsilon  float32
	}{
		{"positive", float16.FromFloat32(2.5), 2.5, 0.01},
		{"negative", float16.FromFloat32(-1.5), -1.5, 0.01},
		{"zero", float16.FromInt(0), 0.0, 0.01},
		{"small", float16.FromFloat32(0.1), 0.1, 0.01},
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

func TestFloat16Ops_Abs(t *testing.T) {
	ops := Float16Ops{}
	tests := []struct {
		name     string
		x        float16.Float16
		expected float16.Float16
	}{
		{"positive", float16.FromFloat32(2.5), float16.FromFloat32(2.5)},
		{"negative", float16.FromFloat32(-1.5), float16.FromFloat32(1.5)},
		{"zero", float16.FromInt(0), float16.FromInt(0)},
		{"small negative", float16.FromFloat32(-0.1), float16.FromFloat32(0.1)},
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

func TestFloat16Ops_Sum(t *testing.T) {
	ops := Float16Ops{}
	tests := []struct {
		name     string
		s        []float16.Float16
		expected float32
		epsilon  float32
	}{
		{"empty slice", []float16.Float16{}, 0.0, 0.01},
		{"single element", []float16.Float16{float16.FromFloat32(2.5)}, 2.5, 0.01},
		{"multiple positive", []float16.Float16{float16.FromFloat32(1.0), float16.FromFloat32(2.0), float16.FromFloat32(3.0)}, 6.0, 0.01},
		{"mixed signs", []float16.Float16{float16.FromFloat32(1.0), float16.FromFloat32(-2.0), float16.FromFloat32(3.0)}, 2.0, 0.01},
		{"all zeros", []float16.Float16{float16.FromInt(0), float16.FromInt(0), float16.FromInt(0)}, 0.0, 0.01},
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
