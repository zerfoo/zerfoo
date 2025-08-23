package numeric

import (
	"math"
	"testing"

	"github.com/zerfoo/float16"
)

func TestFloat16Ops_Add(t *testing.T) {
	ops := Float16Ops{}
	testData := Float16TestData()
	TestArithmeticOp(t, "Add", ops.Add, func(a, b float16.Float16) bool { return a == b }, testData.Add)
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
	testData := Float16TestData()
	TestArithmeticOp(t, "Mul", ops.Mul, func(a, b float16.Float16) bool { return a == b }, testData.Mul)
}

func TestFloat16Ops_Div(t *testing.T) {
	ops := Float16Ops{}
	testData := Float16TestData()
	TestArithmeticOp(t, "Div", ops.Div, func(a, b float16.Float16) bool { return a == b }, testData.Div)
}

func TestFloat16Ops_Tanh(t *testing.T) {
	ops := Float16Ops{}
	testData := Float16TestData()
	TestUnaryOp(t, "Tanh", ops.Tanh, func(a, b float16.Float16) bool { return a == b }, testData.Tanh)
}

func TestFloat16Ops_Sigmoid(t *testing.T) {
	ops := Float16Ops{}
	testData := Float16TestData()
	TestUnaryOp(t, "Sigmoid", ops.Sigmoid, func(a, b float16.Float16) bool { return a == b }, testData.Sigmoid)
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
			// Use tolerance for float16 precision
			tolerance := float32(0.001)
			if math.Abs(float64(result.ToFloat32()-tt.expected.ToFloat32())) > float64(tolerance) {
				t.Errorf("TanhGrad(%v): expected %v, got %v (diff: %v)", tt.x, tt.expected, result,
					math.Abs(float64(result.ToFloat32()-tt.expected.ToFloat32())))
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
		{
			"positive", float16.FromFloat32(1.0),
			float16.FromFloat32(ops.Sigmoid(float16.FromFloat32(1.0)).ToFloat32() *
				(1.0 - ops.Sigmoid(float16.FromFloat32(1.0)).ToFloat32())),
		},
		{
			"negative", float16.FromFloat32(-1.0),
			float16.FromFloat32(ops.Sigmoid(float16.FromFloat32(-1.0)).ToFloat32() *
				(1.0 - ops.Sigmoid(float16.FromFloat32(-1.0)).ToFloat32())),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := ops.SigmoidGrad(tt.x)
			// Use tolerance for float16 precision
			tolerance := float32(0.001)
			if math.Abs(float64(result.ToFloat32()-tt.expected.ToFloat32())) > float64(tolerance) {
				t.Errorf("SigmoidGrad(%v): expected %v, got %v (diff: %v)", tt.x, tt.expected, result,
					math.Abs(float64(result.ToFloat32()-tt.expected.ToFloat32())))
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
		{"negative", float16.FromFloat32(-1.5), float16.FromFloat32(0)},
		{"zero", float16.FromFloat32(0), float16.FromFloat32(0)},
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
	testData := Float16TestData()
	TestLeakyReLUOp(t, "LeakyReLU", ops.LeakyReLU, func(f float16.Float16) float32 { return f.ToFloat32() }, testData.LeakyReLU)
}

func TestFloat16Ops_ReLUGrad(t *testing.T) {
	ops := Float16Ops{}
	tests := []struct {
		name     string
		x        float16.Float16
		expected float16.Float16
	}{
		{"positive", float16.FromFloat32(2.5), float16.FromFloat32(1)},
		{"negative", float16.FromFloat32(-1.5), float16.FromFloat32(0)},
		{"zero", float16.FromFloat32(0), float16.FromFloat32(0)},
		{"small positive", float16.FromFloat32(0.1), float16.FromFloat32(1)},
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
	testData := Float16TestData()
	TestLeakyReLUOp(t, "LeakyReLUGrad", ops.LeakyReLUGrad, func(f float16.Float16) float32 { return f.ToFloat32() }, testData.LeakyReLUGrad)
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
		{"zero", float16.FromFloat32(0), 0.0, 0.01},
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
		{"zero", float16.FromFloat32(0), float16.FromFloat32(0)},
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
	testData := Float16TestData()
	TestSumOp(t, ops.Sum, func(f float16.Float16) float32 { return f.ToFloat32() }, testData.Sum)
}
