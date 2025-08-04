package numeric

import (
	"math"

	"github.com/zerfoo/float8"
)

// Float8Ops provides the implementation of the Arithmetic interface for the float8.Float8 type.
type Float8Ops struct{}

// Add performs element-wise addition.
func (ops Float8Ops) Add(a, b float8.Float8) float8.Float8 { return float8.Add(a, b) }

// Sub performs element-wise subtraction.
func (ops Float8Ops) Sub(a, b float8.Float8) float8.Float8 { return float8.Sub(a, b) }

// Mul performs element-wise multiplication.
func (ops Float8Ops) Mul(a, b float8.Float8) float8.Float8 { return float8.Mul(a, b) }

// Div performs element-wise division.
func (ops Float8Ops) Div(a, b float8.Float8) float8.Float8 { return float8.Div(a, b) }

// Tanh computes the hyperbolic tangent of x.
func (ops Float8Ops) Tanh(x float8.Float8) float8.Float8 {
	f32 := x.ToFloat32()

	return float8.ToFloat8(float32(math.Tanh(float64(f32))))
}

// Sigmoid computes the sigmoid function of x.
func (ops Float8Ops) Sigmoid(x float8.Float8) float8.Float8 {
	f32 := x.ToFloat32()

	return float8.ToFloat8(1.0 / (1.0 + float32(math.Exp(float64(-f32)))))
}

// TanhGrad computes the gradient of the hyperbolic tangent function.
func (ops Float8Ops) TanhGrad(x float8.Float8) float8.Float8 {
	// TanhGrad is 1 - tanh(x)^2
	tanhX := ops.Tanh(x)
	tanhX2 := ops.Mul(tanhX, tanhX)
	one := float8.ToFloat8(1.0)

	return ops.Sub(one, tanhX2)
}

// SigmoidGrad computes the gradient of the sigmoid function.
func (ops Float8Ops) SigmoidGrad(x float8.Float8) float8.Float8 {
	// SigmoidGrad is sigmoid(x) * (1 - sigmoid(x))
	sigX := ops.Sigmoid(x)
	one := float8.ToFloat8(1.0)
	oneMinusSigX := ops.Sub(one, sigX)

	return ops.Mul(sigX, oneMinusSigX)
}

// ReLU computes the Rectified Linear Unit function.
func (ops Float8Ops) ReLU(x float8.Float8) float8.Float8 {
	if x.ToFloat32() > 0 {
		return x
	}

	return float8.ToFloat8(0.0)
}

// LeakyReLU computes the Leaky Rectified Linear Unit function.
func (ops Float8Ops) LeakyReLU(x float8.Float8, alpha float64) float8.Float8 {
	if x.ToFloat32() > 0 {
		return x
	}

	return ops.Mul(x, float8.ToFloat8(float32(alpha)))
}

// ReLUGrad computes the gradient of the Rectified Linear Unit function.
func (ops Float8Ops) ReLUGrad(x float8.Float8) float8.Float8 {
	one := float8.ToFloat8(1.0)
	if x.ToFloat32() > 0 {
		return one
	}

	return float8.ToFloat8(0.0)
}

// LeakyReLUGrad computes the gradient of the Leaky Rectified Linear Unit function.
func (ops Float8Ops) LeakyReLUGrad(x float8.Float8, alpha float64) float8.Float8 {
	one := float8.ToFloat8(1.0)
	if x.ToFloat32() > 0 {
		return one
	}

	return float8.ToFloat8(float32(alpha))
}

// FromFloat32 converts a float32 to a float8.Float8.
func (ops Float8Ops) FromFloat32(f float32) float8.Float8 {
	return float8.ToFloat8(f)
}

// ToFloat32 converts a float8.Float8 to a float32.
func (ops Float8Ops) ToFloat32(t float8.Float8) float32 {
	return t.ToFloat32()
}

// IsZero checks if the given float8.Float8 value is zero.
func (ops Float8Ops) IsZero(v float8.Float8) bool {
	return v.IsZero()
}

// Exp computes the exponential of x.
func (ops Float8Ops) Exp(x float8.Float8) float8.Float8 {
	f32 := x.ToFloat32()

	return float8.ToFloat8(float32(math.Exp(float64(f32))))
}

// Log computes the natural logarithm of x.
func (ops Float8Ops) Log(x float8.Float8) float8.Float8 {
	f32 := x.ToFloat32()

	return float8.ToFloat8(float32(math.Log(float64(f32))))
}

// Pow computes base raised to the power of exponent.
func (ops Float8Ops) Pow(base, exponent float8.Float8) float8.Float8 {
	f32Base := base.ToFloat32()
	f32Exp := exponent.ToFloat32()

	return float8.ToFloat8(float32(math.Pow(float64(f32Base), float64(f32Exp))))
}

// Abs computes the absolute value of x.
func (ops Float8Ops) Abs(x float8.Float8) float8.Float8 {
	if x.ToFloat32() < 0 {
		return float8.ToFloat8(-x.ToFloat32())
	}

	return x
}

// Sqrt computes the square root of x.
func (ops Float8Ops) Sqrt(x float8.Float8) float8.Float8 {
	return float8.ToFloat8(float32(math.Sqrt(float64(x.ToFloat32()))))
}

// Sum computes the sum of elements in a slice.
func (ops Float8Ops) Sum(s []float8.Float8) float8.Float8 {
	var sum float8.Float8
	for _, v := range s {
		sum = float8.Add(sum, v)
	}

	return sum
}

// GreaterThan checks if a is greater than b.
func (ops Float8Ops) GreaterThan(a, b float8.Float8) bool {
	return a.ToFloat32() > b.ToFloat32()
}

// One returns a float8.Float8 with value 1.
func (ops Float8Ops) One() float8.Float8 {
	return float8.ToFloat8(1.0)
}

// FromFloat64 converts a float64 to a float8.Float8.
func (ops Float8Ops) FromFloat64(f float64) float8.Float8 {
	return float8.FromFloat64(f)
}
