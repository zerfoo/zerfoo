package numeric

import "math"

// Uint8Ops provides the implementation of the Arithmetic interface for the uint8 type.
type Uint8Ops struct{}

// Add performs element-wise addition.
func (ops Uint8Ops) Add(a, b uint8) uint8 { return a + b }

// Sub performs element-wise subtraction.
func (ops Uint8Ops) Sub(a, b uint8) uint8 { return a - b }

// Mul performs element-wise multiplication.
func (ops Uint8Ops) Mul(a, b uint8) uint8 { return a * b }

// Div performs element-wise division.
func (ops Uint8Ops) Div(a, b uint8) uint8 {
	if b == 0 {
		return 0 // Avoid panic
	}
	return a / b
}

// Tanh computes the hyperbolic tangent of x.
func (ops Uint8Ops) Tanh(x uint8) uint8 {
	return uint8(math.Tanh(float64(x)))
}

// Sigmoid computes the sigmoid function of x.
func (ops Uint8Ops) Sigmoid(x uint8) uint8 {
	return uint8(1.0 / (1.0 + math.Exp(float64(-x))))
}

// ReLU computes the Rectified Linear Unit function.
func (ops Uint8Ops) ReLU(x uint8) uint8 {
	if x > 0 { // This condition is always true for uint8 > 0, but good for clarity
		return x
	}
	return 0
}

// LeakyReLU computes the Leaky Rectified Linear Unit function.
func (ops Uint8Ops) LeakyReLU(x uint8, alpha float64) uint8 {
	if x > 0 {
		return x
	}
	return uint8(float64(x) * alpha) // This will be 0 for uint8
}

// TanhGrad computes the gradient of the hyperbolic tangent function.
func (ops Uint8Ops) TanhGrad(x uint8) uint8 {
	tanhX := math.Tanh(float64(x))
	return uint8(1.0 - (tanhX * tanhX))
}

// SigmoidGrad computes the gradient of the sigmoid function.
func (ops Uint8Ops) SigmoidGrad(x uint8) uint8 {
	sigX := 1.0 / (1.0 + math.Exp(float64(-x)))
	return uint8(sigX * (1.0 - sigX))
}

// ReLUGrad computes the gradient of the Rectified Linear Unit function.
func (ops Uint8Ops) ReLUGrad(x uint8) uint8 {
	if x > 0 {
		return 1
	}
	return 0
}

// LeakyReLUGrad computes the gradient of the Leaky Rectified Linear Unit function.
func (ops Uint8Ops) LeakyReLUGrad(x uint8, alpha float64) uint8 {
	if x > 0 {
		return 1
	}
	return uint8(alpha)
}

// FromFloat32 converts a float32 to a uint8.
func (ops Uint8Ops) FromFloat32(f float32) uint8 {
	return uint8(f)
}

// FromFloat64 converts a float64 to a uint8.
func (ops Uint8Ops) FromFloat64(f float64) uint8 {
	return uint8(f)
}

// One returns a uint8 with value 1.
func (ops Uint8Ops) One() uint8 {
	return 1
}

// IsZero checks if the given uint8 value is zero.
func (ops Uint8Ops) IsZero(v uint8) bool {
	return v == 0
}

// Abs computes the absolute value of x.
func (ops Uint8Ops) Abs(x uint8) uint8 {
	return x // Always non-negative
}

// Sum computes the sum of elements in a slice.
func (ops Uint8Ops) Sum(s []uint8) uint8 {
	var sum uint8
	for _, v := range s {
		sum += v
	}
	return sum
}

// Exp computes the exponential of x.
func (ops Uint8Ops) Exp(x uint8) uint8 {
	return uint8(math.Exp(float64(x)))
}

// Log computes the natural logarithm of x.
func (ops Uint8Ops) Log(x uint8) uint8 {
	return uint8(math.Log(float64(x)))
}

// Pow computes base raised to the power of exponent.
func (ops Uint8Ops) Pow(base, exponent uint8) uint8 {
	return uint8(math.Pow(float64(base), float64(exponent)))
}

// Sqrt computes the square root of x.
func (ops Uint8Ops) Sqrt(x uint8) uint8 {
	return uint8(math.Sqrt(float64(x)))
}

// GreaterThan checks if a is greater than b.
func (ops Uint8Ops) GreaterThan(a, b uint8) bool {
	return a > b
}
