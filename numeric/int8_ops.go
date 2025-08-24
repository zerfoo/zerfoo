package numeric

import "math"

// Int8Ops provides the implementation of the Arithmetic interface for the int8 type.
type Int8Ops struct{}

// Add performs element-wise addition.
func (ops Int8Ops) Add(a, b int8) int8 { return a + b }

// Sub performs element-wise subtraction.
func (ops Int8Ops) Sub(a, b int8) int8 { return a - b }

// Mul performs element-wise multiplication.
func (ops Int8Ops) Mul(a, b int8) int8 { return a * b }

// Div performs element-wise division.
func (ops Int8Ops) Div(a, b int8) int8 {
	if b == 0 {
		return 0 // Avoid panic
	}
	return a / b
}

// Tanh computes the hyperbolic tangent of x.
func (ops Int8Ops) Tanh(x int8) int8 {
	return int8(math.Tanh(float64(x)))
}

// Sigmoid computes the sigmoid function of x.
func (ops Int8Ops) Sigmoid(x int8) int8 {
	return int8(1.0 / (1.0 + math.Exp(float64(-x))))
}

// ReLU computes the Rectified Linear Unit function.
func (ops Int8Ops) ReLU(x int8) int8 {
	if x > 0 {
		return x
	}
	return 0
}

// LeakyReLU computes the Leaky Rectified Linear Unit function.
func (ops Int8Ops) LeakyReLU(x int8, alpha float64) int8 {
	if x > 0 {
		return x
	}
	return int8(float64(x) * alpha)
}

// TanhGrad computes the gradient of the hyperbolic tangent function.
func (ops Int8Ops) TanhGrad(x int8) int8 {
	tanhX := math.Tanh(float64(x))
	return int8(1.0 - (tanhX * tanhX))
}

// SigmoidGrad computes the gradient of the sigmoid function.
func (ops Int8Ops) SigmoidGrad(x int8) int8 {
	sigX := 1.0 / (1.0 + math.Exp(float64(-x)))
	return int8(sigX * (1.0 - sigX))
}

// ReLUGrad computes the gradient of the Rectified Linear Unit function.
func (ops Int8Ops) ReLUGrad(x int8) int8 {
	if x > 0 {
		return 1
	}
	return 0
}

// LeakyReLUGrad computes the gradient of the Leaky Rectified Linear Unit function.
func (ops Int8Ops) LeakyReLUGrad(x int8, alpha float64) int8 {
	if x > 0 {
		return 1
	}
	return int8(alpha)
}

// FromFloat32 converts a float32 to an int8.
func (ops Int8Ops) FromFloat32(f float32) int8 {
	return int8(f)
}

// FromFloat64 converts a float64 to an int8.
func (ops Int8Ops) FromFloat64(f float64) int8 {
	return int8(f)
}

// One returns an int8 with value 1.
func (ops Int8Ops) One() int8 {
	return 1
}

// IsZero checks if the given int8 value is zero.
func (ops Int8Ops) IsZero(v int8) bool {
	return v == 0
}

// Abs computes the absolute value of x.
func (ops Int8Ops) Abs(x int8) int8 {
	if x < 0 {
		return -x
	}
	return x
}

// Sum computes the sum of elements in a slice.
func (ops Int8Ops) Sum(s []int8) int8 {
	var sum int8
	for _, v := range s {
		sum += v
	}
	return sum
}

// Exp computes the exponential of x.
func (ops Int8Ops) Exp(x int8) int8 {
	return int8(math.Exp(float64(x)))
}

// Log computes the natural logarithm of x.
func (ops Int8Ops) Log(x int8) int8 {
	return int8(math.Log(float64(x)))
}

// Pow computes base raised to the power of exponent.
func (ops Int8Ops) Pow(base, exponent int8) int8 {
	return int8(math.Pow(float64(base), float64(exponent)))
}

// Sqrt computes the square root of x.
func (ops Int8Ops) Sqrt(x int8) int8 {
	return int8(math.Sqrt(float64(x)))
}

// GreaterThan checks if a is greater than b.
func (ops Int8Ops) GreaterThan(a, b int8) bool {
	return a > b
}
