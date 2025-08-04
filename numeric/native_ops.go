package numeric

import "math"

// Float32Ops provides the implementation of the Arithmetic interface for the float32 type.
type Float32Ops struct{}

// Add performs element-wise addition.
func (ops Float32Ops) Add(a, b float32) float32 { return a + b }

// Sub performs element-wise subtraction.
func (ops Float32Ops) Sub(a, b float32) float32 { return a - b }

// Mul performs element-wise multiplication.
func (ops Float32Ops) Mul(a, b float32) float32 { return a * b }

// Div performs element-wise division.
func (ops Float32Ops) Div(a, b float32) float32 {
	if b == 0 {
		return 0 // Avoid NaN
	}

	return a / b
}

// Tanh computes the hyperbolic tangent of x.
func (ops Float32Ops) Tanh(x float32) float32 {
	return float32(math.Tanh(float64(x)))
}

// Sigmoid computes the sigmoid function of x.
func (ops Float32Ops) Sigmoid(x float32) float32 {
	return 1.0 / (1.0 + float32(math.Exp(float64(-x))))
}

// TanhGrad computes the gradient of the hyperbolic tangent function.
func (ops Float32Ops) TanhGrad(x float32) float32 {
	tanhX := ops.Tanh(x)

	return 1.0 - (tanhX * tanhX)
}

// SigmoidGrad computes the gradient of the sigmoid function.
func (ops Float32Ops) SigmoidGrad(x float32) float32 {
	sigX := ops.Sigmoid(x)

	return sigX * (1.0 - sigX)
}

// ReLU computes the Rectified Linear Unit function.
func (ops Float32Ops) ReLU(x float32) float32 {
	if x > 0 {
		return x
	}

	return 0
}

// LeakyReLU computes the Leaky Rectified Linear Unit function.
func (ops Float32Ops) LeakyReLU(x float32, alpha float64) float32 {
	if x > 0 {
		return x
	}

	return float32(float64(x) * alpha)
}

// ReLUGrad computes the gradient of the Rectified Linear Unit function.
func (ops Float32Ops) ReLUGrad(x float32) float32 {
	if x > 0 {
		return 1
	}

	return 0
}

// LeakyReLUGrad computes the gradient of the Leaky Rectified Linear Unit function.
func (ops Float32Ops) LeakyReLUGrad(x float32, alpha float64) float32 {
	if x > 0 {
		return 1
	}

	return float32(alpha)
}

// FromFloat32 converts a float32 to a float32.
func (ops Float32Ops) FromFloat32(f float32) float32 {
	return f
}

// FromFloat64 converts a float64 to a float32.
func (ops Float32Ops) FromFloat64(f float64) float32 {
	return float32(f)
}

// ToFloat32 converts a float32 to a float32.
func (ops Float32Ops) ToFloat32(t float32) float32 {
	return t
}

// IsZero checks if the given float32 value is zero.
func (ops Float32Ops) IsZero(v float32) bool {
	return v == 0
}

// Exp computes the exponential of x.
func (ops Float32Ops) Exp(x float32) float32 {
	return float32(math.Exp(float64(x)))
}

// Log computes the natural logarithm of x.
func (ops Float32Ops) Log(x float32) float32 {
	return float32(math.Log(float64(x)))
}

// Pow computes base raised to the power of exponent.
func (ops Float32Ops) Pow(base, exponent float32) float32 {
	return float32(math.Pow(float64(base), float64(exponent)))
}

// Sqrt computes the square root of x.
func (ops Float32Ops) Sqrt(x float32) float32 {
	return float32(math.Sqrt(float64(x)))
}

// Abs computes the absolute value of x.
func (ops Float32Ops) Abs(x float32) float32 {
	if x < 0 {
		return -x
	}

	return x
}

// Sum computes the sum of elements in a slice.
func (ops Float32Ops) Sum(s []float32) float32 {
	var sum float32
	for _, v := range s {
		sum += v
	}

	return sum
}

// GreaterThan checks if a is greater than b.
func (ops Float32Ops) GreaterThan(a, b float32) bool {
	return a > b
}

// One returns a float32 with value 1.
func (ops Float32Ops) One() float32 {
	return 1.0
}

// Float64Ops provides the implementation of the Arithmetic interface for the float64 type.
type Float64Ops struct{}

// Add performs element-wise addition.
func (ops Float64Ops) Add(a, b float64) float64 { return a + b }

// Sub performs element-wise subtraction.
func (ops Float64Ops) Sub(a, b float64) float64 { return a - b }

// Mul performs element-wise multiplication.
func (ops Float64Ops) Mul(a, b float64) float64 { return a * b }

// Div performs element-wise division.
func (ops Float64Ops) Div(a, b float64) float64 {
	if b == 0 {
		return 0 // Avoid NaN
	}

	return a / b
}

// Tanh computes the hyperbolic tangent of x.
func (ops Float64Ops) Tanh(x float64) float64 {
	return math.Tanh(x)
}

// Sigmoid computes the sigmoid function of x.
func (ops Float64Ops) Sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

// TanhGrad computes the gradient of the hyperbolic tangent function.
func (ops Float64Ops) TanhGrad(x float64) float64 {
	tanhX := ops.Tanh(x)

	return 1.0 - (tanhX * tanhX)
}

// SigmoidGrad computes the gradient of the sigmoid function.
func (ops Float64Ops) SigmoidGrad(x float64) float64 {
	sigX := ops.Sigmoid(x)

	return sigX * (1.0 - sigX)
}

// ReLU computes the Rectified Linear Unit function.
func (ops Float64Ops) ReLU(x float64) float64 {
	if x > 0 {
		return x
	}

	return 0
}

// LeakyReLU computes the Leaky Rectified Linear Unit function.
func (ops Float64Ops) LeakyReLU(x float64, alpha float64) float64 {
	if x > 0 {
		return x
	}

	return x * alpha
}

// ReLUGrad computes the gradient of the Rectified Linear Unit function.
func (ops Float64Ops) ReLUGrad(x float64) float64 {
	if x > 0 {
		return 1
	}

	return 0
}

// LeakyReLUGrad computes the gradient of the Leaky Rectified Linear Unit function.
func (ops Float64Ops) LeakyReLUGrad(x float64, alpha float64) float64 {
	if x > 0 {
		return 1
	}

	return alpha
}

// FromFloat32 converts a float32 to a float64.
func (ops Float64Ops) FromFloat32(f float32) float64 {
	return float64(f)
}

// FromFloat64 converts a float64 to a float64.
func (ops Float64Ops) FromFloat64(f float64) float64 {
	return f
}

// ToFloat32 converts a float64 to a float32.
func (ops Float64Ops) ToFloat32(t float64) float32 {
	return float32(t)
}

// IsZero checks if the given float64 value is zero.
func (ops Float64Ops) IsZero(v float64) bool {
	return v == 0
}

// Exp computes the exponential of x.
func (ops Float64Ops) Exp(x float64) float64 {
	return math.Exp(x)
}

// Log computes the natural logarithm of x.
func (ops Float64Ops) Log(x float64) float64 {
	return math.Log(x)
}

// Pow computes base raised to the power of exponent.
func (ops Float64Ops) Pow(base, exponent float64) float64 {
	return math.Pow(base, exponent)
}

// Sqrt computes the square root of x.
func (ops Float64Ops) Sqrt(x float64) float64 {
	return math.Sqrt(x)
}

// Abs computes the absolute value of x.
func (ops Float64Ops) Abs(x float64) float64 {
	if x < 0 {
		return -x
	}

	return x
}

// Sum computes the sum of elements in a slice.
func (ops Float64Ops) Sum(s []float64) float64 {
	var sum float64
	for _, v := range s {
		sum += v
	}

	return sum
}

// GreaterThan checks if a is greater than b.
func (ops Float64Ops) GreaterThan(a, b float64) bool {
	return a > b
}

// One returns a float64 with value 1.
func (ops Float64Ops) One() float64 {
	return 1.0
}

// IntOps implements Arithmetic for int.
type IntOps struct{}

// Add performs element-wise addition.
func (IntOps) Add(a, b int) int { return a + b }

// Sub performs element-wise subtraction.
func (IntOps) Sub(a, b int) int { return a - b }

// Mul performs element-wise multiplication.
func (IntOps) Mul(a, b int) int { return a * b }

// Div performs element-wise division.
func (IntOps) Div(a, b int) int {
	if b == 0 {
		return 0 // Avoid panic
	}

	return a / b
}

// FromFloat32 converts a float32 to an int.
func (IntOps) FromFloat32(f float32) int { return int(f) }

// FromFloat64 converts a float64 to an int.
func (IntOps) FromFloat64(f float64) int { return int(f) }

// ToFloat32 converts an int to a float32.
func (IntOps) ToFloat32(t int) float32 { return float32(t) }

// Tanh computes the hyperbolic tangent of x.
func (IntOps) Tanh(x int) int { return int(math.Tanh(float64(x))) }

// Sigmoid computes the sigmoid function of x.
func (IntOps) Sigmoid(x int) int { return int(1.0 / (1.0 + math.Exp(float64(-x)))) }

// ReLU computes the Rectified Linear Unit function.
func (IntOps) ReLU(x int) int {
	if x > 0 {
		return x
	}

	return 0
}

// LeakyReLU computes the Leaky Rectified Linear Unit function.
func (IntOps) LeakyReLU(x int, alpha float64) int {
	if x > 0 {
		return x
	}

	return int(float64(x) * alpha)
}

// TanhGrad computes the gradient of the hyperbolic tangent function.
func (IntOps) TanhGrad(x int) int {
	tanhX := int(math.Tanh(float64(x)))

	return 1 - (tanhX * tanhX)
}

// SigmoidGrad computes the gradient of the sigmoid function.
func (IntOps) SigmoidGrad(x int) int {
	sigX := int(1.0 / (1.0 + math.Exp(float64(-x))))

	return sigX * (1 - sigX)
}

// ReLUGrad computes the gradient of the Rectified Linear Unit function.
func (IntOps) ReLUGrad(x int) int {
	if x > 0 {
		return 1
	}

	return 0
}

// LeakyReLUGrad computes the gradient of the Leaky Rectified Linear Unit function.
func (IntOps) LeakyReLUGrad(x int, alpha float64) int {
	if x > 0 {
		return 1
	}

	return int(alpha)
}

// IsZero checks if the given int value is zero.
func (IntOps) IsZero(v int) bool { return v == 0 }

// Exp computes the exponential of x.
func (IntOps) Exp(x int) int { return int(math.Exp(float64(x))) }

// Log computes the natural logarithm of x.
func (IntOps) Log(x int) int { return int(math.Log(float64(x))) }

// Pow computes base raised to the power of exponent.
func (IntOps) Pow(base, exponent int) int {
	return int(math.Pow(float64(base), float64(exponent)))
}

// Sqrt computes the square root of x.
func (IntOps) Sqrt(x int) int {
	return int(math.Sqrt(float64(x)))
}

// Abs computes the absolute value of x.
func (IntOps) Abs(x int) int {
	if x < 0 {
		return -x
	}

	return x
}

// Sum computes the sum of elements in a slice.
func (IntOps) Sum(s []int) int {
	var sum int
	for _, v := range s {
		sum += v
	}

	return sum
}

// GreaterThan checks if a is greater than b.
func (IntOps) GreaterThan(a, b int) bool {
	return a > b
}

// One returns an int with value 1.
func (IntOps) One() int {
	return 1
}
