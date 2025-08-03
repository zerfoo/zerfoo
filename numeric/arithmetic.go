package numeric

// Arithmetic defines a generic interface for all mathematical operations
// required by the compute engine. This allows the engine to be completely
// agnostic to the specific numeric type it is operating on.
type Arithmetic[T any] interface {
	// Basic binary operations
	Add(a, b T) T
	Sub(a, b T) T
	Mul(a, b T) T
	Div(a, b T) T

	// Activation functions and their derivatives
	Tanh(x T) T
	Sigmoid(x T) T
	ReLU(x T) T
	LeakyReLU(x T, alpha float64) T
	TanhGrad(x T) T    // Derivative of Tanh
	SigmoidGrad(x T) T // Derivative of Sigmoid
	ReLUGrad(x T) T
	LeakyReLUGrad(x T, alpha float64) T

	// Conversion from standard types
	FromFloat32(f float32) T
	FromFloat64(f float64) T
	One() T
	

	// IsZero checks if a value is zero.
	IsZero(v T) bool

	// Abs returns the absolute value of x.
	Abs(x T) T
	// Sum returns the sum of all elements in the slice.
	Sum(s []T) T
	// Exp returns e**x.
	Exp(x T) T
	// Log returns the natural logarithm of x.
	Log(x T) T
	// Pow returns base**exponent.
	Pow(base, exponent T) T

	// Sqrt returns the square root of x.
	Sqrt(x T) T

	// GreaterThan returns true if a is greater than b.
	GreaterThan(a, b T) bool
}
