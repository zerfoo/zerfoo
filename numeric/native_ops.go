package numeric

import "math"

// Float32Ops provides the implementation of the Arithmetic interface for the float32 type.
type Float32Ops struct{}

func (ops Float32Ops) Add(a, b float32) float32 { return a + b }
func (ops Float32Ops) Sub(a, b float32) float32 { return a - b }
func (ops Float32Ops) Mul(a, b float32) float32 { return a * b }
func (ops Float32Ops) Div(a, b float32) float32 {
	if b == 0 {
		return 0 // Avoid NaN
	}
	return a / b
}

func (ops Float32Ops) Tanh(x float32) float32 {
	return float32(math.Tanh(float64(x)))
}

func (ops Float32Ops) Sigmoid(x float32) float32 {
	return 1.0 / (1.0 + float32(math.Exp(float64(-x))))
}

func (ops Float32Ops) TanhGrad(x float32) float32 {
	tanhX := ops.Tanh(x)
	return 1.0 - (tanhX * tanhX)
}

func (ops Float32Ops) SigmoidGrad(x float32) float32 {
	sigX := ops.Sigmoid(x)
	return sigX * (1.0 - sigX)
}

func (ops Float32Ops) ReLU(x float32) float32 {
	if x > 0 {
		return x
	}
	return 0
}

func (ops Float32Ops) LeakyReLU(x float32, alpha float64) float32 {
	if x > 0 {
		return x
	}
	return float32(float64(x) * alpha)
}

func (ops Float32Ops) ReLUGrad(x float32) float32 {
	if x > 0 {
		return 1
	}
	return 0
}

func (ops Float32Ops) LeakyReLUGrad(x float32, alpha float64) float32 {
	if x > 0 {
		return 1
	}
	return float32(alpha)
}

func (ops Float32Ops) FromFloat32(f float32) float32 {
	return f
}

func (ops Float32Ops) FromFloat64(f float64) float32 {
	return float32(f)
}

func (ops Float32Ops) ToFloat32(t float32) float32 {
	return t
}

func (ops Float32Ops) IsZero(v float32) bool {
	return v == 0
}
func (ops Float32Ops) Exp(x float32) float32 {
	return float32(math.Exp(float64(x)))
}
func (ops Float32Ops) Log(x float32) float32 {
	return float32(math.Log(float64(x)))
}
func (ops Float32Ops) Pow(base, exponent float32) float32 {
	return float32(math.Pow(float64(base), float64(exponent)))
}
func (ops Float32Ops) Sqrt(x float32) float32 {
	return float32(math.Sqrt(float64(x)))
}
func (ops Float32Ops) Abs(x float32) float32 {
	if x < 0 {
		return -x
	}
	return x
}

func (ops Float32Ops) Sum(s []float32) float32 {
	var sum float32
	for _, v := range s {
		sum += v
	}
	return sum
}

func (ops Float32Ops) GreaterThan(a, b float32) bool {
	return a > b
}

func (ops Float32Ops) One() float32 {
	return 1.0
}



// Float64Ops provides the implementation of the Arithmetic interface for the float64 type.
type Float64Ops struct{}

func (ops Float64Ops) Add(a, b float64) float64 { return a + b }
func (ops Float64Ops) Sub(a, b float64) float64 { return a - b }
func (ops Float64Ops) Mul(a, b float64) float64 { return a * b }
func (ops Float64Ops) Div(a, b float64) float64 {
	if b == 0 {
		return 0 // Avoid NaN
	}
	return a / b
}

func (ops Float64Ops) Tanh(x float64) float64 {
	return math.Tanh(x)
}

func (ops Float64Ops) Sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

func (ops Float64Ops) TanhGrad(x float64) float64 {
	tanhX := ops.Tanh(x)
	return 1.0 - (tanhX * tanhX)
}

func (ops Float64Ops) SigmoidGrad(x float64) float64 {
	sigX := ops.Sigmoid(x)
	return sigX * (1.0 - sigX)
}

func (ops Float64Ops) ReLU(x float64) float64 {
	if x > 0 {
		return x
	}
	return 0
}

func (ops Float64Ops) LeakyReLU(x float64, alpha float64) float64 {
	if x > 0 {
		return x
	}
	return x * alpha
}

func (ops Float64Ops) ReLUGrad(x float64) float64 {
	if x > 0 {
		return 1
	}
	return 0
}

func (ops Float64Ops) LeakyReLUGrad(x float64, alpha float64) float64 {
	if x > 0 {
		return 1
	}
	return alpha
}

func (ops Float64Ops) FromFloat32(f float32) float64 {
	return float64(f)
}

func (ops Float64Ops) FromFloat64(f float64) float64 {
	return f
}

func (ops Float64Ops) ToFloat32(t float64) float32 {
	return float32(t)
}

func (ops Float64Ops) IsZero(v float64) bool {
	return v == 0
}
func (ops Float64Ops) Exp(x float64) float64 {
	return math.Exp(x)
}
func (ops Float64Ops) Log(x float64) float64 {
	return math.Log(x)
}
func (ops Float64Ops) Pow(base, exponent float64) float64 {
	return math.Pow(base, exponent)
}
func (ops Float64Ops) Sqrt(x float64) float64 {
	return math.Sqrt(x)
}
func (ops Float64Ops) Abs(x float64) float64 {
	if x < 0 {
		return -x
	}
	return x
}

func (ops Float64Ops) Sum(s []float64) float64 {
	var sum float64
	for _, v := range s {
		sum += v
	}
	return sum
}

func (ops Float64Ops) GreaterThan(a, b float64) bool {
	return a > b
}

func (ops Float64Ops) One() float64 {
	return 1.0
}

// IntOps implements Arithmetic for int.
type IntOps struct{}

func (IntOps) Add(a, b int) int { return a + b }
func (IntOps) Sub(a, b int) int { return a - b }
func (IntOps) Mul(a, b int) int { return a * b }
func (IntOps) Div(a, b int) int {
	if b == 0 {
		return 0 // Avoid panic
	}
	return a / b
}
func (IntOps) FromFloat32(f float32) int { return int(f) }
func (IntOps) FromFloat64(f float64) int { return int(f) }
func (IntOps) ToFloat32(t int) float32   { return float32(t) }
func (IntOps) Tanh(x int) int            { return int(math.Tanh(float64(x))) }
func (IntOps) Sigmoid(x int) int         { return int(1.0 / (1.0 + math.Exp(float64(-x)))) }
func (IntOps) ReLU(x int) int {
	if x > 0 {
		return x
	}
	return 0
}
func (IntOps) LeakyReLU(x int, alpha float64) int {
	if x > 0 {
		return x
	}
	return int(float64(x) * alpha)
}
func (IntOps) TanhGrad(x int) int {
	tanhX := int(math.Tanh(float64(x)))
	return 1 - (tanhX * tanhX)
}
func (IntOps) SigmoidGrad(x int) int {
	sigX := int(1.0 / (1.0 + math.Exp(float64(-x))))
	return sigX * (1 - sigX)
}
func (IntOps) ReLUGrad(x int) int {
	if x > 0 {
		return 1
	}
	return 0
}
func (IntOps) LeakyReLUGrad(x int, alpha float64) int {
	if x > 0 {
		return 1
	}
	return int(alpha)
}
func (IntOps) IsZero(v int) bool { return v == 0 }
func (IntOps) Exp(x int) int     { return int(math.Exp(float64(x))) }
func (IntOps) Log(x int) int     { return int(math.Log(float64(x))) }
func (IntOps) Pow(base, exponent int) int {
	return int(math.Pow(float64(base), float64(exponent)))
}
func (IntOps) Sqrt(x int) int {
	return int(math.Sqrt(float64(x)))
}
func (IntOps) Abs(x int) int {
	if x < 0 {
		return -x
	}
	return x
}

func (IntOps) Sum(s []int) int {
	var sum int
	for _, v := range s {
		sum += v
	}
	return sum
}

func (IntOps) GreaterThan(a, b int) bool {
	return a > b
}

func (IntOps) One() int {
	return 1
}