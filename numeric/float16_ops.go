package numeric

import (
	"math"

	"github.com/zerfoo/float16"
)

// Float16Ops provides the implementation of the Arithmetic interface for the float16.Float16 type.
type Float16Ops struct{}

func (ops Float16Ops) Add(a, b float16.Float16) float16.Float16 {
	res, _ := float16.AddWithMode(a, b, float16.ModeFastArithmetic, float16.RoundNearestEven)
	return res
}
func (ops Float16Ops) Sub(a, b float16.Float16) float16.Float16 {
	res, _ := float16.SubWithMode(a, b, float16.ModeFastArithmetic, float16.RoundNearestEven)
	return res
}
func (ops Float16Ops) Mul(a, b float16.Float16) float16.Float16 {
	res, _ := float16.MulWithMode(a, b, float16.ModeFastArithmetic, float16.RoundNearestEven)
	return res
}
func (ops Float16Ops) Div(a, b float16.Float16) float16.Float16 {
	res, _ := float16.DivWithMode(a, b, float16.ModeFastArithmetic, float16.RoundNearestEven)
	return res
}

func (ops Float16Ops) Tanh(x float16.Float16) float16.Float16 {
	return float16.Tanh(x)
}

func (ops Float16Ops) Sigmoid(x float16.Float16) float16.Float16 {
	// The float16 library does not have a Sigmoid function. We will simulate it.
	f32 := x.ToFloat32()
	return float16.ToFloat16(1.0 / (1.0 + float32(math.Exp(float64(-f32)))))
}

func (ops Float16Ops) TanhGrad(x float16.Float16) float16.Float16 {
	// TanhGrad is 1 - tanh(x)^2
	tanhX := ops.Tanh(x)
	tanhX2 := ops.Mul(tanhX, tanhX)
	one := float16.FromInt(1)
	return ops.Sub(one, tanhX2)
}

func (ops Float16Ops) SigmoidGrad(x float16.Float16) float16.Float16 {
	// SigmoidGrad is sigmoid(x) * (1 - sigmoid(x))
	sigX := ops.Sigmoid(x)
	one := float16.FromInt(1)
	oneMinusSigX := ops.Sub(one, sigX)
	return ops.Mul(sigX, oneMinusSigX)
}

func (ops Float16Ops) ReLU(x float16.Float16) float16.Float16 {
	if x.ToFloat32() > 0 {
		return x
	}
	return float16.FromInt(0)
}

func (ops Float16Ops) LeakyReLU(x float16.Float16, alpha float64) float16.Float16 {
	if x.ToFloat32() > 0 {
		return x
	}
	return ops.Mul(x, float16.ToFloat16(float32(alpha)))
}

func (ops Float16Ops) ReLUGrad(x float16.Float16) float16.Float16 {
	one := float16.FromInt(1)
	if x.ToFloat32() > 0 {
		return one
	}
	return float16.FromInt(0)
}

func (ops Float16Ops) LeakyReLUGrad(x float16.Float16, alpha float64) float16.Float16 {
	one := float16.FromInt(1)
	if x.ToFloat32() > 0 {
		return one
	}
	return float16.ToFloat16(float32(alpha))
}

func (ops Float16Ops) FromFloat32(f float32) float16.Float16 {
	return float16.ToFloat16(f)
}

func (ops Float16Ops) ToFloat32(t float16.Float16) float32 {
	return t.ToFloat32()
}

func (ops Float16Ops) IsZero(v float16.Float16) bool {
	return v.IsZero()
}

func (ops Float16Ops) Exp(x float16.Float16) float16.Float16 {
	return float16.Exp(x)
}

func (ops Float16Ops) Log(x float16.Float16) float16.Float16 {
	return float16.Log(x)
}

func (ops Float16Ops) Pow(base, exponent float16.Float16) float16.Float16 {
	return float16.Pow(base, exponent)
}

func (ops Float16Ops) Abs(x float16.Float16) float16.Float16 {
	return float16.Abs(x)
}

func (ops Float16Ops) Sqrt(x float16.Float16) float16.Float16 {
	return float16.Sqrt(x)
}

func (ops Float16Ops) Sum(s []float16.Float16) float16.Float16 {
	var sum float16.Float16
	for _, v := range s {
		sum, _ = float16.AddWithMode(sum, v, float16.ModeFastArithmetic, float16.RoundNearestEven)
	}
	return sum
}

func (ops Float16Ops) GreaterThan(a, b float16.Float16) bool {
	return a.ToFloat32() > b.ToFloat32()
}

func (ops Float16Ops) One() float16.Float16 {
	return float16.FromInt(1)
}

func (ops Float16Ops) FromFloat64(f float64) float16.Float16 {
	return float16.FromFloat64(f)
}
