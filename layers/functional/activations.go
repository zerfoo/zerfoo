package functional

import (
	"context"
	"math"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
)

// GELU applies the Gaussian Error Linear Unit activation using the tanh
// approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715*x^3))).
// All arithmetic is routed through Engine[T] primitives.
func GELU[T tensor.Float](ctx context.Context, engine compute.Engine[T], ops numeric.Arithmetic[T],
	x *tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {

	// x^2
	x2, err := engine.Mul(ctx, x, x)
	if err != nil {
		return nil, err
	}
	// x^3
	x3, err := engine.Mul(ctx, x2, x)
	if err != nil {
		return nil, err
	}
	// 0.044715 * x^3
	coeff, err := engine.MulScalar(ctx, x3, ops.FromFloat64(0.044715))
	if err != nil {
		return nil, err
	}
	// x + 0.044715 * x^3
	inner, err := engine.Add(ctx, x, coeff)
	if err != nil {
		return nil, err
	}
	// sqrt(2/pi) * (x + 0.044715 * x^3)
	scaled, err := engine.MulScalar(ctx, inner, ops.FromFloat64(math.Sqrt(2/math.Pi)))
	if err != nil {
		return nil, err
	}
	// tanh(...)
	th, err := engine.Tanh(ctx, scaled)
	if err != nil {
		return nil, err
	}
	// 1 + tanh(...)
	onePlus, err := engine.AddScalar(ctx, th, ops.One())
	if err != nil {
		return nil, err
	}
	// x * (1 + tanh(...))
	prod, err := engine.Mul(ctx, x, onePlus)
	if err != nil {
		return nil, err
	}
	// 0.5 * x * (1 + tanh(...))
	return engine.MulScalar(ctx, prod, ops.FromFloat64(0.5))
}

// Softmax applies the softmax function along the given axis.
func Softmax[T tensor.Numeric](ctx context.Context, engine compute.Engine[T],
	x *tensor.TensorNumeric[T], axis int) (*tensor.TensorNumeric[T], error) {
	return engine.Softmax(ctx, x, axis)
}

// ReLU applies the Rectified Linear Unit activation: max(0, x).
func ReLU[T tensor.Numeric](ctx context.Context, engine compute.Engine[T], ops numeric.Arithmetic[T],
	x *tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	return engine.UnaryOp(ctx, x, ops.ReLU)
}

// Sigmoid applies the sigmoid activation: exp(x) / (1 + exp(x)).
// All arithmetic is routed through Engine[T] primitives.
func Sigmoid[T tensor.Numeric](ctx context.Context, engine compute.Engine[T], ops numeric.Arithmetic[T],
	x *tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {

	expX, err := engine.Exp(ctx, x)
	if err != nil {
		return nil, err
	}
	denom, err := engine.AddScalar(ctx, expX, ops.One())
	if err != nil {
		return nil, err
	}
	return engine.Div(ctx, expX, denom)
}

// SiLU applies the Sigmoid Linear Unit (SiLU / Swish) activation: x * sigmoid(x).
// All arithmetic is routed through Engine[T] primitives.
func SiLU[T tensor.Float](ctx context.Context, engine compute.Engine[T], ops numeric.Arithmetic[T],
	x *tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {

	sig, err := Sigmoid(ctx, engine, ops, x)
	if err != nil {
		return nil, err
	}
	return engine.Mul(ctx, x, sig)
}
