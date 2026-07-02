package functional

import (
	"context"

	"github.com/zerfoo/zerfoo/layers/activations"
	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
)

// GELU applies the Gaussian Error Linear Unit activation using the tanh
// approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715*x^3))).
// Thin wrapper that delegates to the canonical activations.Gelu Node so
// there is a single source of truth for the math (T124.2.2).
func GELU[T tensor.Float](ctx context.Context, engine compute.Engine[T], ops numeric.Arithmetic[T],
	x *tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	return activations.NewGelu(engine, ops).Forward(ctx, x)
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
// Thin wrapper that delegates to the canonical activations.Sigmoid Node
// (T124.2.2).
func Sigmoid[T tensor.Numeric](ctx context.Context, engine compute.Engine[T], ops numeric.Arithmetic[T],
	x *tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	return activations.NewSigmoid(engine, ops).Forward(ctx, x)
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
