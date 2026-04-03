package functional

import (
	"context"
	"math"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
)

// GELUBackward computes the gradient of the GELU activation.
// dOutput: gradient from upstream
// input: original input to GELU
// Returns: dInput (same shape as input)
//
// Using the tanh approximation GELU(x) = 0.5 * x * (1 + tanh(u))
// where u = sqrt(2/pi) * (x + 0.044715*x^3), the derivative is:
// GELU'(x) = 0.5*(1+tanh(u)) + 0.5*x*(1-tanh^2(u))*sqrt(2/pi)*(1+3*0.044715*x^2)
func GELUBackward[T tensor.Float](ctx context.Context, engine compute.Engine[T], ops numeric.Arithmetic[T],
	dOutput, input *tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {

	// x^2
	x2, err := engine.Mul(ctx, input, input)
	if err != nil {
		return nil, err
	}

	// x^3
	x3, err := engine.Mul(ctx, x2, input)
	if err != nil {
		return nil, err
	}

	// 0.044715 * x^3
	bx3, err := engine.MulScalar(ctx, x3, ops.FromFloat64(0.044715))
	if err != nil {
		return nil, err
	}

	// x + 0.044715 * x^3
	inner, err := engine.Add(ctx, input, bx3)
	if err != nil {
		return nil, err
	}

	// u = sqrt(2/pi) * (x + 0.044715 * x^3)
	u, err := engine.MulScalar(ctx, inner, ops.FromFloat64(math.Sqrt(2/math.Pi)))
	if err != nil {
		return nil, err
	}

	// tanh(u)
	tanhU, err := engine.Tanh(ctx, u)
	if err != nil {
		return nil, err
	}

	// sech^2(u) = 1 - tanh^2(u): negate tanh^2 then add 1
	tanhU2, err := engine.Mul(ctx, tanhU, tanhU)
	if err != nil {
		return nil, err
	}
	negTanhU2, err := engine.MulScalar(ctx, tanhU2, ops.FromFloat64(-1))
	if err != nil {
		return nil, err
	}
	sechSq, err := engine.AddScalar(ctx, negTanhU2, ops.One())
	if err != nil {
		return nil, err
	}

	// du/dx = sqrt(2/pi) * (1 + 3*0.044715*x^2)
	dterm, err := engine.MulScalar(ctx, x2, ops.FromFloat64(3*0.044715))
	if err != nil {
		return nil, err
	}
	dterm, err = engine.AddScalar(ctx, dterm, ops.One())
	if err != nil {
		return nil, err
	}
	dudx, err := engine.MulScalar(ctx, dterm, ops.FromFloat64(math.Sqrt(2/math.Pi)))
	if err != nil {
		return nil, err
	}

	// 0.5 * (1 + tanh(u))
	onePlusTanh, err := engine.AddScalar(ctx, tanhU, ops.One())
	if err != nil {
		return nil, err
	}
	halfOnePlusTanh, err := engine.MulScalar(ctx, onePlusTanh, ops.FromFloat64(0.5))
	if err != nil {
		return nil, err
	}

	// 0.5 * x * sech^2(u) * du/dx
	xSechSq, err := engine.Mul(ctx, input, sechSq)
	if err != nil {
		return nil, err
	}
	xSechSqDudx, err := engine.Mul(ctx, xSechSq, dudx)
	if err != nil {
		return nil, err
	}
	halfXSechSqDudx, err := engine.MulScalar(ctx, xSechSqDudx, ops.FromFloat64(0.5))
	if err != nil {
		return nil, err
	}

	// derivative = 0.5*(1+tanh(u)) + 0.5*x*sech^2(u)*du/dx
	derivative, err := engine.Add(ctx, halfOnePlusTanh, halfXSechSqDudx)
	if err != nil {
		return nil, err
	}

	// dInput = dOutput * derivative
	return engine.Mul(ctx, dOutput, derivative)
}
