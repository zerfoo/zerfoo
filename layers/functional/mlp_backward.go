package functional

import (
	"context"
	"fmt"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
)

// MLPBackward computes gradients for a 2-layer MLP: y = Linear2(activation(Linear1(x))).
// dOutput: gradient from upstream [batch, out_features]
// input: original input [batch, in_features]
// weight1: [hidden, in_features], bias1: [hidden]
// weight2: [out_features, hidden], bias2: [out_features]
// hidden: output of Linear1 (pre-activation) [batch, hidden]
// activated: output after activation [batch, hidden]
// activation: "relu" or "gelu"
// Returns: dInput, dWeight1, dBias1, dWeight2, dBias2
func MLPBackward[T tensor.Float](ctx context.Context, engine compute.Engine[T], ops numeric.Arithmetic[T],
	dOutput, input, weight1, bias1, weight2, bias2, hidden, activated *tensor.TensorNumeric[T],
	activation string) (dInput, dWeight1, dBias1, dWeight2, dBias2 *tensor.TensorNumeric[T], err error) {

	if dOutput == nil {
		return nil, nil, nil, nil, nil, fmt.Errorf("functional.MLPBackward: dOutput is nil")
	}
	if input == nil {
		return nil, nil, nil, nil, nil, fmt.Errorf("functional.MLPBackward: input is nil")
	}
	if weight1 == nil {
		return nil, nil, nil, nil, nil, fmt.Errorf("functional.MLPBackward: weight1 is nil")
	}
	if weight2 == nil {
		return nil, nil, nil, nil, nil, fmt.Errorf("functional.MLPBackward: weight2 is nil")
	}
	if hidden == nil {
		return nil, nil, nil, nil, nil, fmt.Errorf("functional.MLPBackward: hidden is nil")
	}
	if activated == nil {
		return nil, nil, nil, nil, nil, fmt.Errorf("functional.MLPBackward: activated is nil")
	}

	// Step 1: Backward through Linear2.
	// Linear2 forward: y = activated @ weight2^T + bias2
	// LinearBackward gives us dLinear2Input (grad w.r.t. activated), dWeight2, dBias2.
	dLinear2Input, dWeight2, dBias2, err := LinearBackward(ctx, engine, dOutput, activated, weight2)
	if err != nil {
		return nil, nil, nil, nil, nil, fmt.Errorf("functional.MLPBackward: linear2 backward: %w", err)
	}

	// Step 2: Backward through activation.
	var dHidden *tensor.TensorNumeric[T]
	switch activation {
	case "relu":
		// ReLU'(x) = 1 if x > 0, else 0.
		// dHidden = dLinear2Input * step(hidden)
		zero := ops.FromFloat64(0)
		one := ops.One()
		reluStep := func(x T) T {
			if x > zero {
				return one
			}
			return zero
		}
		mask, err2 := engine.UnaryOp(ctx, hidden, reluStep)
		if err2 != nil {
			return nil, nil, nil, nil, nil, fmt.Errorf("functional.MLPBackward: relu mask: %w", err2)
		}
		dHidden, err = engine.Mul(ctx, dLinear2Input, mask)
		if err != nil {
			return nil, nil, nil, nil, nil, fmt.Errorf("functional.MLPBackward: relu backward mul: %w", err)
		}
	case "gelu":
		dHidden, err = GELUBackward(ctx, engine, ops, dLinear2Input, hidden)
		if err != nil {
			return nil, nil, nil, nil, nil, fmt.Errorf("functional.MLPBackward: gelu backward: %w", err)
		}
	default:
		return nil, nil, nil, nil, nil, fmt.Errorf("functional.MLPBackward: unsupported activation %q (want \"relu\" or \"gelu\")", activation)
	}

	// Step 3: Backward through Linear1.
	// Linear1 forward: hidden = input @ weight1^T + bias1
	dInput, dWeight1, dBias1, err = LinearBackward(ctx, engine, dHidden, input, weight1)
	if err != nil {
		return nil, nil, nil, nil, nil, fmt.Errorf("functional.MLPBackward: linear1 backward: %w", err)
	}

	return dInput, dWeight1, dBias1, dWeight2, dBias2, nil
}
