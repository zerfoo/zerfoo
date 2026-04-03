package functional

import (
	"context"
	"fmt"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/tensor"
)

// LinearBackward computes gradients for y = x @ weight^T + bias.
// dOutput: gradient from upstream [batch, out_features]
// input: original input [batch, in_features]
// weight: weight matrix [out_features, in_features]
// Returns: dInput [batch, in_features], dWeight [out_features, in_features], dBias [out_features]
func LinearBackward[T tensor.Numeric](ctx context.Context, engine compute.Engine[T],
	dOutput, input, weight *tensor.TensorNumeric[T]) (dInput, dWeight, dBias *tensor.TensorNumeric[T], err error) {

	if dOutput == nil {
		return nil, nil, nil, fmt.Errorf("functional.LinearBackward: dOutput is nil")
	}
	if input == nil {
		return nil, nil, nil, fmt.Errorf("functional.LinearBackward: input is nil")
	}
	if weight == nil {
		return nil, nil, nil, fmt.Errorf("functional.LinearBackward: weight is nil")
	}

	// dInput = dOutput @ weight
	// dOutput: [batch, out_features], weight: [out_features, in_features]
	// result: [batch, in_features]
	dInput, err = engine.MatMul(ctx, dOutput, weight)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("functional.LinearBackward: dInput matmul: %w", err)
	}

	// dWeight = dOutput^T @ input
	// dOutput^T: [out_features, batch], input: [batch, in_features]
	// result: [out_features, in_features]
	dOutputT, err := engine.Transpose(ctx, dOutput, []int{1, 0})
	if err != nil {
		return nil, nil, nil, fmt.Errorf("functional.LinearBackward: transpose dOutput: %w", err)
	}

	dWeight, err = engine.MatMul(ctx, dOutputT, input)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("functional.LinearBackward: dWeight matmul: %w", err)
	}

	// dBias = sum(dOutput, axis=0)
	// dOutput: [batch, out_features] → [out_features]
	dBias, err = engine.ReduceSum(ctx, dOutput, 0, false)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("functional.LinearBackward: dBias reduce sum: %w", err)
	}

	return dInput, dWeight, dBias, nil
}
