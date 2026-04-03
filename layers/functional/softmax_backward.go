package functional

import (
	"context"
	"fmt"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
)

// SoftmaxBackward computes the gradient of the softmax function.
// dOutput: gradient from upstream [*, features]
// softmaxOutput: output of softmax forward pass [*, features] (already computed s_i values)
// Returns: dInput [*, features]
//
// For each row: dInput_i = s_i * (dOutput_i - sum_j(dOutput_j * s_j))
func SoftmaxBackward[T tensor.Float](ctx context.Context, engine compute.Engine[T], ops numeric.Arithmetic[T],
	dOutput, softmaxOutput *tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {

	if dOutput == nil {
		return nil, fmt.Errorf("functional.SoftmaxBackward: dOutput tensor is nil")
	}
	if softmaxOutput == nil {
		return nil, fmt.Errorf("functional.SoftmaxBackward: softmaxOutput tensor is nil")
	}

	lastDim := len(dOutput.Shape()) - 1

	// dot = sum(dOutput * softmaxOutput, axis=-1, keepdim=true)
	prod, err := engine.Mul(ctx, dOutput, softmaxOutput)
	if err != nil {
		return nil, fmt.Errorf("functional.SoftmaxBackward: mul dOutput*softmaxOutput: %w", err)
	}

	dot, err := engine.ReduceSum(ctx, prod, lastDim, true)
	if err != nil {
		return nil, fmt.Errorf("functional.SoftmaxBackward: reduce sum: %w", err)
	}

	// dInput = softmaxOutput * (dOutput - dot)
	diff, err := engine.Sub(ctx, dOutput, dot)
	if err != nil {
		return nil, fmt.Errorf("functional.SoftmaxBackward: sub dot: %w", err)
	}

	dInput, err := engine.Mul(ctx, softmaxOutput, diff)
	if err != nil {
		return nil, fmt.Errorf("functional.SoftmaxBackward: mul softmaxOutput*(dOutput-dot): %w", err)
	}

	return dInput, nil
}
