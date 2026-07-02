package functional

import (
	"context"
	"fmt"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/tensor"
)

// LayerNormBackward computes gradients for layer normalization.
// dOutput: gradient from upstream [*, features]
// input: original input [*, features]
// scale: gamma [features]
// eps: epsilon used in forward
// Returns: dInput [*, features], dScale [features], dBias [features]
func LayerNormBackward[T tensor.Numeric](ctx context.Context, engine compute.Engine[T],
	dOutput, input, scale *tensor.TensorNumeric[T], eps T) (dInput, dScale, dBias *tensor.TensorNumeric[T], err error) {

	if dOutput == nil {
		return nil, nil, nil, fmt.Errorf("functional.LayerNormBackward: dOutput is nil")
	}
	if input == nil {
		return nil, nil, nil, fmt.Errorf("functional.LayerNormBackward: input is nil")
	}
	if scale == nil {
		return nil, nil, nil, fmt.Errorf("functional.LayerNormBackward: scale is nil")
	}

	lastDim := len(input.Shape()) - 1
	featureSize := engine.Ops().FromFloat64(float64(input.Shape()[lastDim]))

	// Recompute mean from input
	sum, err := engine.ReduceSum(ctx, input, lastDim, true, nil)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("functional.LayerNormBackward: reduce sum for mean: %w", err)
	}
	mean, err := engine.DivScalar(ctx, sum, featureSize, nil)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("functional.LayerNormBackward: div scalar for mean: %w", err)
	}

	// Recompute variance
	diff, err := engine.Sub(ctx, input, mean, nil)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("functional.LayerNormBackward: sub mean: %w", err)
	}
	sq, err := engine.Mul(ctx, diff, diff, nil)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("functional.LayerNormBackward: square diff: %w", err)
	}
	sqSum, err := engine.ReduceSum(ctx, sq, lastDim, true, nil)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("functional.LayerNormBackward: reduce sum for variance: %w", err)
	}
	variance, err := engine.DivScalar(ctx, sqSum, featureSize, nil)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("functional.LayerNormBackward: div scalar for variance: %w", err)
	}

	// std = sqrt(variance + eps)
	varEps, err := engine.AddScalar(ctx, variance, eps, nil)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("functional.LayerNormBackward: add epsilon: %w", err)
	}
	std, err := engine.Sqrt(ctx, varEps, nil)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("functional.LayerNormBackward: sqrt: %w", err)
	}

	// xhat = (input - mean) / std
	xhat, err := engine.Div(ctx, diff, std, nil)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("functional.LayerNormBackward: xhat: %w", err)
	}

	// dBias = sum(dOutput, axis=0...-1) — reduce all batch dims
	dBias = dOutput
	for d := 0; d < lastDim; d++ {
		dBias, err = engine.ReduceSum(ctx, dBias, 0, false, nil)
		if err != nil {
			return nil, nil, nil, fmt.Errorf("functional.LayerNormBackward: dBias reduce dim %d: %w", d, err)
		}
	}

	// dScale = sum(dOutput * xhat, axis=0...-1)
	dOutXhat, err := engine.Mul(ctx, dOutput, xhat, nil)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("functional.LayerNormBackward: dOutput * xhat: %w", err)
	}
	dScale = dOutXhat
	for d := 0; d < lastDim; d++ {
		dScale, err = engine.ReduceSum(ctx, dScale, 0, false, nil)
		if err != nil {
			return nil, nil, nil, fmt.Errorf("functional.LayerNormBackward: dScale reduce dim %d: %w", d, err)
		}
	}

	// dxhat = dOutput * scale
	dxhat, err := engine.Mul(ctx, dOutput, scale, nil)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("functional.LayerNormBackward: dxhat: %w", err)
	}

	// mean(dxhat) over features
	meanDxhat, err := engine.ReduceMean(ctx, dxhat, lastDim, true)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("functional.LayerNormBackward: mean dxhat: %w", err)
	}

	// mean(dxhat * xhat) over features
	dxhatXhat, err := engine.Mul(ctx, dxhat, xhat, nil)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("functional.LayerNormBackward: dxhat * xhat: %w", err)
	}
	meanDxhatXhat, err := engine.ReduceMean(ctx, dxhatXhat, lastDim, true)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("functional.LayerNormBackward: mean dxhat*xhat: %w", err)
	}

	// dInput = (1/std) * (dxhat - mean(dxhat) - xhat * mean(dxhat * xhat))
	xhatTerm, err := engine.Mul(ctx, xhat, meanDxhatXhat, nil)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("functional.LayerNormBackward: xhat * mean(dxhat*xhat): %w", err)
	}

	inner, err := engine.Sub(ctx, dxhat, meanDxhat, nil)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("functional.LayerNormBackward: dxhat - mean(dxhat): %w", err)
	}

	inner, err = engine.Sub(ctx, inner, xhatTerm, nil)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("functional.LayerNormBackward: sub xhat term: %w", err)
	}

	dInput, err = engine.Div(ctx, inner, std, nil)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("functional.LayerNormBackward: div by std: %w", err)
	}

	return dInput, dScale, dBias, nil
}
