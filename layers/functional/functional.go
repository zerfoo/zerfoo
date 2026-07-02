package functional

import (
	"context"
	"fmt"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/tensor"
)

// LayerNorm applies layer normalization to x using the provided scale (gamma)
// and bias (beta) tensors.  Normalization is performed over the last dimension.
//
// output = (x - mean) / sqrt(variance + eps) * scale + bias
func LayerNorm[T tensor.Numeric](ctx context.Context, engine compute.Engine[T],
	x, scale, bias *tensor.TensorNumeric[T], eps T) (*tensor.TensorNumeric[T], error) {

	if x == nil {
		return nil, fmt.Errorf("functional.LayerNorm: input tensor x is nil")
	}
	if scale == nil {
		return nil, fmt.Errorf("functional.LayerNorm: scale tensor is nil")
	}
	if bias == nil {
		return nil, fmt.Errorf("functional.LayerNorm: bias tensor is nil")
	}

	lastDim := len(x.Shape()) - 1

	// mean = ReduceSum(x, lastDim) / featureSize
	sum, err := engine.ReduceSum(ctx, x, lastDim, true, nil)
	if err != nil {
		return nil, fmt.Errorf("functional.LayerNorm: reduce sum for mean: %w", err)
	}

	featureSize := engine.Ops().FromFloat64(float64(x.Shape()[lastDim]))
	mean, err := engine.DivScalar(ctx, sum, featureSize, nil)
	if err != nil {
		return nil, fmt.Errorf("functional.LayerNorm: div scalar for mean: %w", err)
	}

	// variance = mean((x - mean)^2)
	diff, err := engine.Sub(ctx, x, mean, nil)
	if err != nil {
		return nil, fmt.Errorf("functional.LayerNorm: sub mean: %w", err)
	}

	sq, err := engine.Mul(ctx, diff, diff, nil)
	if err != nil {
		return nil, fmt.Errorf("functional.LayerNorm: square diff: %w", err)
	}

	sqSum, err := engine.ReduceSum(ctx, sq, lastDim, true, nil)
	if err != nil {
		return nil, fmt.Errorf("functional.LayerNorm: reduce sum for variance: %w", err)
	}

	variance, err := engine.DivScalar(ctx, sqSum, featureSize, nil)
	if err != nil {
		return nil, fmt.Errorf("functional.LayerNorm: div scalar for variance: %w", err)
	}

	// stddev = sqrt(variance + eps)
	varEps, err := engine.AddScalar(ctx, variance, eps, nil)
	if err != nil {
		return nil, fmt.Errorf("functional.LayerNorm: add epsilon: %w", err)
	}

	stddev, err := engine.Sqrt(ctx, varEps, nil)
	if err != nil {
		return nil, fmt.Errorf("functional.LayerNorm: sqrt: %w", err)
	}

	// normed = (x - mean) / stddev
	normed, err := engine.Div(ctx, diff, stddev, nil)
	if err != nil {
		return nil, fmt.Errorf("functional.LayerNorm: div by stddev: %w", err)
	}

	// output = normed * scale + bias
	scaled, err := engine.Mul(ctx, normed, scale, nil)
	if err != nil {
		return nil, fmt.Errorf("functional.LayerNorm: mul scale: %w", err)
	}

	out, err := engine.Add(ctx, scaled, bias, nil)
	if err != nil {
		return nil, fmt.Errorf("functional.LayerNorm: add bias: %w", err)
	}

	return out, nil
}

// RMSNorm applies root-mean-square normalization to x using the provided scale
// (gain) tensor.  Normalization is performed over the last dimension.
//
// output = x * rsqrt(mean(x^2) + eps) * scale
func RMSNorm[T tensor.Numeric](ctx context.Context, engine compute.Engine[T],
	x, scale *tensor.TensorNumeric[T], eps T) (*tensor.TensorNumeric[T], error) {

	if x == nil {
		return nil, fmt.Errorf("functional.RMSNorm: input tensor x is nil")
	}
	if scale == nil {
		return nil, fmt.Errorf("functional.RMSNorm: scale tensor is nil")
	}

	lastDim := len(x.Shape()) - 1

	// mean(x^2) over last dim
	sq, err := engine.Mul(ctx, x, x, nil)
	if err != nil {
		return nil, fmt.Errorf("functional.RMSNorm: square input: %w", err)
	}

	meanSq, err := engine.ReduceMean(ctx, sq, lastDim, true)
	if err != nil {
		return nil, fmt.Errorf("functional.RMSNorm: reduce mean: %w", err)
	}

	// rsqrt(mean(x^2) + eps)
	meanSqEps, err := engine.AddScalar(ctx, meanSq, eps, nil)
	if err != nil {
		return nil, fmt.Errorf("functional.RMSNorm: add epsilon: %w", err)
	}

	rsqrt, err := engine.Rsqrt(ctx, meanSqEps, nil)
	if err != nil {
		return nil, fmt.Errorf("functional.RMSNorm: rsqrt: %w", err)
	}

	// x * rsqrt * scale
	normed, err := engine.Mul(ctx, x, rsqrt, nil)
	if err != nil {
		return nil, fmt.Errorf("functional.RMSNorm: mul rsqrt: %w", err)
	}

	out, err := engine.Mul(ctx, normed, scale, nil)
	if err != nil {
		return nil, fmt.Errorf("functional.RMSNorm: mul scale: %w", err)
	}

	return out, nil
}
