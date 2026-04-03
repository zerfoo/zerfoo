package normalization

import (
	"context"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/tensor"
)

// rmsNormalizeResult holds the outputs of rmsNormalize so callers can cache
// whichever fields they need for their backward pass.
type rmsNormalizeResult[T tensor.Numeric] struct {
	output     *tensor.TensorNumeric[T]
	rsqrt      *tensor.TensorNumeric[T] // 1/sqrt(mean(x^2) + eps), needed by both backward passes
	normalized *tensor.TensorNumeric[T] // input * rsqrt (before gain), needed by SLN backward
}

// rmsNormalize computes root-mean-square normalization:
//
//	rsqrt      = 1 / sqrt(mean(x^2, axis=-1, keepdims=true) + epsilon)
//	normalized = x * rsqrt
//	output     = normalized * gain
//
// It first attempts fused GPU/CPU kernels for float32 and falls back to a
// multi-step path for other types or when the fused path is unavailable.
func rmsNormalize[T tensor.Numeric](
	ctx context.Context,
	engine compute.Engine[T],
	input *tensor.TensorNumeric[T],
	gain *tensor.TensorNumeric[T],
	epsilon T,
) (rmsNormalizeResult[T], error) {
	var zero rmsNormalizeResult[T]

	// GPU fused single-pass kernel for float32.
	if fused, ok := engine.(compute.FusedRMSNormer); ok {
		if f32Input, iof := any(input).(*tensor.TensorNumeric[float32]); iof {
			f32Gain, gOk := any(gain).(*tensor.TensorNumeric[float32])
			f32Eps, eOk := any(epsilon).(float32)
			if gOk && eOk && f32Gain.Size() == f32Input.Shape()[len(f32Input.Shape())-1] {
				out, scales, err := fused.FusedRMSNormGPU(f32Input, f32Gain, f32Eps)
				if err != nil {
					return zero, err
				}
				var rsqrt *tensor.TensorNumeric[T]
				if scales != nil {
					rsqrt = any(scales).(*tensor.TensorNumeric[T])
				}
				var normalized *tensor.TensorNumeric[T]
				if rsqrt != nil {
					normalized, err = engine.Mul(ctx, input, rsqrt)
					if err != nil {
						return zero, err
					}
				}
				return rmsNormalizeResult[T]{
					output:     any(out).(*tensor.TensorNumeric[T]),
					rsqrt:      rsqrt,
					normalized: normalized,
				}, nil
			}
		}
	}

	// CPU fused single-pass kernel for float32.
	if _, isCPU := engine.(*compute.CPUEngine[T]); isCPU {
		if f32Input, ok := any(input).(*tensor.TensorNumeric[float32]); ok {
			f32Gain, gOk := any(gain).(*tensor.TensorNumeric[float32])
			f32Eps, eOk := any(epsilon).(float32)
			if gOk && eOk && f32Gain.Size() == f32Input.Shape()[len(f32Input.Shape())-1] {
				out, scales, err := compute.FusedRMSNorm(f32Input, f32Gain, f32Eps)
				if err != nil {
					return zero, err
				}
				rsqrt := any(scales).(*tensor.TensorNumeric[T])
				normalized, err := engine.Mul(ctx, input, rsqrt)
				if err != nil {
					return zero, err
				}
				return rmsNormalizeResult[T]{
					output:     any(out).(*tensor.TensorNumeric[T]),
					rsqrt:      rsqrt,
					normalized: normalized,
				}, nil
			}
		}
	}

	// Fallback: multi-step path for non-float32 types or when fused is unavailable.
	squared, err := engine.Mul(ctx, input, input)
	if err != nil {
		return zero, err
	}

	lastDim := len(input.Shape()) - 1

	meanSq, err := engine.ReduceMean(ctx, squared, lastDim, true)
	if err != nil {
		return zero, err
	}

	meanSqPlusEps, err := engine.AddScalar(ctx, meanSq, epsilon)
	if err != nil {
		return zero, err
	}

	rsqrt, err := engine.Rsqrt(ctx, meanSqPlusEps)
	if err != nil {
		return zero, err
	}

	normalized, err := engine.Mul(ctx, input, rsqrt)
	if err != nil {
		return zero, err
	}

	output, err := engine.Mul(ctx, normalized, gain)
	if err != nil {
		return zero, err
	}

	return rmsNormalizeResult[T]{
		output:     output,
		rsqrt:      rsqrt,
		normalized: normalized,
	}, nil
}
