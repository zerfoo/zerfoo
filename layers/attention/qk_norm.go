package attention

import (
	"context"
	"errors"
	"fmt"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/tensor"
)

// QKNorm applies a form of normalization to Query (Q) and Key (K) tensors
// to stabilize attention score scales, similar to RMSNorm.
// It normalizes Q and K independently by their respective RMS values.
// All operations use Engine primitives so they appear in the ExecutionPlan
// instruction tape.
func QKNorm[T tensor.Numeric](ctx context.Context, engine compute.Engine[T], q, k *tensor.TensorNumeric[T], epsilon float64) (qNorm, kNorm *tensor.TensorNumeric[T], err error) {
	if q == nil {
		return nil, nil, errors.New("query tensor (q) cannot be nil")
	}

	if k == nil {
		return nil, nil, errors.New("key tensor (k) cannot be nil")
	}

	if !q.ShapeEquals(k) {
		return nil, nil, fmt.Errorf("query and key tensors must have the same shape: q %v, k %v", q.Shape(), k.Shape())
	}

	ops := engine.Ops()
	eps := ops.FromFloat64(epsilon)

	normalizedQ, err := rmsNormalize(ctx, engine, q, eps)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to normalize Q: %w", err)
	}

	normalizedK, err := rmsNormalize(ctx, engine, k, eps)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to normalize K: %w", err)
	}

	return normalizedQ, normalizedK, nil
}

// rmsNormalize computes x / sqrt(mean(x^2) + epsilon) using engine primitives.
func rmsNormalize[T tensor.Numeric](ctx context.Context, engine compute.Engine[T], x *tensor.TensorNumeric[T], epsilon T) (*tensor.TensorNumeric[T], error) {
	// x^2 (element-wise)
	xSq, err := engine.Mul(ctx, x, x)
	if err != nil {
		return nil, err
	}

	// mean(x^2) -- axis -1 reduces all dimensions to scalar
	meanSq, err := engine.ReduceMean(ctx, xSq, -1, false)
	if err != nil {
		return nil, err
	}

	// mean(x^2) + epsilon
	meanSqEps, err := engine.AddScalar(ctx, meanSq, epsilon)
	if err != nil {
		return nil, err
	}

	// 1 / sqrt(mean(x^2) + epsilon)
	invRms, err := engine.Rsqrt(ctx, meanSqEps)
	if err != nil {
		return nil, err
	}

	// x * invRms (broadcasts scalar across x)
	return engine.Mul(ctx, x, invRms)
}
