// Package optimizer provides various optimization algorithms for neural networks.
package optimizer

import (
	"context"
	"fmt"
	"math"

	"github.com/zerfoo/float16"
	"github.com/zerfoo/float8"
	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/tensor"
)

// AdamW implements the AdamW optimizer.
type AdamW[T tensor.Numeric] struct {
	engine       compute.Engine[T]
	learningRate T
	beta1        T
	beta2        T
	epsilon      T
	weightDecay  T
	maxGradNorm  float64 // If > 0, clip global gradient norm to this value.

	// State variables for each parameter
	m map[*graph.Parameter[T]]*tensor.TensorNumeric[T] // First moment estimates
	v map[*graph.Parameter[T]]*tensor.TensorNumeric[T] // Second moment estimates
	t int                                              // Timestep
}

// NewAdamW creates a new AdamW optimizer.
func NewAdamW[T tensor.Numeric](engine compute.Engine[T], learningRate, beta1, beta2, epsilon, weightDecay T) *AdamW[T] {
	return &AdamW[T]{
		engine:       engine,
		learningRate: learningRate,
		beta1:        beta1,
		beta2:        beta2,
		epsilon:      epsilon,
		weightDecay:  weightDecay,
		m:            make(map[*graph.Parameter[T]]*tensor.TensorNumeric[T]),
		v:            make(map[*graph.Parameter[T]]*tensor.TensorNumeric[T]),
		t:            0,
	}
}

// SetMaxGradNorm sets the maximum gradient norm for gradient clipping.
// If maxGradNorm <= 0, gradient clipping is disabled.
func (a *AdamW[T]) SetMaxGradNorm(maxGradNorm float64) {
	a.maxGradNorm = maxGradNorm
}

// Step updates the parameters based on their gradients.
func (a *AdamW[T]) Step(ctx context.Context, params []*graph.Parameter[T]) error {
	// NaN/Inf guard and optional gradient clipping.
	if err := a.guardAndClipGradients(params); err != nil {
		return err
	}

	a.t++ // Increment timestep

	// Bias correction terms
	ops := a.engine.Ops()
	one := ops.FromFloat64(1.0)
	tAsT := ops.FromFloat64(float64(a.t))
	// sqrt(1 - beta2^t) / (1 - beta1^t)
	numer := ops.Sqrt(ops.Sub(one, ops.Pow(a.beta2, tAsT)))
	denom := ops.Sub(one, ops.Pow(a.beta1, tAsT))
	biasCorr := ops.Div(numer, denom)
	alpha := ops.Mul(a.learningRate, biasCorr)

	for _, param := range params {
		grad := param.Gradient

		if grad == nil {
			continue // Skip if no gradient
		}

		// Initialize m and v for this parameter if not already done
		if _, ok := a.m[param]; !ok {
			mTensor, err := tensor.New[T](param.Value.Shape(), nil)
			if err != nil {
				return err
			}

			if err := a.engine.Zeros(ctx, mTensor, param.Value.Shape()); err != nil {
				return err
			}

			a.m[param] = mTensor

			vTensor, err := tensor.New[T](param.Value.Shape(), nil)
			if err != nil {
				return err
			}

			if err := a.engine.Zeros(ctx, vTensor, param.Value.Shape()); err != nil {
				return err
			}

			a.v[param] = vTensor
		}

		m := a.m[param]
		v := a.v[param]
		paramValue := param.Value

		// Update biased first moment estimate: m = beta1 * m + (1 - beta1) * grad
		mNew, err := a.engine.MulScalar(ctx, m, a.beta1, nil)
		if err != nil {
			return err
		}

		gradScaled, err := a.engine.MulScalar(ctx, grad, ops.Sub(one, a.beta1), nil)
		if err != nil {
			return err
		}

		m, err = a.engine.Add(ctx, mNew, gradScaled, m) // Update m in-place
		if err != nil {
			return err
		}

		// Update biased second raw moment estimate: v = beta2 * v + (1 - beta2) * grad^2
		vNew, err := a.engine.MulScalar(ctx, v, a.beta2, nil)
		if err != nil {
			return err
		}

		gradSquared, err := a.engine.Mul(ctx, grad, grad, nil)
		if err != nil {
			return err
		}

		gradSquaredScaled, err := a.engine.MulScalar(ctx, gradSquared, ops.Sub(one, a.beta2), nil)
		if err != nil {
			return err
		}

		v, err = a.engine.Add(ctx, vNew, gradSquaredScaled, v)
		if err != nil { // Update v in-place
			return err
		}

		// Compute update: update = alpha * m_hat / (sqrt(v_hat) + epsilon)
		// m_hat is already bias-corrected by alpha
		// v_hat is already bias-corrected by alpha
		sqrtV, err := a.engine.Sqrt(ctx, v, nil)
		if err != nil {
			return err
		}

		sqrtVPlusEpsilon, err := a.engine.AddScalar(ctx, sqrtV, a.epsilon, nil)
		if err != nil {
			return err
		}

		updateTerm, err := a.engine.Div(ctx, m, sqrtVPlusEpsilon, nil)
		if err != nil {
			return err
		}

		updateTermScaled, err := a.engine.MulScalar(ctx, updateTerm, alpha, nil)
		if err != nil {
			return err
		}

		// Apply weight decay: param = param - (learningRate * weightDecay) * param
		lrWd := ops.Mul(a.learningRate, a.weightDecay)
		weightDecayTerm, err := a.engine.MulScalar(ctx, paramValue, lrWd, nil)
		if err != nil {
			return err
		}

		// Final update: param = param - updateTermScaled - weightDecayTerm
		paramNew, err := a.engine.Sub(ctx, paramValue, updateTermScaled, nil)
		if err != nil {
			return err
		}

		param.Value, err = a.engine.Sub(ctx, paramNew, weightDecayTerm, paramValue)
		if err != nil { // Update paramValue in-place
			return err
		}

		// Clear gradient for next step.
		// Use engine.Fill instead of param.ClearGradient() because the latter
		// modifies a D2H copy that is never written back to GPU storage.
		var zero T
		if err := a.engine.Fill(ctx, param.Gradient, zero); err != nil {
			param.ClearGradient() // Fallback to CPU path.
		}
	}

	return nil
}

// numericToFloat64 converts a tensor.Numeric value to float64.
func numericToFloat64[T tensor.Numeric](v T) float64 {
	switch val := any(v).(type) {
	case float32:
		return float64(val)
	case float64:
		return val
	case int:
		return float64(val)
	case int8:
		return float64(val)
	case int16:
		return float64(val)
	case int32:
		return float64(val)
	case int64:
		return float64(val)
	case uint:
		return float64(val)
	case uint8:
		return float64(val)
	case uint32:
		return float64(val)
	case uint64:
		return float64(val)
	case float16.Float16:
		return float64(val.ToFloat32())
	case float16.BFloat16:
		return float64(val.ToFloat32())
	case float8.Float8:
		return val.ToFloat64()
	default:
		return 0
	}
}

// guardAndClipGradients checks all gradient values for NaN/Inf and optionally
// clips the global gradient norm to MaxGradNorm.
func (a *AdamW[T]) guardAndClipGradients(params []*graph.Parameter[T]) error {
	ops := a.engine.Ops()
	var globalNormSq float64

	for _, param := range params {
		grad := param.Gradient
		if grad == nil {
			continue
		}

		data := grad.Data()
		for i, v := range data {
			f := numericToFloat64(v)
			if math.IsNaN(f) {
				return fmt.Errorf("adamw: NaN detected in gradient of parameter %q at index %d", param.Name, i)
			}

			if math.IsInf(f, 0) {
				return fmt.Errorf("adamw: Inf detected in gradient of parameter %q at index %d", param.Name, i)
			}

			globalNormSq += f * f
		}
	}

	if a.maxGradNorm > 0 {
		globalNorm := math.Sqrt(globalNormSq)
		if globalNorm > a.maxGradNorm {
			scale := a.maxGradNorm / globalNorm
			for _, param := range params {
				grad := param.Gradient
				if grad == nil {
					continue
				}

				data := grad.Data()
				for i, v := range data {
					f := numericToFloat64(v)
					data[i] = ops.FromFloat64(f * scale)
				}

				grad.SetData(data)
			}
		}
	}

	return nil
}

// SetLR sets the learning rate. This is typically called by a scheduler.
func (a *AdamW[T]) SetLR(lr T) {
	a.learningRate = lr
}

// Statically assert that the type implements the Optimizer interface.
var _ Optimizer[float32] = (*AdamW[float32])(nil)
