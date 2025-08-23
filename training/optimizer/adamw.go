// Package optimizer provides various optimization algorithms for neural networks.
package optimizer

import (
	"context"
	"math"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/graph"
	"github.com/zerfoo/zerfoo/tensor"
)

// AdamW implements the AdamW optimizer.
type AdamW[T tensor.Numeric] struct {
	engine       compute.Engine[T]
	learningRate T
	beta1        T
	beta2        T
	epsilon      T
	weightDecay  T

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

// Step updates the parameters based on their gradients.
func (a *AdamW[T]) Step(ctx context.Context, params []*graph.Parameter[T]) error {
	a.t++ // Increment timestep

	// Bias correction terms
	alpha := a.learningRate * T(math.Sqrt(float64(1.0-math.Pow(float64(a.beta2), float64(a.t))))/(1.0-math.Pow(float64(a.beta1), float64(a.t))))

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

		gradScaled, err := a.engine.MulScalar(ctx, grad, T(1.0)-a.beta1, nil)
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

		gradSquaredScaled, err := a.engine.MulScalar(ctx, gradSquared, T(1.0)-a.beta2, nil)
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

		// Apply weight decay: param = param - learningRate * weightDecay * param
		weightDecayTerm, err := a.engine.MulScalar(ctx, paramValue, a.learningRate*a.weightDecay, nil)
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

		// Clear gradient for next step
		param.ClearGradient()
	}

	return nil
}
