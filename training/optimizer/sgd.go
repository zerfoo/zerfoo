// Package optimizer provides various optimization algorithms for neural networks.
package optimizer

import (
	"context"
	"fmt"

	"github.com/zerfoo/float16"
	"github.com/zerfoo/float8"
	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/graph"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
)

// SGD implements the stochastic gradient descent optimizer.
type SGD[T tensor.Numeric] struct {
	engine       compute.Engine[T]
	ops          numeric.Arithmetic[T]
	learningRate T
}

// NewSGD creates a new SGD optimizer.
func NewSGD[T tensor.Numeric](engine compute.Engine[T], ops numeric.Arithmetic[T], learningRate float32) *SGD[T] {
	return &SGD[T]{
		engine:       engine,
		ops:          ops,
		learningRate: ops.FromFloat32(learningRate),
	}
}

// Clip clips the gradients of the parameters by a threshold.
func (s *SGD[T]) Clip(ctx context.Context, params []*graph.Parameter[T], threshold float32) {
	minVal := s.ops.FromFloat32(-threshold)
	maxVal := s.ops.FromFloat32(threshold)

	for _, p := range params {
		p.Gradient, _ = s.engine.UnaryOp(ctx, p.Gradient, func(g T) T {
			switch any(g).(type) {
			case float32:
				gF32 := any(g).(float32)
				maxF32 := any(maxVal).(float32)
				minF32 := any(minVal).(float32)

				if gF32 > maxF32 {
					return maxVal
				}

				if gF32 < minF32 {
					return minVal
				}
			case float16.Float16:
				gF32 := any(g).(float16.Float16).ToFloat32()
				maxF32 := any(maxVal).(float16.Float16).ToFloat32()
				minF32 := any(minVal).(float16.Float16).ToFloat32()

				if gF32 > maxF32 {
					return maxVal
				}

				if gF32 < minF32 {
					return minVal
				}
			case float8.Float8:
				gF32 := any(g).(float8.Float8).ToFloat32()
				maxF32 := any(maxVal).(float8.Float8).ToFloat32()
				minF32 := any(minVal).(float8.Float8).ToFloat32()

				if gF32 > maxF32 {
					return maxVal
				}

				if gF32 < minF32 {
					return minVal
				}
			}

			return g
		}, nil)
	}
}

// Step updates the parameters based on their gradients.
func (s *SGD[T]) Step(ctx context.Context, params []*graph.Parameter[T]) error {
	for _, p := range params {
		// Create a tensor for the learning rate with the same shape as the gradient.
		lrData := make([]T, p.Gradient.Size())
		for i := range lrData {
			lrData[i] = s.learningRate
		}

		lrTensor, _ := tensor.New[T](p.Gradient.Shape(), lrData)

		// scaled_grad = learning_rate * gradient
		scaledGrad, err := s.engine.Mul(ctx, lrTensor, p.Gradient, nil)
		if err != nil {
			return fmt.Errorf("failed to scale gradient: %w", err)
		}

		// value = value - scaled_grad
		newValue, err := s.engine.Sub(ctx, p.Value, scaledGrad, nil)
		if err != nil {
			return fmt.Errorf("failed to update parameter value: %w", err)
		}

		p.Value = newValue
	}

	return nil
}
