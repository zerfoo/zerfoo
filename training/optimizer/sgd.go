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

func NewSGD[T tensor.Numeric](engine compute.Engine[T], ops numeric.Arithmetic[T], learningRate float32) *SGD[T] {
	return &SGD[T]{
		engine:       engine,
		ops:          ops,
		learningRate: ops.FromFloat32(learningRate),
	}
}

func (s *SGD[T]) Clip(params []*graph.Parameter[T], threshold float32) {
	minVal := s.ops.FromFloat32(-threshold)
	maxVal := s.ops.FromFloat32(threshold)
	ctx := context.Background()

	for _, p := range params {
		p.Gradient, _ = s.engine.UnaryOp(ctx, p.Gradient, func(g T) T {
			switch any(g).(type) {
			case float32:
				g_f32 := any(g).(float32)
				max_f32 := any(maxVal).(float32)
				min_f32 := any(minVal).(float32)

				if g_f32 > max_f32 {
					return maxVal
				}
				if g_f32 < min_f32 {
					return minVal
				}
			case float16.Float16:
				g_f32 := any(g).(float16.Float16).ToFloat32()
				max_f32 := any(maxVal).(float16.Float16).ToFloat32()
				min_f32 := any(minVal).(float16.Float16).ToFloat32()

				if g_f32 > max_f32 {
					return maxVal
				}
				if g_f32 < min_f32 {
					return minVal
				}
			case float8.Float8:
				g_f32 := any(g).(float8.Float8).ToFloat32()
				max_f32 := any(maxVal).(float8.Float8).ToFloat32()
				min_f32 := any(minVal).(float8.Float8).ToFloat32()

				if g_f32 > max_f32 {
					return maxVal
				}
				if g_f32 < min_f32 {
					return minVal
				}
			}
			return g
		}, nil)
	}
}

// Step updates the parameters based on their gradients.
func (s *SGD[T]) Step(params []*graph.Parameter[T]) {
	ctx := context.Background()
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
			panic(fmt.Sprintf("failed to scale gradient: %v", err))
		}

		// value = value - scaled_grad
		newValue, err := s.engine.Sub(ctx, p.Value, scaledGrad, nil)
		if err != nil {
			panic(fmt.Sprintf("failed to update parameter value: %v", err))
		}
		p.Value = newValue
	}
}
