// Package core provides core layer implementations for the Zerfoo ML framework.
package core

import (
	"context"
	"fmt"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/graph"
	"github.com/zerfoo/zerfoo/tensor"
	"github.com/zerfoo/zerfoo/types"
)

// Sub is a layer that performs element-wise subtraction of two tensors.
type Sub[T tensor.Numeric] struct {
	engine      compute.Engine[T]
	outputShape []int
}

// NewSub creates a new Sub layer.
func NewSub[T tensor.Numeric](engine compute.Engine[T]) *Sub[T] {
	return &Sub[T]{
		engine: engine,
	}
}

// OutputShape returns the output shape of the Sub layer.
func (s *Sub[T]) OutputShape() []int {
	return s.outputShape
}

// Parameters returns no trainable parameters for the Sub layer.
func (s *Sub[T]) Parameters() []*graph.Parameter[T] {
	return nil
}

// Forward computes the element-wise subtraction of two input tensors (a - b).
func (s *Sub[T]) Forward(ctx context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if len(inputs) == 1 {
		// Handle single input case - subtract from zero (negate)
		a := inputs[0]
		s.outputShape = a.Shape()

		// Create a zero tensor with same shape
		zeroTensor, err := tensor.New[T](a.Shape(), nil)
		if err != nil {
			return nil, fmt.Errorf("failed to create zero tensor: %w", err)
		}

		return s.engine.Sub(ctx, zeroTensor, a)
	}

	if len(inputs) == 2 {
		a := inputs[0]
		b := inputs[1]

		// The output shape should be the broadcasted shape of the two inputs
		// For simplicity, we'll assume they have compatible shapes
		s.outputShape = a.Shape()

		return s.engine.Sub(ctx, a, b)
	}

	return nil, fmt.Errorf("Sub layer expects 1 or 2 inputs, got %d", len(inputs))
}

// Backward computes the gradients for the Sub layer.
func (s *Sub[T]) Backward(ctx context.Context, mode types.BackwardMode, outputGradient *tensor.TensorNumeric[T], inputs ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	if len(inputs) != 2 {
		panic("Sub layer requires exactly 2 inputs")
	}

	// Gradient w.r.t. a: outputGradient (derivative of a - b w.r.t. a is 1)
	gradA := outputGradient

	// Gradient w.r.t. b: -outputGradient (derivative of a - b w.r.t. b is -1)
	// Create a tensor of -1s and multiply
	ops := s.engine.Ops()
	negOne := ops.FromFloat32(-1.0)

	gradB, err := s.engine.UnaryOp(ctx, outputGradient, func(x T) T { return ops.Mul(x, negOne) })
	if err != nil {
		return nil, err
	}

	return []*tensor.TensorNumeric[T]{gradA, gradB}, nil
}

// OpType returns the operation type of the Sub layer.
func (s *Sub[T]) OpType() string {
	return "Sub"
}

// Attributes returns nil for the Sub layer.
func (s *Sub[T]) Attributes() map[string]interface{} {
	return nil
}
