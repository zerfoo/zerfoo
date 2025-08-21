// Package core provides core layer implementations for the Zerfoo ML framework.
package core

import (
	"context"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/graph"
	"github.com/zerfoo/zerfoo/tensor"
)

// Mul is a layer that performs element-wise multiplication of two tensors.
type Mul[T tensor.Numeric] struct {
	engine      compute.Engine[T]
	outputShape []int
}

// NewMul creates a new Mul layer.
func NewMul[T tensor.Numeric](engine compute.Engine[T]) *Mul[T] {
	return &Mul[T]{
		engine: engine,
	}
}

// OutputShape returns the output shape of the Mul layer.
func (m *Mul[T]) OutputShape() []int {
	return m.outputShape
}

// Parameters returns no trainable parameters for the Mul layer.
func (m *Mul[T]) Parameters() []*graph.Parameter[T] {
	return nil
}

// Forward computes the element-wise multiplication of two input tensors.
func (m *Mul[T]) Forward(ctx context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if len(inputs) != 2 {
		panic("Mul layer requires exactly 2 inputs")
	}
	
	a := inputs[0]
	b := inputs[1]
	
	// The output shape should be the broadcasted shape of the two inputs
	// For simplicity, we'll assume they have compatible shapes
	m.outputShape = a.Shape()
	
	return m.engine.Mul(ctx, a, b)
}

// Backward computes the gradients for the Mul layer.
func (m *Mul[T]) Backward(ctx context.Context, outputGradient *tensor.TensorNumeric[T], inputs ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	if len(inputs) != 2 {
		panic("Mul layer requires exactly 2 inputs")
	}
	
	a := inputs[0]
	b := inputs[1]
	
	// Gradient w.r.t. a: outputGradient * b
	gradA, err := m.engine.Mul(ctx, outputGradient, b)
	if err != nil {
		return nil, err
	}
	
	// Gradient w.r.t. b: outputGradient * a
	gradB, err := m.engine.Mul(ctx, outputGradient, a)
	if err != nil {
		return nil, err
	}
	
	return []*tensor.TensorNumeric[T]{gradA, gradB}, nil
}
