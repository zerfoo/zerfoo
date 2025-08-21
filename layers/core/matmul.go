// Package core provides core layer implementations for the Zerfoo ML framework.
package core

import (
	"context"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/graph"
	"github.com/zerfoo/zerfoo/tensor"
)

// MatMul is a layer that performs matrix multiplication of two tensors.
type MatMul[T tensor.Numeric] struct {
	engine      compute.Engine[T]
	outputShape []int
}

// NewMatMul creates a new MatMul layer.
func NewMatMul[T tensor.Numeric](engine compute.Engine[T]) *MatMul[T] {
	return &MatMul[T]{
		engine: engine,
	}
}

// OutputShape returns the output shape of the MatMul layer.
func (m *MatMul[T]) OutputShape() []int {
	return m.outputShape
}

// Parameters returns no trainable parameters for the MatMul layer.
func (m *MatMul[T]) Parameters() []*graph.Parameter[T] {
	return nil
}

// Forward computes the matrix multiplication of two input tensors.
func (m *MatMul[T]) Forward(ctx context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if len(inputs) != 2 {
		panic("MatMul layer requires exactly 2 inputs")
	}
	
	a := inputs[0]
	b := inputs[1]
	
	// Calculate output shape for matrix multiplication
	aShape := a.Shape()
	bShape := b.Shape()
	
	// For 2D matrices: (M, K) x (K, N) -> (M, N)
	if len(aShape) >= 2 && len(bShape) >= 2 {
		m.outputShape = make([]int, len(aShape))
		copy(m.outputShape, aShape)
		m.outputShape[len(aShape)-1] = bShape[len(bShape)-1]
	} else {
		m.outputShape = aShape
	}
	
	return m.engine.MatMul(ctx, a, b)
}

// Backward computes the gradients for the MatMul layer.
func (m *MatMul[T]) Backward(ctx context.Context, outputGradient *tensor.TensorNumeric[T], inputs ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	if len(inputs) != 2 {
		panic("MatMul layer requires exactly 2 inputs")
	}
	
	a := inputs[0]
	b := inputs[1]
	
	// Gradient w.r.t. a: outputGradient @ b^T
	gradA, err := m.engine.MatMul(ctx, outputGradient, b)
	if err != nil {
		return nil, err
	}
	
	// Gradient w.r.t. b: a^T @ outputGradient
	gradB, err := m.engine.MatMul(ctx, a, outputGradient)
	if err != nil {
		return nil, err
	}
	
	return []*tensor.TensorNumeric[T]{gradA, gradB}, nil
}
