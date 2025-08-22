// Package core provides core layer implementations for the Zerfoo ML framework.
package core

import (
	"context"
	"fmt"

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

// Forward computes the matrix multiplication.
func (m *MatMul[T]) Forward(ctx context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if len(inputs) != 2 {
		return nil, fmt.Errorf("MatMul layer requires exactly 2 inputs, got %d", len(inputs))
	}

	a, b := inputs[0], inputs[1]
	
	// Check for dimension mismatch and attempt transpose if needed
	if a.Shape()[len(a.Shape())-1] != b.Shape()[0] {
		// Check if this is a case where b needs to be transposed
		if len(a.Shape()) >= 2 && len(b.Shape()) == 2 {
			aInner := a.Shape()[len(a.Shape())-1]
			bInner := b.Shape()[1]
			
			// If a's inner dimension matches b's inner dimension, we might need to transpose b
			if aInner == bInner {
				bTransposed, err := m.engine.Transpose(ctx, b, []int{1, 0})
				if err != nil {
					return nil, fmt.Errorf("failed to transpose second operand: %w", err)
				}
				result, err := m.engine.MatMul(ctx, a, bTransposed)
				if err != nil {
					return nil, err
				}
				m.outputShape = result.Shape()
				return result, nil
			}
		}
		
		return nil, fmt.Errorf("incompatible dimensions for matrix multiplication: %v x %v", a.Shape(), b.Shape())
	}

	result, err := m.engine.MatMul(ctx, a, b)
	if err != nil {
		return nil, err
	}

	m.outputShape = result.Shape()
	return result, nil
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

// OpType returns the operation type of the MatMul layer.
func (m *MatMul[T]) OpType() string {
	return "MatMul"
}

// Attributes returns nil for the MatMul layer.
func (m *MatMul[T]) Attributes() map[string]interface{} {
	return nil
}
