package components

import (
	"context"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/tensor"
)

// MatrixMultiplier handles matrix multiplication operations for layers.
type MatrixMultiplier[T tensor.Numeric] struct {
	engine compute.Engine[T]
}

// NewMatrixMultiplier creates a new matrix multiplier.
func NewMatrixMultiplier[T tensor.Numeric](engine compute.Engine[T]) *MatrixMultiplier[T] {
	return &MatrixMultiplier[T]{engine: engine}
}

// Multiply performs matrix multiplication: result = a * b.
func (m *MatrixMultiplier[T]) Multiply(ctx context.Context, a, b *tensor.Tensor[T]) (*tensor.Tensor[T], error) {
	return m.engine.MatMul(ctx, a, b)
}

// MultiplyWithDestination performs matrix multiplication with a pre-allocated destination tensor.
func (m *MatrixMultiplier[T]) MultiplyWithDestination(ctx context.Context, a, b, dst *tensor.Tensor[T]) (*tensor.Tensor[T], error) {
	return m.engine.MatMul(ctx, a, b, dst)
}

// Transpose transposes a matrix.
func (m *MatrixMultiplier[T]) Transpose(ctx context.Context, a *tensor.Tensor[T]) (*tensor.Tensor[T], error) {
	return m.engine.Transpose(ctx, a, []int{1, 0})
}

// TransposeWithDestination transposes a matrix with a pre-allocated destination tensor.
func (m *MatrixMultiplier[T]) TransposeWithDestination(ctx context.Context, a, dst *tensor.Tensor[T]) (*tensor.Tensor[T], error) {
	return m.engine.Transpose(ctx, a, []int{1, 0}, dst)
}
