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

// MatrixMultiplierOptions represents configuration options for MatrixMultiplier.
type MatrixMultiplierOptions[T tensor.Numeric] struct {
	// No specific options for now, but kept for consistency.
}

// MatrixMultiplierOption applies an option to MatrixMultiplierOptions.
type MatrixMultiplierOption[T tensor.Numeric] func(*MatrixMultiplierOptions[T])

// NewMatrixMultiplier creates a new matrix multiplier.
func NewMatrixMultiplier[T tensor.Numeric](engine compute.Engine[T], opts ...MatrixMultiplierOption[T]) *MatrixMultiplier[T] {
	options := &MatrixMultiplierOptions[T]{}
	for _, opt := range opts {
		opt(options)
	}

	return &MatrixMultiplier[T]{engine: engine}
}

// Multiply performs matrix multiplication: result = a * b.
func (m *MatrixMultiplier[T]) Multiply(ctx context.Context, a, b *tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	return m.engine.MatMul(ctx, a, b)
}

// MultiplyWithDestination performs matrix multiplication with a pre-allocated destination tensor.
func (m *MatrixMultiplier[T]) MultiplyWithDestination(ctx context.Context, a, b, dst *tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	return m.engine.MatMul(ctx, a, b, dst)
}

// Transpose transposes a matrix.
func (m *MatrixMultiplier[T]) Transpose(ctx context.Context, a *tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	return m.engine.Transpose(ctx, a, []int{1, 0})
}

// TransposeWithDestination transposes a matrix with a pre-allocated destination tensor.
func (m *MatrixMultiplier[T]) TransposeWithDestination(ctx context.Context, a, dst *tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	return m.engine.Transpose(ctx, a, []int{1, 0}, dst)
}
