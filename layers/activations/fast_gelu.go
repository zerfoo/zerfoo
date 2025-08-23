// Package activations provides activation function layers.
package activations

import (
	"context"
	"fmt"
	"math"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/graph"
	"github.com/zerfoo/zerfoo/tensor"
)

// FastGelu is an approximation of the GELU activation function.
type FastGelu[T tensor.Float] struct {
	engine compute.Engine[T]
}

// NewFastGelu creates a new FastGelu layer.
func NewFastGelu[T tensor.Float](engine compute.Engine[T]) *FastGelu[T] {
	return &FastGelu[T]{engine: engine}
}

// Forward applies the forward pass of the FastGelu layer.
func (g *FastGelu[T]) Forward(ctx context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if len(inputs) != 1 {
		return nil, fmt.Errorf("FastGelu expects 1 input, got %d", len(inputs))
	}
	x := inputs[0]

	// y = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
	//
	// Intermediate calculations:
	// 1. x^3
	x3, err := g.engine.Mul(ctx, x, x)
	if err != nil {
		return nil, err
	}
	x3, err = g.engine.Mul(ctx, x3, x)
	if err != nil {
		return nil, err
	}

	// 2. 0.044715 * x^3
	term1, err := g.engine.MulScalar(ctx, x3, T(0.044715))
	if err != nil {
		return nil, err
	}

	// 3. x + 0.044715 * x^3
	term2, err := g.engine.Add(ctx, x, term1)
	if err != nil {
		return nil, err
	}

	// 4. sqrt(2/pi) * (x + 0.044715 * x^3)
	term3, err := g.engine.MulScalar(ctx, term2, T(math.Sqrt(2/math.Pi)))
	if err != nil {
		return nil, err
	}

	// 5. tanh(...)
	tanh, err := g.engine.UnaryOp(ctx, term3, func(val T) T {
		return T(math.Tanh(float64(val)))
	})
	if err != nil {
		return nil, err
	}

	// 6. 1 + tanh(...)
	term4, err := g.engine.AddScalar(ctx, tanh, T(1))
	if err != nil {
		return nil, err
	}

	// 7. x * (1 + tanh(...))
	term5, err := g.engine.Mul(ctx, x, term4)
	if err != nil {
		return nil, err
	}

	// 8. 0.5 * ...
	output, err := g.engine.MulScalar(ctx, term5, T(0.5))
	if err != nil {
		return nil, err
	}

	return output, nil
}

// Backward applies the backward pass of the FastGelu layer.
func (g *FastGelu[T]) Backward(_ context.Context, _ *tensor.TensorNumeric[T], _ ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	return nil, nil
}

// OpType returns the operation type of the FastGelu layer.
func (g *FastGelu[T]) OpType() string {
	return "FastGelu"
}

// Attributes returns nil for the FastGelu layer.
func (g *FastGelu[T]) Attributes() map[string]interface{} {
	return nil
}

// Parameters returns the learnable parameters of the layer.
func (g *FastGelu[T]) Parameters() []*graph.Parameter[T] {
	return nil
}

// OutputShape returns the output shape of the layer.
func (g *FastGelu[T]) OutputShape() []int {
	return nil
}
