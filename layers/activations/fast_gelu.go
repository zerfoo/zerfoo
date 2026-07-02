// Package activations provides activation function layers.
package activations

import (
	"context"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/ztensor/types"
)

// FastGelu is an approximation of the GELU activation function.
// It delegates its Forward pass to Gelu (same approximation) but has
// a no-op Backward, making it suitable for inference-only paths.
type FastGelu[T tensor.Float] struct {
	graph.NoParameters[T]
	gelu *Gelu[T]
}

// NewFastGelu creates a new FastGelu layer.
func NewFastGelu[T tensor.Float](engine compute.Engine[T]) *FastGelu[T] {
	return &FastGelu[T]{gelu: NewGelu(engine, engine.Ops())}
}

// Forward applies the forward pass of the FastGelu layer.
// Uses the same GELU approximation as Gelu:
// y = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
func (g *FastGelu[T]) Forward(ctx context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	return g.gelu.Forward(ctx, inputs...)
}

// Backward is a no-op for FastGelu (inference-only).
func (g *FastGelu[T]) Backward(_ context.Context, _ types.BackwardMode, _ *tensor.TensorNumeric[T], _ ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
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

// OutputShape returns the output shape of the layer.
func (g *FastGelu[T]) OutputShape() []int {
	return nil
}

// Statically assert that the type implements the graph.Node interface.
var _ graph.Node[float32] = (*FastGelu[float32])(nil)
