// Package training defines default backpropagation strategy.
package training

import (
	"context"

	"github.com/zerfoo/zerfoo/graph"
	"github.com/zerfoo/zerfoo/tensor"
	"github.com/zerfoo/zerfoo/types"
)

// DefaultBackpropStrategy performs standard backpropagation through the loss
// and model graph.
type DefaultBackpropStrategy[T tensor.Numeric] struct{}

// NewDefaultBackpropStrategy constructs a DefaultBackpropStrategy.
func NewDefaultBackpropStrategy[T tensor.Numeric]() *DefaultBackpropStrategy[T] {
	return &DefaultBackpropStrategy[T]{}
}

// ComputeGradients runs forward pass, computes loss, runs backward passes,
// and leaves parameter gradients populated on the graph's parameters.
func (s *DefaultBackpropStrategy[T]) ComputeGradients(
	ctx context.Context,
	g *graph.Graph[T],
	loss graph.Node[T],
	batch Batch[T],
) (T, error) {
	return computeGradientsCommon[T](ctx, g, loss, batch, types.FullBackprop)
}

// Statically assert that the type implements the GradientStrategy interface.
var _ GradientStrategy[float32] = (*DefaultBackpropStrategy[float32])(nil)
