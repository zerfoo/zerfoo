// Package training defines the one-step gradient approximation strategy.
package training

import (
	"context"

	"github.com/zerfoo/zerfoo/graph"
	"github.com/zerfoo/zerfoo/tensor"
	"github.com/zerfoo/zerfoo/types"
)

// OneStepApproximationStrategy performs a one-step gradient approximation.
// It is designed for training recurrent models without full BPTT.
type OneStepApproximationStrategy[T tensor.Numeric] struct{}

// NewOneStepApproximationStrategy constructs a OneStepApproximationStrategy.
func NewOneStepApproximationStrategy[T tensor.Numeric]() *OneStepApproximationStrategy[T] {
	return &OneStepApproximationStrategy[T]{}
}

// ComputeGradients performs a forward pass and a one-step backward pass.
func (s *OneStepApproximationStrategy[T]) ComputeGradients(
	ctx context.Context,
	g *graph.Graph[T],
	loss graph.Node[T],
	batch Batch[T],
) (T, error) {
	return computeGradientsCommon[T](ctx, g, loss, batch, types.OneStepApproximation)
}

// Ensure OneStepApproximationStrategy implements the GradientStrategy interface.
var _ GradientStrategy[float32] = (*OneStepApproximationStrategy[float32])(nil)
