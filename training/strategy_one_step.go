// Package training defines the one-step gradient approximation strategy.
package training

import (
	"context"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/ztensor/types"
)

// OneStepApproximationStrategy performs a one-step gradient approximation.
// It is designed for training recurrent models without full BPTT.
//
// Like DefaultBackpropStrategy, it maintains persistent (non-arena) gradient
// accumulators per parameter (issue #850); reuse one strategy instance across
// all samples and batches of a training run.
type OneStepApproximationStrategy[T tensor.Numeric] struct {
	grads gradAccumulator[T]
}

// NewOneStepApproximationStrategy constructs a OneStepApproximationStrategy.
func NewOneStepApproximationStrategy[T tensor.Numeric]() *OneStepApproximationStrategy[T] {
	return &OneStepApproximationStrategy[T]{}
}

// SetEngine configures an optional compute engine used to perform the
// persistent gradient accumulation as an in-place engine.Add with
// dst=accumulator. See DefaultBackpropStrategy.SetEngine.
func (s *OneStepApproximationStrategy[T]) SetEngine(e compute.Engine[T]) {
	s.grads.setEngine(e)
}

// ComputeGradients performs a forward pass and a one-step backward pass.
func (s *OneStepApproximationStrategy[T]) ComputeGradients(
	ctx context.Context,
	g *graph.Graph[T],
	loss graph.Node[T],
	batch Batch[T],
) (T, error) {
	return computeGradientsCommon[T](ctx, g, loss, batch, types.OneStepApproximation, &s.grads)
}

// Ensure OneStepApproximationStrategy implements the GradientStrategy interface.
var _ GradientStrategy[float32] = (*OneStepApproximationStrategy[float32])(nil)
