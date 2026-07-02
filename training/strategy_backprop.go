// Package training defines default backpropagation strategy.
package training

import (
	"context"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/ztensor/types"
)

// DefaultBackpropStrategy performs standard backpropagation through the loss
// and model graph.
//
// The strategy maintains persistent (non-arena) gradient accumulators per
// parameter (issue #850): after each Backward, any arena-backed
// Parameter.Gradient is accumulated into a parameter-owned persistent buffer
// and Parameter.Gradient is repointed at it, so per-sample engine ResetPool
// calls (the Wolf crossasset pattern) cannot corrupt accumulated gradients.
// The accumulators live as long as the strategy: reuse one strategy instance
// across all samples and batches of a training run.
type DefaultBackpropStrategy[T tensor.Numeric] struct {
	grads gradAccumulator[T]
}

// NewDefaultBackpropStrategy constructs a DefaultBackpropStrategy.
func NewDefaultBackpropStrategy[T tensor.Numeric]() *DefaultBackpropStrategy[T] {
	return &DefaultBackpropStrategy[T]{}
}

// SetEngine configures an optional compute engine used to perform the
// persistent gradient accumulation as an in-place engine.Add with
// dst=accumulator (device-side, no host round-trip). Without an engine a
// host read-modify-writeback fallback is used, which is correct but slower
// for device gradients.
func (s *DefaultBackpropStrategy[T]) SetEngine(e compute.Engine[T]) {
	s.grads.setEngine(e)
}

// ComputeGradients runs forward pass, computes loss, runs backward passes,
// and leaves parameter gradients populated on the graph's parameters.
func (s *DefaultBackpropStrategy[T]) ComputeGradients(
	ctx context.Context,
	g *graph.Graph[T],
	loss graph.Node[T],
	batch Batch[T],
) (T, error) {
	return computeGradientsCommon[T](ctx, g, loss, batch, types.FullBackprop, &s.grads)
}

// ComputeGradientsTensor is ComputeGradients without the final host
// readback of the loss value: it returns the loss tensor produced by the
// loss node. On GPU engines the readback is a D2H copy, which is illegal
// inside a CUDA-graph capture region; CaptureReplayRunner records the step
// through this variant and reads the loss only after the captured graph
// has executed. Callers outside a capture region can read the value with
// lossTensor.Data()[0].
func (s *DefaultBackpropStrategy[T]) ComputeGradientsTensor(
	ctx context.Context,
	g *graph.Graph[T],
	loss graph.Node[T],
	batch Batch[T],
) (*tensor.TensorNumeric[T], error) {
	return computeGradientsTensorCommon[T](ctx, g, loss, batch, types.FullBackprop, &s.grads)
}

// Statically assert that the type implements the GradientStrategy interface.
var _ GradientStrategy[float32] = (*DefaultBackpropStrategy[float32])(nil)
