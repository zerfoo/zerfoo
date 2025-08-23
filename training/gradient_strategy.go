// Package training defines training-time gradient computation strategies.
package training

import (
	"context"

	"github.com/zerfoo/zerfoo/graph"
	"github.com/zerfoo/zerfoo/tensor"
)

// GradientStrategy encapsulates how to compute gradients for a training step.
//
// Implementations may perform standard backprop through the loss, use
// approximations, or incorporate auxiliary losses (e.g., deep supervision).
// The strategy must leave parameter gradients populated on the graph's
// parameters so that the optimizer can apply updates afterwards.
type GradientStrategy[T tensor.Numeric] interface {
	ComputeGradients(
		ctx context.Context,
		g *graph.Graph[T],
		loss graph.Node[T],
		batch Batch[T],
	) (lossValue T, err error)
}
