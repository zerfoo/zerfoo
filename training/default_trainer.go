// Package training provides the V2 trainer API using a Batch and pluggable strategy.
package training

import (
	"context"

	"github.com/zerfoo/zerfoo/graph"
	"github.com/zerfoo/zerfoo/tensor"
	opt "github.com/zerfoo/zerfoo/training/optimizer"
)

// DefaultTrainer encapsulates stable training components and delegates
// gradient computation to a strategy.
type DefaultTrainer[T tensor.Numeric] struct {
	g        *graph.Graph[T]
	loss     graph.Node[T]
	opt      opt.Optimizer[T]
	strategy GradientStrategy[T]
}

// NewDefaultTrainer constructs a new DefaultTrainer. If strategy is nil,
// DefaultBackpropStrategy is used.
func NewDefaultTrainer[T tensor.Numeric](
	g *graph.Graph[T],
	loss graph.Node[T],
	optimizer opt.Optimizer[T],
	strategy GradientStrategy[T],
) *DefaultTrainer[T] {
	if strategy == nil {
		strategy = NewDefaultBackpropStrategy[T]()
	}

	return &DefaultTrainer[T]{
		g:        g,
		loss:     loss,
		opt:      optimizer,
		strategy: strategy,
	}
}

// TrainStep performs a single training step using the configured strategy and optimizer.
func (t *DefaultTrainer[T]) TrainStep(
	ctx context.Context,
	g *graph.Graph[T],
	optimizer opt.Optimizer[T],
	inputs map[graph.Node[T]]*tensor.TensorNumeric[T],
	targets *tensor.TensorNumeric[T],
) (T, error) {
	batch := Batch[T]{
		Inputs:  inputs,
		Targets: targets,
	}
	lossVal, err := t.strategy.ComputeGradients(ctx, g, t.loss, batch)
	if err != nil {
		var zero T
		return zero, err
	}

	if err := optimizer.Step(ctx, g.Parameters()); err != nil {
		var zero T
		return zero, err
	}

	return lossVal, nil
}

// Statically assert that the type implements the Trainer interface.
var _ Trainer[float32] = (*DefaultTrainer[float32])(nil)
