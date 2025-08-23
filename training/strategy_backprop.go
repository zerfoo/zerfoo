// Package training defines default backpropagation strategy.
package training

import (
	"context"
	"fmt"

	"github.com/zerfoo/zerfoo/graph"
	"github.com/zerfoo/zerfoo/tensor"
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
	// Materialize inputs in graph input order
	var inputSlice []*tensor.TensorNumeric[T]
	for _, inputNode := range g.Inputs() {
		inputSlice = append(inputSlice, batch.Inputs[inputNode])
	}

	// Forward pass
	output, err := g.Forward(ctx, inputSlice...)
	if err != nil {
		return 0, fmt.Errorf("forward pass failed: %w", err)
	}

	// Loss forward
	lossTensor, err := loss.Forward(ctx, output, batch.Targets)
	if err != nil {
		return 0, fmt.Errorf("loss computation failed: %w", err)
	}

	// Loss backward
	lossGrads, err := loss.Backward(ctx, lossTensor, output, batch.Targets)
	if err != nil {
		return 0, fmt.Errorf("loss backward pass failed: %w", err)
	}

	// Model backward
	if err := g.Backward(ctx, lossGrads[0]); err != nil {
		return 0, fmt.Errorf("model backward pass failed: %w", err)
	}

	return lossTensor.Data()[0], nil
}
