// Package training defines the one-step gradient approximation strategy.
package training

import (
	"context"
	"fmt"

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
	var zero T
	// Materialize inputs in graph input order
	var inputSlice []*tensor.TensorNumeric[T]
	for _, inputNode := range g.Inputs() {
		inputSlice = append(inputSlice, batch.Inputs[inputNode])
	}

	// Forward pass
	output, err := g.Forward(ctx, inputSlice...)
	if err != nil {
		return zero, fmt.Errorf("forward pass failed: %w", err)
	}

	// Loss forward
	lossTensor, err := loss.Forward(ctx, output, batch.Targets)
	if err != nil {
		return zero, fmt.Errorf("loss computation failed: %w", err)
	}

	// Loss backward
	lossGrads, err := loss.Backward(ctx, types.OneStepApproximation, lossTensor, output, batch.Targets)
	if err != nil {
		return zero, fmt.Errorf("loss backward pass failed: %w", err)
	}

	// Model backward
	if err := g.Backward(ctx, types.OneStepApproximation, lossGrads[0]); err != nil {
		return zero, fmt.Errorf("model backward pass failed: %w", err)
	}

	return lossTensor.Data()[0], nil
}

// Ensure OneStepApproximationStrategy implements the GradientStrategy interface.
var _ GradientStrategy[float32] = (*OneStepApproximationStrategy[float32])(nil)
