package training

import (
	"context"
	"fmt"

	"github.com/zerfoo/zerfoo/graph"
	"github.com/zerfoo/zerfoo/tensor"
	"github.com/zerfoo/zerfoo/types"
)

// computeGradientsCommon consolidates the shared logic between training strategies.
func computeGradientsCommon[T tensor.Numeric](
	ctx context.Context,
	g *graph.Graph[T],
	loss graph.Node[T],
	batch Batch[T],
	mode types.BackwardMode,
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
	lossGrads, err := loss.Backward(ctx, mode, lossTensor, output, batch.Targets)
	if err != nil {
		return zero, fmt.Errorf("loss backward pass failed: %w", err)
	}

	// Model backward
	if err := g.Backward(ctx, mode, lossGrads[0]); err != nil {
		return zero, fmt.Errorf("model backward pass failed: %w", err)
	}

	return lossTensor.Data()[0], nil
}
