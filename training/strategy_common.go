package training

import (
	"context"
	"fmt"

	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/ztensor/types"
)

// computeGradientsCommon consolidates the shared logic between training strategies.
//
// acc, when non-nil, is the persistent parameter-gradient accumulator hook
// (issue #850): after Backward, arena-backed Parameter.Gradient tensors are
// accumulated into persistent buffers so a subsequent engine ResetPool (the
// Wolf per-sample reset pattern) cannot recycle the memory the optimizer and
// cross-sample accumulation will read.
func computeGradientsCommon[T tensor.Numeric](
	ctx context.Context,
	g *graph.Graph[T],
	loss graph.Node[T],
	batch Batch[T],
	mode types.BackwardMode,
	acc *gradAccumulator[T],
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

	// Migrate arena-backed parameter gradients to persistent buffers BEFORE
	// any tensor release / arena reset can recycle them (issue #850).
	if acc != nil {
		if err := acc.capture(ctx, g); err != nil {
			return zero, fmt.Errorf("persistent gradient accumulation failed: %w", err)
		}
	}

	lossVal := lossTensor.Data()[0]

	// Release intermediate tensors to free GPU memory between training steps.
	g.ClearMemo()

	return lossVal, nil
}
