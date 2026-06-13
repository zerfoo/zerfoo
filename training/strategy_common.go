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
	lossTensor, err := computeGradientsTensorCommon(ctx, g, loss, batch, mode, acc)
	if err != nil {
		return zero, err
	}
	return lossTensor.Data()[0], nil
}

// computeGradientsTensorCommon is computeGradientsCommon without the final
// host readback of the loss value: it returns the loss TENSOR produced by
// the loss node instead of loss.Data()[0]. On a GPU engine the readback is
// a device-to-host copy, which is illegal inside a CUDA-graph capture
// region; CaptureReplayRunner uses this variant during capture and defers
// the read until after the captured graph has actually executed.
func computeGradientsTensorCommon[T tensor.Numeric](
	ctx context.Context,
	g *graph.Graph[T],
	loss graph.Node[T],
	batch Batch[T],
	mode types.BackwardMode,
	acc *gradAccumulator[T],
) (*tensor.TensorNumeric[T], error) {
	// Materialize inputs in graph input order
	var inputSlice []*tensor.TensorNumeric[T]
	for _, inputNode := range g.Inputs() {
		inputSlice = append(inputSlice, batch.Inputs[inputNode])
	}

	// Forward pass
	output, err := g.Forward(ctx, inputSlice...)
	if err != nil {
		return nil, fmt.Errorf("forward pass failed: %w", err)
	}

	// Loss forward
	lossTensor, err := loss.Forward(ctx, output, batch.Targets)
	if err != nil {
		return nil, fmt.Errorf("loss computation failed: %w", err)
	}

	// Loss backward
	lossGrads, err := loss.Backward(ctx, mode, lossTensor, output, batch.Targets)
	if err != nil {
		return nil, fmt.Errorf("loss backward pass failed: %w", err)
	}

	// Model backward
	if err := g.Backward(ctx, mode, lossGrads[0]); err != nil {
		return nil, fmt.Errorf("model backward pass failed: %w", err)
	}

	// Migrate arena-backed parameter gradients to persistent buffers BEFORE
	// any tensor release / arena reset can recycle them (issue #850).
	if acc != nil {
		if err := acc.capture(ctx, g); err != nil {
			return nil, fmt.Errorf("persistent gradient accumulation failed: %w", err)
		}
	}

	// Release intermediate tensors to free GPU memory between training steps.
	g.ClearMemo()

	return lossTensor, nil
}
