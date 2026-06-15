package training

import (
	"context"
	"fmt"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/ztensor/types"
)

// onesLike builds a host-backed tensor of ones with the same shape as ref,
// used as the d(loss)/d(loss) = 1 seed for loss.Backward (issue #872).
//
// The element value 1 is produced via the type's arithmetic ops so it is
// correct for every supported numeric type (float32/64, minifloats, ints).
// When the graph carries an engine (the production path) its ops are used;
// some graphs are built with a nil engine (e.g. parameter-fixture tests), in
// which case onesValue derives 1 from the type directly so the seed never
// depends on an engine being wired in.
//
// The seed is deliberately a plain host tensor: the loss layers consume it
// only through engine.Mul(localGrad, dOut), which already accepts the
// host-backed scalar tensor that loss.Forward produces, so this preserves the
// existing storage contract while flipping the value from L to 1.
func onesLike[T tensor.Numeric](engine compute.Engine[T], ref *tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	shape := ref.Shape()
	n := 1
	for _, d := range shape {
		n *= d
	}
	one, err := onesValue[T](engine)
	if err != nil {
		return nil, err
	}
	data := make([]T, n)
	for i := range data {
		data[i] = one
	}
	return tensor.New[T](shape, data)
}

// onesValue returns the value 1 of type T. It prefers the engine's arithmetic
// ops (the only general source of 1 for minifloat types) and falls back to a
// per-type literal for the built-in numeric kinds when no engine is available.
func onesValue[T tensor.Numeric](engine compute.Engine[T]) (T, error) {
	if engine != nil {
		return engine.Ops().FromFloat64(1.0), nil
	}
	var zero T
	switch any(zero).(type) {
	case float32:
		return any(float32(1)).(T), nil
	case float64:
		return any(float64(1)).(T), nil
	case int:
		return any(int(1)).(T), nil
	case int8:
		return any(int8(1)).(T), nil
	case int16:
		return any(int16(1)).(T), nil
	case int32:
		return any(int32(1)).(T), nil
	case int64:
		return any(int64(1)).(T), nil
	case uint:
		return any(uint(1)).(T), nil
	case uint8:
		return any(uint8(1)).(T), nil
	case uint32:
		return any(uint32(1)).(T), nil
	case uint64:
		return any(uint64(1)).(T), nil
	default:
		return zero, fmt.Errorf("training: cannot seed loss backward for type %T without an engine; build the graph with an engine so its ops can produce a unit seed", zero)
	}
}

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

	// Loss backward.
	//
	// The upstream-gradient SEED handed to loss.Backward is d(loss)/d(loss) = 1,
	// NOT the loss value. Loss layers (cross-entropy, MSE, BCE, ...) end their
	// Backward with grad = local_grad * dOut (engine.Mul broadcasts dOut over
	// the local gradient). Seeding dOut with the scalar loss value L would scale
	// every model gradient by L, i.e. compute L * dL/dparams = the gradient of
	// (1/2)L^2 -- a different objective with a loss-dependent effective learning
	// rate (issue #872). A ones tensor matching the loss shape ([1] for the
	// scalar loss) broadcasts correctly through that final Mul.
	//
	// The seed is DEVICE-RESIDENT and cached on the strategy's accumulator,
	// built once (during an eager warmup step, outside any CUDA-graph capture
	// region) and reused every step (issue #875). The pre-#875 per-call
	// onesLike built a HOST tensor that the engine host->device cudaMemcpy'd on
	// every step; inside CaptureReplayRunner's capture region that host copy is
	// illegal ("operation not permitted when stream is capturing") and crashed
	// capture-on training. Reusing the cached device seed enqueues no host copy.
	//
	// When acc is nil (a few unit tests bypass the strategy accumulator) there
	// is nowhere to cache, so fall back to the per-call host seed: those paths
	// never run inside a capture region.
	var ones *tensor.TensorNumeric[T]
	if acc != nil {
		ones, err = acc.seedFor(g.Engine(), lossTensor)
	} else {
		ones, err = onesLike[T](g.Engine(), lossTensor)
	}
	if err != nil {
		return nil, fmt.Errorf("seeding loss backward failed: %w", err)
	}
	lossGrads, err := loss.Backward(ctx, mode, ones, output, batch.Targets)
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
