// Package regularization provides regularization layers for neural networks.
package regularization

import (
	"context"
	"fmt"
	"math/rand/v2" //#nosec G404

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/ztensor/types"
)

// Dropout implements inverted dropout regularization.
// During training, each element is zeroed with probability `rate` and the surviving
// elements are scaled by 1/(1-rate) so that expected values are preserved.
// During evaluation (the default mode) the input is returned unchanged.
type Dropout[T tensor.Float] struct {
	graph.NoParameters[T]

	engine   compute.Engine[T]
	ops      numeric.Arithmetic[T]
	rate     T
	training bool

	// Cache for backward pass. The mask is registered with the
	// save-for-backward contract (ztensor ADR 006) every training-mode
	// Forward: it cannot be recomputed (it is random), so arena-backed
	// storage must stay pinned until Backward consumes it.
	mask        *tensor.TensorNumeric[T]
	outputShape []int
	saver       graph.Saver[T] // wired by graph Builder (graph.SaverAware); nil outside a Graph
}

// SetSaver implements graph.SaverAware.
func (d *Dropout[T]) SetSaver(sv graph.Saver[T]) {
	d.saver = sv
}

// NewDropout creates a new Dropout layer with the given drop rate.
// The rate must be in [0, 1). A rate of 0 disables dropout entirely.
func NewDropout[T tensor.Float](engine compute.Engine[T], ops numeric.Arithmetic[T], rate T) *Dropout[T] {
	return &Dropout[T]{
		engine: engine,
		ops:    ops,
		rate:   rate,
	}
}

// OpType returns the operation type.
func (d *Dropout[T]) OpType() string {
	return "Dropout"
}

// Attributes returns the non-tensor attributes of the layer.
func (d *Dropout[T]) Attributes() map[string]any {
	return map[string]any{
		"rate": d.rate,
	}
}

// OutputShape returns the output shape from the most recent Forward call.
func (d *Dropout[T]) OutputShape() []int {
	return d.outputShape
}

// SetTraining enables or disables training mode.
func (d *Dropout[T]) SetTraining(training bool) {
	d.training = training
}

// IsTraining returns whether the layer is in training mode.
func (d *Dropout[T]) IsTraining() bool {
	return d.training
}

// Forward computes the forward pass.
// In evaluation mode the input is returned unchanged.
// In training mode each element is independently zeroed with probability rate,
// and surviving elements are scaled by 1/(1-rate) (inverted dropout).
func (d *Dropout[T]) Forward(ctx context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if len(inputs) != 1 {
		return nil, fmt.Errorf("Dropout: %w, expected 1, got %d", graph.ErrInvalidInputCount, len(inputs))
	}

	input := inputs[0]
	d.outputShape = input.Shape()

	if !d.training {
		d.mask = nil
		return input, nil
	}

	// Build the inverted-dropout mask.
	size := input.Size()
	rate64 := float64(d.rate)
	scale := d.ops.FromFloat64(1.0 / (1.0 - rate64))
	zero := d.ops.FromFloat64(0.0)

	maskData := make([]T, size)
	for i := range maskData {
		if rand.Float64() >= rate64 { //#nosec G404
			maskData[i] = scale
		} else {
			maskData[i] = zero
		}
	}

	mask, err := tensor.New(input.Shape(), maskData)
	if err != nil {
		return nil, fmt.Errorf("Dropout: failed to create mask tensor: %w", err)
	}
	d.mask = mask
	if d.saver != nil {
		d.saver.SaveForBackward(mask)
	}

	output, err := d.engine.Mul(ctx, input, mask, nil)
	if err != nil {
		return nil, fmt.Errorf("Dropout: forward multiplication failed: %w", err)
	}

	return output, nil
}

// Backward computes the backward pass.
// In evaluation mode the upstream gradient is returned unchanged.
// In training mode the upstream gradient is multiplied by the cached mask
// from the most recent Forward call.
func (d *Dropout[T]) Backward(ctx context.Context, _ types.BackwardMode, dOut *tensor.TensorNumeric[T], inputs ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	if len(inputs) != 1 {
		return nil, fmt.Errorf("Dropout: %w, expected 1, got %d", graph.ErrInvalidInputCount, len(inputs))
	}

	if !d.training || d.mask == nil {
		return []*tensor.TensorNumeric[T]{dOut}, nil
	}

	dInput, err := d.engine.Mul(ctx, dOut, d.mask, nil)
	if err != nil {
		return nil, fmt.Errorf("Dropout: backward multiplication failed: %w", err)
	}

	return []*tensor.TensorNumeric[T]{dInput}, nil
}

// Statically assert that the type implements the graph.Node interface.
var _ graph.Node[float32] = (*Dropout[float32])(nil)

// Statically assert that the type participates in the save-for-backward contract.
var _ graph.SaverAware[float32] = (*Dropout[float32])(nil)
