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

	// rng, when non-nil, is a seeded source used to draw the dropout mask so
	// that training is reproducible. When nil (the default) the package-global
	// unseeded math/rand/v2 source is used, preserving the historical
	// non-deterministic behavior. See WithDropoutSeed / WithDropoutSource.
	rng *rand.Rand

	// useEngineOp routes the mask through the engine's capture-safe Dropout op
	// (compute.Dropouter) instead of generating a host-side mask. The op derives
	// the mask deterministically from a per-step counter-based seed (Philox) and
	// regenerates it in Backward, so no mask tensor is pinned — this is what
	// makes dropout eligible for CUDA-graph capture on the GPU. Enable with
	// WithEngineDropout; falls back to the host path if the engine does not
	// implement compute.Dropouter. See WithEngineDropout.
	useEngineOp bool
	seedSet     bool   // a base seed was supplied explicitly (WithDropoutSeed)
	baseSeed    uint64 // base seed for the engine-op path
	counter     uint64 // advances each training Forward so masks vary across steps
	lastSeed    uint64 // seed used by the most recent training Forward, for Backward

	// Cache for backward pass. The mask is registered with the
	// save-for-backward contract (ztensor ADR 006) every training-mode
	// Forward: it cannot be recomputed (it is random), so arena-backed
	// storage must stay pinned until Backward consumes it.
	mask        *tensor.TensorNumeric[T]
	outputShape []int
	saver       graph.Saver[T] // wired by graph Builder (graph.SaverAware); nil outside a Graph
}

// DropoutOption configures optional behavior of a Dropout layer.
type DropoutOption[T tensor.Float] func(*Dropout[T])

// WithDropoutSeed returns a DropoutOption that makes the dropout mask
// reproducible by drawing it from a deterministic source seeded with the
// given value. Two layers constructed with the same seed produce identical
// masks for identical inputs and call sequences. Without this option (the
// default) masks are drawn from the unseeded package-global source and are
// not reproducible.
func WithDropoutSeed[T tensor.Float](seed uint64) DropoutOption[T] {
	return func(d *Dropout[T]) {
		d.rng = rand.New(rand.NewPCG(seed, seed^0x9E3779B97F4A7C15)) //#nosec G404
		d.baseSeed = seed
		d.seedSet = true
	}
}

// WithEngineDropout returns a DropoutOption that routes the dropout mask through
// the engine's capture-safe Dropout op (compute.Dropouter) instead of generating
// the mask on the host. The op derives the mask deterministically on-device from
// a per-step counter-based seed (Philox) and regenerates it in Backward, so no
// mask tensor is pinned for the backward pass — this is what makes dropout
// eligible for CUDA-graph capture on the GPU, where a host-generated random mask
// is capture-ineligible.
//
// If the engine does not implement compute.Dropouter, the layer transparently
// falls back to the host mask path. Combine with WithDropoutSeed to make the
// engine-op masks reproducible across runs; without it a distinct base seed is
// drawn once at construction so multiple layers decorrelate.
func WithEngineDropout[T tensor.Float]() DropoutOption[T] {
	return func(d *Dropout[T]) {
		d.useEngineOp = true
	}
}

// WithDropoutSource returns a DropoutOption that draws the dropout mask from
// the provided source, allowing callers to supply their own seeded or shared
// rand.Source for reproducible training. A nil source is ignored, leaving the
// default unseeded behavior in place.
func WithDropoutSource[T tensor.Float](src rand.Source) DropoutOption[T] {
	return func(d *Dropout[T]) {
		if src != nil {
			d.rng = rand.New(src) //#nosec G404
		}
	}
}

// SetSaver implements graph.SaverAware.
func (d *Dropout[T]) SetSaver(sv graph.Saver[T]) {
	d.saver = sv
}

// NewDropout creates a new Dropout layer with the given drop rate.
// The rate must be in [0, 1). A rate of 0 disables dropout entirely.
//
// By default the dropout mask is drawn from the unseeded package-global
// math/rand/v2 source, so masks are not reproducible across runs. Pass
// WithDropoutSeed or WithDropoutSource to make the mask deterministic.
func NewDropout[T tensor.Float](engine compute.Engine[T], ops numeric.Arithmetic[T], rate T, opts ...DropoutOption[T]) *Dropout[T] {
	d := &Dropout[T]{
		engine: engine,
		ops:    ops,
		rate:   rate,
	}
	for _, opt := range opts {
		opt(d)
	}

	// For the engine-op path, derive a base seed once when none was supplied so
	// that multiple dropout layers decorrelate and masks differ across runs.
	// A seeded source (WithDropoutSource) is reused for reproducibility.
	if d.useEngineOp && !d.seedSet {
		if d.rng != nil {
			d.baseSeed = d.rng.Uint64()
		} else {
			d.baseSeed = rand.Uint64() //#nosec G404
		}
	}

	return d
}

// randFloat64 returns the next mask draw, using the layer's seeded source when
// configured and falling back to the unseeded package-global source otherwise.
func (d *Dropout[T]) randFloat64() float64 {
	if d.rng != nil {
		return d.rng.Float64()
	}

	return rand.Float64() //#nosec G404
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

	rate64 := float64(d.rate)

	// Capture-safe path: delegate to the engine's Dropout op when requested and
	// supported. The op generates the mask on-device from a counter-based seed
	// and regenerates it in Backward, so nothing is pinned for the backward pass.
	if d.useEngineOp {
		if dr, ok := d.engine.(compute.Dropouter[T]); ok {
			d.mask = nil
			if rate64 <= 0 {
				return input, nil
			}
			d.lastSeed = d.baseSeed + d.counter
			d.counter++
			output, err := dr.Dropout(ctx, input, rate64, d.lastSeed, true)
			if err != nil {
				return nil, fmt.Errorf("Dropout: engine dropout failed: %w", err)
			}
			return output, nil
		}
	}

	// Build the inverted-dropout mask.
	size := input.Size()
	scale := d.ops.FromFloat64(1.0 / (1.0 - rate64))
	zero := d.ops.FromFloat64(0.0)

	maskData := make([]T, size)
	for i := range maskData {
		if d.randFloat64() >= rate64 {
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

	if !d.training {
		return []*tensor.TensorNumeric[T]{dOut}, nil
	}

	// Capture-safe path: regenerate the mask on-device from the Forward seed.
	if d.useEngineOp {
		if dr, ok := d.engine.(compute.Dropouter[T]); ok {
			rate64 := float64(d.rate)
			if rate64 <= 0 {
				return []*tensor.TensorNumeric[T]{dOut}, nil
			}
			dInput, err := dr.DropoutBackward(ctx, dOut, rate64, d.lastSeed, true)
			if err != nil {
				return nil, fmt.Errorf("Dropout: engine dropout backward failed: %w", err)
			}

			return []*tensor.TensorNumeric[T]{dInput}, nil
		}
	}

	if d.mask == nil {
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
