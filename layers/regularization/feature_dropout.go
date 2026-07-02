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

// FeatureDropout implements feature-level (column-wise) inverted dropout.
// During training, entire feature columns are zeroed with probability rate,
// and surviving columns are scaled by 1/(1-rate).
// During evaluation the input is returned unchanged.
type FeatureDropout[T tensor.Float] struct {
	graph.NoParameters[T]

	engine   compute.Engine[T]
	ops      numeric.Arithmetic[T]
	rate     T
	training bool

	// rng, when non-nil, is a seeded source used to draw the per-feature mask
	// so that training is reproducible. When nil (the default) the
	// package-global unseeded math/rand/v2 source is used, preserving the
	// historical non-deterministic behavior. See WithFeatureDropoutSeed /
	// WithFeatureDropoutSource.
	rng *rand.Rand

	// Cache for backward pass. The mask is registered with the
	// save-for-backward contract (ztensor ADR 006) every training-mode
	// Forward: it cannot be recomputed (it is random), so arena-backed
	// storage must stay pinned until Backward consumes it.
	mask        *tensor.TensorNumeric[T]
	outputShape []int
	saver       graph.Saver[T] // wired by graph Builder (graph.SaverAware); nil outside a Graph
}

// FeatureDropoutOption configures optional behavior of a FeatureDropout layer.
type FeatureDropoutOption[T tensor.Float] func(*FeatureDropout[T])

// WithFeatureDropoutSeed returns a FeatureDropoutOption that makes the
// per-feature dropout mask reproducible by drawing it from a deterministic
// source seeded with the given value. Without this option (the default) masks
// are drawn from the unseeded package-global source and are not reproducible.
func WithFeatureDropoutSeed[T tensor.Float](seed uint64) FeatureDropoutOption[T] {
	return func(d *FeatureDropout[T]) {
		d.rng = rand.New(rand.NewPCG(seed, seed^0x9E3779B97F4A7C15)) //#nosec G404
	}
}

// WithFeatureDropoutSource returns a FeatureDropoutOption that draws the mask
// from the provided source, allowing callers to supply their own seeded or
// shared rand.Source for reproducible training. A nil source is ignored,
// leaving the default unseeded behavior in place.
func WithFeatureDropoutSource[T tensor.Float](src rand.Source) FeatureDropoutOption[T] {
	return func(d *FeatureDropout[T]) {
		if src != nil {
			d.rng = rand.New(src) //#nosec G404
		}
	}
}

// SetSaver implements graph.SaverAware.
func (d *FeatureDropout[T]) SetSaver(sv graph.Saver[T]) {
	d.saver = sv
}

// NewFeatureDropout creates a new FeatureDropout layer with the given drop rate.
// The rate must be in [0, 1). A rate of 0 disables dropout entirely.
//
// By default the per-feature mask is drawn from the unseeded package-global
// math/rand/v2 source, so masks are not reproducible across runs. Pass
// WithFeatureDropoutSeed or WithFeatureDropoutSource to make the mask
// deterministic.
func NewFeatureDropout[T tensor.Float](engine compute.Engine[T], ops numeric.Arithmetic[T], rate T, opts ...FeatureDropoutOption[T]) *FeatureDropout[T] {
	d := &FeatureDropout[T]{
		engine: engine,
		ops:    ops,
		rate:   rate,
	}
	for _, opt := range opts {
		opt(d)
	}

	return d
}

// randFloat64 returns the next mask draw, using the layer's seeded source when
// configured and falling back to the unseeded package-global source otherwise.
func (d *FeatureDropout[T]) randFloat64() float64 {
	if d.rng != nil {
		return d.rng.Float64()
	}

	return rand.Float64() //#nosec G404
}

// OpType returns the operation type.
func (d *FeatureDropout[T]) OpType() string {
	return "FeatureDropout"
}

// Attributes returns the non-tensor attributes of the layer.
func (d *FeatureDropout[T]) Attributes() map[string]any {
	return map[string]any{
		"rate": d.rate,
	}
}

// OutputShape returns the output shape from the most recent Forward call.
func (d *FeatureDropout[T]) OutputShape() []int {
	return d.outputShape
}

// SetTraining enables or disables training mode.
func (d *FeatureDropout[T]) SetTraining(training bool) {
	d.training = training
}

// IsTraining returns whether the layer is in training mode.
func (d *FeatureDropout[T]) IsTraining() bool {
	return d.training
}

// Forward computes the forward pass.
// In evaluation mode the input is returned unchanged.
// In training mode entire feature columns (axis=1) are independently zeroed
// with probability rate, and surviving columns are scaled by 1/(1-rate).
func (d *FeatureDropout[T]) Forward(ctx context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if len(inputs) != 1 {
		return nil, fmt.Errorf("FeatureDropout: %w, expected 1, got %d", graph.ErrInvalidInputCount, len(inputs))
	}

	input := inputs[0]
	shape := input.Shape()
	d.outputShape = shape

	if !d.training {
		d.mask = nil
		return input, nil
	}

	if len(shape) < 2 {
		return nil, fmt.Errorf("FeatureDropout: input must be at least 2D, got %dD", len(shape))
	}

	numFeatures := shape[1]
	rate64 := float64(d.rate)
	scale := d.ops.FromFloat64(1.0 / (1.0 - rate64))
	zero := d.ops.FromFloat64(0.0)

	// Generate a per-feature mask.
	featureMask := make([]T, numFeatures)
	for j := range featureMask {
		if d.randFloat64() >= rate64 {
			featureMask[j] = scale
		} else {
			featureMask[j] = zero
		}
	}

	// Broadcast the per-feature mask across the full input shape.
	size := input.Size()
	maskData := make([]T, size)
	// Elements are laid out as [..., batch, features, ...].
	// Stride for the feature dimension: product of dims after axis 1.
	featureStride := 1
	for i := 2; i < len(shape); i++ {
		featureStride *= shape[i]
	}

	for i := range maskData {
		// Determine which feature index this element belongs to.
		featureIdx := (i / featureStride) % numFeatures
		maskData[i] = featureMask[featureIdx]
	}

	mask, err := tensor.New(shape, maskData)
	if err != nil {
		return nil, fmt.Errorf("FeatureDropout: failed to create mask tensor: %w", err)
	}
	d.mask = mask
	if d.saver != nil {
		d.saver.SaveForBackward(mask)
	}

	output, err := d.engine.Mul(ctx, input, mask, nil)
	if err != nil {
		return nil, fmt.Errorf("FeatureDropout: forward multiplication failed: %w", err)
	}

	return output, nil
}

// Backward computes the backward pass.
// In evaluation mode the upstream gradient is returned unchanged.
// In training mode the upstream gradient is multiplied by the cached mask
// from the most recent Forward call.
func (d *FeatureDropout[T]) Backward(ctx context.Context, _ types.BackwardMode, dOut *tensor.TensorNumeric[T], inputs ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	if len(inputs) != 1 {
		return nil, fmt.Errorf("FeatureDropout: %w, expected 1, got %d", graph.ErrInvalidInputCount, len(inputs))
	}

	if !d.training || d.mask == nil {
		return []*tensor.TensorNumeric[T]{dOut}, nil
	}

	dInput, err := d.engine.Mul(ctx, dOut, d.mask, nil)
	if err != nil {
		return nil, fmt.Errorf("FeatureDropout: backward multiplication failed: %w", err)
	}

	return []*tensor.TensorNumeric[T]{dInput}, nil
}

// Statically assert that the type implements the graph.Node interface.
var _ graph.Node[float32] = (*FeatureDropout[float32])(nil)

// Statically assert that the type participates in the save-for-backward contract.
var _ graph.SaverAware[float32] = (*FeatureDropout[float32])(nil)
