package functional

import (
	"context"
	"fmt"

	"github.com/zerfoo/zerfoo/layers/activations"
	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/ztensor/types"
)

// GELUBackward computes the gradient of the GELU activation.
// dOutput: gradient from upstream
// input: original input to GELU
// Returns: dInput (same shape as input)
//
// Thin wrapper that delegates to the canonical activations.Gelu Node's
// Backward (T124.2.2). The node's Backward consumes the lastInput cached
// during Forward, so we run Forward(input) first to seed it, then call
// Backward(dOutput).
func GELUBackward[T tensor.Float](ctx context.Context, engine compute.Engine[T], ops numeric.Arithmetic[T],
	dOutput, input *tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	node := activations.NewGelu(engine, ops)
	if _, err := node.Forward(ctx, input); err != nil {
		return nil, fmt.Errorf("GELUBackward: forward seed: %w", err)
	}
	grads, err := node.Backward(ctx, types.FullBackprop, dOutput)
	if err != nil {
		return nil, err
	}
	if len(grads) != 1 {
		return nil, fmt.Errorf("GELUBackward: expected 1 gradient, got %d", len(grads))
	}
	return grads[0], nil
}
