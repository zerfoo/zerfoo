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
// Backward (T124.2.2). Since the ADR 006 (T2.3) migration the node's
// Backward recomputes everything from the live input it receives, so no
// forward-seeding of a cache is needed (or possible).
func GELUBackward[T tensor.Float](ctx context.Context, engine compute.Engine[T], ops numeric.Arithmetic[T],
	dOutput, input *tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	node := activations.NewGelu(engine, ops)
	grads, err := node.Backward(ctx, types.FullBackprop, dOutput, input)
	if err != nil {
		return nil, err
	}
	if len(grads) != 1 {
		return nil, fmt.Errorf("GELUBackward: expected 1 gradient, got %d", len(grads))
	}
	return grads[0], nil
}
