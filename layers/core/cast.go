// Package core provides core layer implementations for the Zerfoo ML framework.
package core

import (
	"context"
	"fmt"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/ztensor/types"
)

// Cast is a layer that converts tensor elements to a different data type.
type Cast[T tensor.Numeric] struct {
	engine      compute.Engine[T]
	outputShape []int
}

// NewCast creates a new Cast layer.
func NewCast[T tensor.Numeric](engine compute.Engine[T]) *Cast[T] {
	return &Cast[T]{
		engine: engine,
	}
}

// OutputShape returns the output shape of the Cast layer.
func (c *Cast[T]) OutputShape() []int {
	return c.outputShape
}

// Parameters returns no trainable parameters for the Cast layer.
func (c *Cast[T]) Parameters() []*graph.Parameter[T] {
	return nil
}

// Forward computes the cast operation (for same type, this is essentially a copy).
func (c *Cast[T]) Forward(_ context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if len(inputs) != 1 {
		return nil, fmt.Errorf("Cast layer requires exactly 1 input, got %d", len(inputs))
	}

	input := inputs[0]
	c.outputShape = input.Shape()

	// Create a new tensor wrapper so the graph memo has a distinct object.
	// Returning input directly causes aliased memo entries; the pool's
	// ref-count release can free the upstream tensor before downstream
	// consumers of this Cast node have run.
	//
	// For GPU storage with refcounting, use View to increment the refcount
	// so the storage survives until both tensors are freed.
	storage := input.GetStorage()
	if gs, ok := storage.(*tensor.GPUStorage[T]); ok {
		storage = gs.View(gs.Len())
	}
	return tensor.NewWithStorage[T](input.Shape(), storage)
}

// Backward computes the gradients for the Cast layer.
func (c *Cast[T]) Backward(_ context.Context, mode types.BackwardMode, outputGradient *tensor.TensorNumeric[T], inputs ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	if len(inputs) != 1 {
		return nil, fmt.Errorf("Cast layer requires exactly 1 input, got %d", len(inputs))
	}

	// For same-type casting, gradient passes through unchanged
	return []*tensor.TensorNumeric[T]{outputGradient}, nil
}

// OpType returns the operation type of the Cast layer.
func (c *Cast[T]) OpType() string {
	return "Cast"
}

// Attributes returns nil for the Cast layer.
func (c *Cast[T]) Attributes() map[string]interface{} {
	return nil
}

// Statically assert that the type implements the graph.Node interface.
var _ graph.Node[float32] = (*Cast[float32])(nil)
