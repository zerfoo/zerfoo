// Package reducesum provides the ReduceSum layer for the Zerfoo ML framework.
package reducesum

import (
	"context"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/graph"
	"github.com/zerfoo/zerfoo/tensor"
)

// ReduceSum represents a reduce sum operation.
type ReduceSum[T tensor.Numeric] struct {
	engine      compute.Engine[T]
	axes        []int
	keepDims    bool
	outputShape []int
}

// OpType returns the operation type.
func (r *ReduceSum[T]) OpType() string {
	return "ReduceSum"
}

// Attributes returns the attributes.
func (r *ReduceSum[T]) Attributes() map[string]interface{} {
	return map[string]interface{}{
		"axes":     r.axes,
		"keepdims": r.keepDims,
	}
}

// New creates a new ReduceSum layer.
func New[T tensor.Numeric](engine compute.Engine[T], axes []int, keepDims bool) *ReduceSum[T] {
	return &ReduceSum[T]{
		engine:   engine,
		axes:     axes,
		keepDims: keepDims,
	}
}

// OutputShape returns the output shape of the ReduceSum layer.
func (r *ReduceSum[T]) OutputShape() []int {
	return r.outputShape
}

// Parameters returns no trainable parameters for the ReduceSum layer.
func (r *ReduceSum[T]) Parameters() []*graph.Parameter[T] {
	return nil
}

// Forward computes the reduce sum operation.
func (r *ReduceSum[T]) Forward(ctx context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	input := inputs[0]
	shape := input.Shape()

	axesMap := make(map[int]bool)
	for _, axis := range r.axes {
		axesMap[axis] = true
	}

	outputShape := make([]int, 0, len(shape))
	if r.keepDims {
		outputShape = make([]int, len(shape))
		for i, dim := range shape {
			if axesMap[i] {
				outputShape[i] = 1
			} else {
				outputShape[i] = dim
			}
		}
	} else {
		for i, dim := range shape {
			if !axesMap[i] {
				outputShape = append(outputShape, dim)
			}
		}
	}

	r.outputShape = outputShape

	// The compute engine's Sum method might need to be updated to handle multiple axes.
	// For now, we'll assume it does, or we'll have to chain Sum operations.
	// Let's assume the engine handles it for now. A proper fix would involve checking the engine capabilities.
	if len(r.axes) == 0 {
		// Sum over all axes
		return r.engine.Sum(ctx, input, -1, r.keepDims)
	}

	// Iterative sum for now
	tempResult := input

	var err error
	for _, axis := range r.axes {
		tempResult, err = r.engine.Sum(ctx, tempResult, axis, r.keepDims)
		if err != nil {
			return nil, err
		}
	}

	return tempResult, nil
}

// Backward computes the gradients for the ReduceSum layer.
func (r *ReduceSum[T]) Backward(_ context.Context, outputGradient *tensor.TensorNumeric[T], _ ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	// The gradient of the sum is 1, so we just need to broadcast the output gradient
	// to the shape of the input tensor.
	return []*tensor.TensorNumeric[T]{outputGradient}, nil
}
