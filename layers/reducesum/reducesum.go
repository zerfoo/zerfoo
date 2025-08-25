// Package reducesum provides the ReduceSum layer for the Zerfoo ML framework.
package reducesum

import (
	"context"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/graph"
	"github.com/zerfoo/zerfoo/tensor"
	"github.com/zerfoo/zerfoo/types"
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
func (r *ReduceSum[T]) Backward(ctx context.Context, _ types.BackwardMode, outputGradient *tensor.TensorNumeric[T], inputs ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	// Broadcast the output gradient back to the input shape.
	// If keepDims is true, outputGradient already has 1s in reduced axes; just repeat along those axes.
	// If keepDims is false, first reshape to re-insert 1s at reduced axes, then repeat.
	if len(inputs) != 1 {
		panic("ReduceSum layer requires exactly 1 input for backward")
	}

	input := inputs[0]
	inputShape := input.Shape()

	grad := outputGradient

	// Build a map for quick axis lookup
	axesMap := make(map[int]bool)
	for _, ax := range r.axes {
		axesMap[ax] = true
	}

	if !r.keepDims {
		// Re-insert singleton dimensions at reduced axes positions
		reshaped := make([]int, len(inputShape))
		outShape := grad.Shape()
		outIdx := 0
		for i := 0; i < len(reshaped); i++ {
			if axesMap[i] {
				reshaped[i] = 1
			} else {
				reshaped[i] = outShape[outIdx]
				outIdx++
			}
		}
		var err error
		grad, err = r.engine.Reshape(ctx, grad, reshaped)
		if err != nil {
			return nil, err
		}
	}

	// Now repeat along each reduced axis to match the input shape
	var err error
	for _, ax := range r.axes {
		grad, err = r.engine.Repeat(ctx, grad, ax, inputShape[ax])
		if err != nil {
			return nil, err
		}
	}

	return []*tensor.TensorNumeric[T]{grad}, nil
}

// Statically assert that the type implements the graph.Node interface.
var _ graph.Node[float32] = (*ReduceSum[float32])(nil)
