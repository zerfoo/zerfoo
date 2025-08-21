// Package core provides core layer implementations for the Zerfoo ML framework.
package core

import (
	"context"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/graph"
	"github.com/zerfoo/zerfoo/tensor"
)

// Unsqueeze is a layer that adds dimensions of size 1 to a tensor at specified axes.
type Unsqueeze[T tensor.Numeric] struct {
	engine      compute.Engine[T]
	axes        []int
	outputShape []int
}

// NewUnsqueeze creates a new Unsqueeze layer.
func NewUnsqueeze[T tensor.Numeric](engine compute.Engine[T], axes []int) *Unsqueeze[T] {
	return &Unsqueeze[T]{
		engine: engine,
		axes:   axes,
	}
}

// OutputShape returns the output shape of the Unsqueeze layer.
func (u *Unsqueeze[T]) OutputShape() []int {
	return u.outputShape
}

// Parameters returns no trainable parameters for the Unsqueeze layer.
func (u *Unsqueeze[T]) Parameters() []*graph.Parameter[T] {
	return nil
}

// Forward computes the unsqueeze operation by reshaping the input tensor.
func (u *Unsqueeze[T]) Forward(ctx context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if len(inputs) != 1 {
		panic("Unsqueeze layer requires exactly 1 input")
	}
	
	input := inputs[0]
	inputShape := input.Shape()
	
	// Calculate the output shape by inserting 1s at the specified axes
	outputShape := make([]int, len(inputShape)+len(u.axes))
	
	// Normalize negative axes and create a map of axes to insert 1s
	axesMap := make(map[int]bool)
	for _, axis := range u.axes {
		normalizedAxis := axis
		if axis < 0 {
			normalizedAxis = len(outputShape) + axis
		}
		axesMap[normalizedAxis] = true
	}
	
	inputIdx := 0
	for i := 0; i < len(outputShape); i++ {
		if axesMap[i] {
			outputShape[i] = 1
		} else {
			outputShape[i] = inputShape[inputIdx]
			inputIdx++
		}
	}
	
	u.outputShape = outputShape
	
	// Reshape the tensor to the new shape
	return u.engine.Reshape(ctx, input, outputShape)
}

// Backward computes the gradients for the Unsqueeze layer.
func (u *Unsqueeze[T]) Backward(ctx context.Context, outputGradient *tensor.TensorNumeric[T], inputs ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	if len(inputs) != 1 {
		panic("Unsqueeze layer requires exactly 1 input")
	}
	
	input := inputs[0]
	inputShape := input.Shape()
	
	// The gradient just needs to be reshaped back to the input shape
	gradInput, err := u.engine.Reshape(ctx, outputGradient, inputShape)
	if err != nil {
		return nil, err
	}
	
	return []*tensor.TensorNumeric[T]{gradInput}, nil
}
