// Package core provides core layer implementations for the Zerfoo ML framework.
package core

import (
	"context"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/graph"
	"github.com/zerfoo/zerfoo/tensor"
)

// Reshape is a layer that changes the shape of a tensor without changing its data.
type Reshape[T tensor.Numeric] struct {
	engine      compute.Engine[T]
	targetShape []int
	outputShape []int
}

// NewReshape creates a new Reshape layer.
func NewReshape[T tensor.Numeric](engine compute.Engine[T], targetShape []int) *Reshape[T] {
	return &Reshape[T]{
		engine:      engine,
		targetShape: targetShape,
	}
}

// OutputShape returns the output shape of the Reshape layer.
func (r *Reshape[T]) OutputShape() []int {
	return r.outputShape
}

// Parameters returns no trainable parameters for the Reshape layer.
func (r *Reshape[T]) Parameters() []*graph.Parameter[T] {
	return nil
}

// Forward computes the reshape operation.
func (r *Reshape[T]) Forward(ctx context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if len(inputs) != 1 {
		panic("Reshape layer requires exactly 1 input")
	}
	
	input := inputs[0]
	inputShape := input.Shape()
	
	// Handle shape inference
	if len(r.targetShape) == 1 && r.targetShape[0] == -1 {
		// Identity reshape - keep the same shape
		r.outputShape = inputShape
		return input, nil
	}
	
	// Handle -1 dimension inference (this is a minimal implementation for architectural compliance)
	resolvedShape := make([]int, len(r.targetShape))
	inferIndex := -1
	totalSize := input.Size()
	knownSize := 1
	
	for i, dim := range r.targetShape {
		if dim == -1 {
			if inferIndex != -1 {
				panic("Reshape layer can have at most one -1 dimension")
			}
			inferIndex = i
			resolvedShape[i] = -1 // Will be resolved later
		} else {
			resolvedShape[i] = dim
			knownSize *= dim
		}
	}
	
	// Resolve the -1 dimension if present
	if inferIndex != -1 {
		if knownSize == 0 {
			panic("Cannot infer dimension when known size is 0")
		}
		resolvedShape[inferIndex] = totalSize / knownSize
	}
	
	r.outputShape = resolvedShape
	return r.engine.Reshape(ctx, input, resolvedShape)
}

// Backward computes the gradients for the Reshape layer.
func (r *Reshape[T]) Backward(ctx context.Context, outputGradient *tensor.TensorNumeric[T], inputs ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	if len(inputs) != 1 {
		panic("Reshape layer requires exactly 1 input")
	}
	
	input := inputs[0]
	inputShape := input.Shape()
	
	// The gradient just needs to be reshaped back to the input shape
	gradInput, err := r.engine.Reshape(ctx, outputGradient, inputShape)
	if err != nil {
		return nil, err
	}
	
	return []*tensor.TensorNumeric[T]{gradInput}, nil
}

// OpType returns the operation type of the Reshape layer.
func (r *Reshape[T]) OpType() string {
	return "Reshape"
}

// Attributes returns the attributes of the Reshape layer.
func (r *Reshape[T]) Attributes() map[string]interface{} {
	return map[string]interface{}{"shape": r.targetShape}
}
