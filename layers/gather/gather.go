// Package gather provides the Gather layer for the Zerfoo ML framework.
package gather

import (
	"context"
	"fmt"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/graph"
	"github.com/zerfoo/zerfoo/tensor"
)

// Gather is a layer that gathers slices from a tensor.
type Gather[T tensor.Numeric] struct {
	engine      compute.Engine[T]
	outputShape []int
	weights     *tensor.TensorNumeric[T] // Optional embedded weights
}

// New creates a new Gather layer.
func New[T tensor.Numeric](engine compute.Engine[T]) *Gather[T] {
	return &Gather[T]{
		engine: engine,
	}
}

// NewWithWeights creates a new Gather layer with embedded weights.
func NewWithWeights[T tensor.Numeric](engine compute.Engine[T], weights *tensor.TensorNumeric[T]) *Gather[T] {
	return &Gather[T]{
		engine:  engine,
		weights: weights,
	}
}

// OutputShape returns the output shape of the Gather layer.
func (g *Gather[T]) OutputShape() []int {
	return g.outputShape
}

// Parameters returns no trainable parameters for the Gather layer.
func (g *Gather[T]) Parameters() []*graph.Parameter[T] {
	return nil
}

// HasEmbeddedWeights returns true if this Gather layer has embedded weights
func (g *Gather[T]) HasEmbeddedWeights() bool {
	return g.weights != nil
}

// Forward computes the gather operation.
func (g *Gather[T]) Forward(ctx context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	var params *tensor.TensorNumeric[T]
	var indices *tensor.TensorNumeric[int]
	
	// If we have embedded weights, use them as params and expect only indices as input
	if g.weights != nil {
		if len(inputs) != 1 {
			return nil, fmt.Errorf("Gather layer with embedded weights expects 1 input (indices), got %d", len(inputs))
		}
		params = g.weights
		
		// Create int tensor with same shape as input
		inputTensor := inputs[0]
		fmt.Printf("DEBUG: Gather Forward - input shape: %v, weights shape: %v\n", inputTensor.Shape(), g.weights.Shape())
		
		intTensor, err := tensor.New[int](inputTensor.Shape(), nil)
		if err != nil {
			return nil, fmt.Errorf("failed to create int tensor: %w", err)
		}
		
		// Convert input values to int (assuming they represent valid indices)
		// For now, we'll access the underlying data directly since we need to convert types
		// This is a simplified approach - in a production system, we'd want better type handling
		inputData := inputTensor.Data()
		intData := make([]int, len(inputData))
		for i, val := range inputData {
			// Convert to int (assuming T represents valid indices)
			intData[i] = int(float64(val))
		}
		intTensor.SetData(intData)
		
		// Ensure indices tensor is 2D as required by the engine
		if len(intTensor.Shape()) == 1 {
			// Reshape 1D tensor to 2D: [N] -> [1, N]
			newShape := []int{1, intTensor.Shape()[0]}
			reshapedTensor, err := tensor.New[int](newShape, intData)
			if err != nil {
				return nil, fmt.Errorf("failed to reshape indices tensor: %w", err)
			}
			indices = reshapedTensor
			fmt.Printf("DEBUG: Reshaped indices from %v to %v\n", intTensor.Shape(), newShape)
		} else {
			indices = intTensor
		}
	} else {
		// Original behavior: expect params and indices as inputs
		if len(inputs) != 2 {
			return nil, fmt.Errorf("Gather layer expects 2 inputs (params, indices), got %d", len(inputs))
		}
		params = inputs[0]
		var ok bool
		indices, ok = any(inputs[1]).(*tensor.TensorNumeric[int])
		if !ok {
			return nil, fmt.Errorf("Gather layer expects indices to be of type *tensor.TensorNumeric[int], got %T", inputs[1])
		}
	}

	// The output shape is the shape of the indices tensor, with the last dimension
	// replaced by the shape of the params tensor after the first dimension.
	outputShape := indices.Shape()
	outputShape = append(outputShape, params.Shape()[1:]...)
	g.outputShape = outputShape

	output, err := tensor.New[T](outputShape, nil)
	if err != nil {
		return nil, err
	}

	return output, g.engine.Gather(ctx, params, indices, output)
}

// Backward computes the gradients for the Gather layer.
func (g *Gather[T]) Backward(ctx context.Context, outputGradient *tensor.TensorNumeric[T], inputs ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	// The Gather layer has no trainable parameters, so the gradient is passed
	// through to the params tensor.
	params := inputs[0]
	indices, ok := any(inputs[1]).(*tensor.TensorNumeric[int])
	if !ok {
		return nil, fmt.Errorf("Gather layer expects indices to be of type *tensor.TensorNumeric[int], got %T", inputs[1])
	}

	dParams, err := tensor.New[T](params.Shape(), nil)
	if err != nil {
		return nil, err
	}

	if err := g.engine.ScatterAdd(ctx, dParams, indices, outputGradient); err != nil {
		return nil, err
	}

	return []*tensor.TensorNumeric[T]{dParams, nil}, nil
}

// OpType returns the operation type of the Gather layer.
func (g *Gather[T]) OpType() string {
	return "Gather"
}

// Attributes returns nil for the Gather layer.
func (g *Gather[T]) Attributes() map[string]interface{} {
	return nil
}
