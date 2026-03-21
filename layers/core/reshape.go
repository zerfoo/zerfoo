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
// Accepts 1 input (static target shape from attributes) or 2 inputs
// (ONNX opset 5+: data + shape tensor). In the shape tensor, -1 means
// infer and 0 means copy from the corresponding input dimension.
func (r *Reshape[T]) Forward(ctx context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if len(inputs) == 0 || len(inputs) > 2 {
		return nil, fmt.Errorf("Reshape requires 1 or 2 inputs, got %d", len(inputs))
	}

	input := inputs[0]
	targetShape := r.targetShape

	if len(inputs) == 2 {
		// Dynamic shape from second input tensor.
		shapeData := inputs[1].Data()
		targetShape = make([]int, len(shapeData))
		for i, v := range shapeData {
			targetShape[i] = int(v)
		}
	}

	// ONNX semantics: 0 means copy from input shape at that position.
	inputShape := input.Shape()
	resolved := make([]int, len(targetShape))
	copy(resolved, targetShape)
	for i, d := range resolved {
		if d == 0 && i < len(inputShape) {
			resolved[i] = inputShape[i]
		}
	}

	r.outputShape = resolved
	return r.engine.Reshape(ctx, input, resolved)
}

// Backward computes the gradients for the Reshape layer.
func (r *Reshape[T]) Backward(ctx context.Context, mode types.BackwardMode, outputGradient *tensor.TensorNumeric[T], inputs ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	if len(inputs) != 1 {
		return nil, fmt.Errorf("Reshape layer requires exactly 1 input for backward, got %d", len(inputs))
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

// Statically assert that the type implements the graph.Node interface.
var _ graph.Node[float32] = (*Reshape[float32])(nil)
