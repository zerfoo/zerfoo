package core

import (
	"context"
	"fmt"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/graph"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
	"github.com/zerfoo/zerfoo/types"
)

// Constant represents a layer that outputs a constant tensor value.
// This is used for embedding constant tensors from ONNX TENSOR attributes.
type Constant[T tensor.Numeric] struct {
	name   string
	engine compute.Engine[T]
	ops    numeric.Arithmetic[T]
	value  *tensor.TensorNumeric[T]
}

// NewConstant creates a new Constant layer with the given tensor value.
func NewConstant[T tensor.Numeric](
	name string,
	engine compute.Engine[T],
	ops numeric.Arithmetic[T],
	value *tensor.TensorNumeric[T],
) (*Constant[T], error) {
	if name == "" {
		return nil, fmt.Errorf("constant layer name cannot be empty")
	}
	
	if value == nil {
		return nil, fmt.Errorf("constant value cannot be nil")
	}

	return &Constant[T]{
		name:   name,
		engine: engine,
		ops:    ops,
		value:  value,
	}, nil
}

// NewConstantFromData creates a new Constant layer with data and shape.
func NewConstantFromData[T tensor.Numeric](
	name string,
	engine compute.Engine[T],
	ops numeric.Arithmetic[T],
	shape []int,
	data []T,
) (*Constant[T], error) {
	if name == "" {
		return nil, fmt.Errorf("constant layer name cannot be empty")
	}
	
	value, err := tensor.New[T](shape, data)
	if err != nil {
		return nil, fmt.Errorf("failed to create constant tensor: %w", err)
	}

	return &Constant[T]{
		name:   name,
		engine: engine,
		ops:    ops,
		value:  value,
	}, nil
}

// OpType returns the operation type of the node.
func (c *Constant[T]) OpType() string {
	return "Constant"
}

// Attributes returns the node's attributes.
func (c *Constant[T]) Attributes() map[string]interface{} {
	attrs := map[string]interface{}{
		"shape": c.value.Shape(),
	}
	
	// Include tensor data type information
	switch any(c.value.Data()).(type) {
	case []float32:
		attrs["dtype"] = "float32"
	case []float64:
		attrs["dtype"] = "float64"
	case []int32:
		attrs["dtype"] = "int32"
	case []int64:
		attrs["dtype"] = "int64"
	case []uint8:
		attrs["dtype"] = "uint8"
	default:
		attrs["dtype"] = "unknown"
	}
	
	return attrs
}

// Parameters returns the trainable parameters (empty for constant layers).
func (c *Constant[T]) Parameters() []*graph.Parameter[T] {
	// Constants have no trainable parameters
	return []*graph.Parameter[T]{}
}

// OutputShape returns the shape of the output tensor.
func (c *Constant[T]) OutputShape() []int {
	return c.value.Shape()
}

// Forward performs the forward pass by returning the constant value.
// The constant value is independent of inputs.
func (c *Constant[T]) Forward(ctx context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	// Constant layers can accept 0 or more inputs (which are ignored)
	// This flexibility is useful in computational graphs where constants
	// might be connected to other nodes for graph structure reasons
	
	// Return a copy of the constant value to prevent modification
	// Create a new tensor with the same shape and data
	shape := c.value.Shape()
	data := c.value.Data()
	
	// Make a copy of the data to prevent modification
	dataCopy := make([]T, len(data))
	copy(dataCopy, data)
	
	result, err := tensor.New[T](shape, dataCopy)
	if err != nil {
		return nil, fmt.Errorf("failed to create constant output tensor: %w", err)
	}
	
	return result, nil
}

// Backward performs the backward pass for Constant.
// Constants produce zero gradients for all inputs since they don't depend on inputs.
func (c *Constant[T]) Backward(ctx context.Context, mode types.BackwardMode, outputGradient *tensor.TensorNumeric[T], inputs ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	// Constants don't depend on inputs, so gradients w.r.t. inputs are zero
	inputGradients := make([]*tensor.TensorNumeric[T], len(inputs))
	
	for i, input := range inputs {
		// Create zero gradient with same shape as input
		shape := input.Shape()
		
		// Calculate total number of elements
		size := 1
		for _, dim := range shape {
			size *= dim
		}
		
		// Create zero-filled data slice
		zeroData := make([]T, size)
		// Go initializes slice elements to zero value automatically
		
		zeros, err := tensor.New[T](shape, zeroData)
		if err != nil {
			return nil, fmt.Errorf("failed to create zero gradient for input %d: %w", i, err)
		}
		inputGradients[i] = zeros
	}
	
	return inputGradients, nil
}

// GetValue returns the constant tensor value.
// Useful for inspection and debugging.
func (c *Constant[T]) GetValue() *tensor.TensorNumeric[T] {
	return c.value
}

// String returns a string representation of the constant layer.
func (c *Constant[T]) String() string {
	shape := c.value.Shape()
	return fmt.Sprintf("Constant(%v)", shape)
}

// Name returns the name of the constant layer.
func (c *Constant[T]) Name() string {
	return c.name
}

// SetName sets the name of the constant layer.
func (c *Constant[T]) SetName(name string) {
	c.name = name
}

// NumElements returns the total number of elements in the constant tensor.
func (c *Constant[T]) NumElements() int {
	shape := c.value.Shape()
	total := 1
	for _, dim := range shape {
		total *= dim
	}
	return total
}

// IsScalar returns true if the constant is a scalar (0-dimensional tensor).
func (c *Constant[T]) IsScalar() bool {
	return len(c.value.Shape()) == 0
}

// IsVector returns true if the constant is a vector (1-dimensional tensor).
func (c *Constant[T]) IsVector() bool {
	return len(c.value.Shape()) == 1
}

// IsMatrix returns true if the constant is a matrix (2-dimensional tensor).
func (c *Constant[T]) IsMatrix() bool {
	return len(c.value.Shape()) == 2
}