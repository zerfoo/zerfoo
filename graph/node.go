package graph

import (
	"context"

	"github.com/zerfoo/zerfoo/tensor"
)

// Node represents a node in the computation graph.
type Node[T tensor.Numeric] interface {
	// OpType returns the operation type of the node, e.g., "ReLU", "Dense".
	OpType() string

	// Attributes returns a map of the node's non-tensor attributes.
	Attributes() map[string]interface{}

	// Forward computes the output of the node given the inputs.
	Forward(ctx context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error)
	// Backward computes the gradients of the node with respect to its inputs.
	Backward(ctx context.Context, outputGradient *tensor.TensorNumeric[T], inputs ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error)
	// Parameters returns the trainable parameters of the node.
	Parameters() []*Parameter[T]
	// OutputShape returns the shape of the output tensor.
	OutputShape() []int
}
