package training

import (
	"github.com/zerfoo/zerfoo/graph"
	"github.com/zerfoo/zerfoo/tensor"
)

// Model defines the interface for a trainable model.
type Model[T tensor.Numeric] interface {
	// Forward performs the forward pass of the model.
	Forward(inputs ...*tensor.Tensor[T]) *tensor.Tensor[T]
	// Backward performs the backward pass of the model.
	Backward(grad *tensor.Tensor[T]) []*tensor.Tensor[T]
	// Parameters returns the parameters of the model.
	Parameters() []*graph.Parameter[T]
}
