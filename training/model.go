// Package training provides core components for neural network training.
package training

import (
	"context"

	"github.com/zerfoo/zerfoo/graph"
	"github.com/zerfoo/zerfoo/tensor"
)

// Model defines the interface for a trainable model.
type Model[T tensor.Numeric] interface {
	// Forward performs the forward pass of the model.
	Forward(ctx context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error)
	// Backward performs the backward pass of the model.
	Backward(ctx context.Context, grad *tensor.TensorNumeric[T], inputs ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error)
	// Parameters returns the parameters of the model.
	Parameters() []*graph.Parameter[T]
}