package activations

import (
	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/graph"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
)

// ReLU implements the Rectified Linear Unit activation function.
type ReLU[T tensor.Numeric] struct {
	*BaseActivation[T]
}

// NewReLU creates a new ReLU activation function.
func NewReLU[T tensor.Numeric](engine compute.Engine[T], ops numeric.Arithmetic[T]) *BaseActivation[T] {
	return NewBaseActivation(engine, ops, "ReLU", WithForwardOp(ops.ReLU), WithBackwardOp(ops.ReLUGrad))
}

// Statically assert that the type implements the graph.Node interface.
var _ graph.Node[float32] = (*ReLU[float32])(nil)
