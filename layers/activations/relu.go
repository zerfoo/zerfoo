package activations

import (
	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
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
