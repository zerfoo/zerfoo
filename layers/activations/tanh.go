package activations

import (
	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
)

// Tanh implements the hyperbolic tangent activation function.
type Tanh[T tensor.Numeric] struct {
	*BaseActivation[T]
}

// NewTanh creates a new Tanh activation function.
func NewTanh[T tensor.Numeric](engine compute.Engine[T], ops numeric.Arithmetic[T]) *BaseActivation[T] {
	return NewBaseActivation(engine, ops, "Tanh", WithForwardOp(ops.Tanh), WithBackwardOp(ops.TanhGrad))
}
