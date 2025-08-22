package activations

import (
	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
)

// Sigmoid implements the sigmoid activation function.
type Sigmoid[T tensor.Numeric] struct {
	*BaseActivation[T]
}

// NewSigmoid creates a new Sigmoid activation function.
func NewSigmoid[T tensor.Numeric](engine compute.Engine[T], ops numeric.Arithmetic[T]) *BaseActivation[T] {
	return NewBaseActivation(engine, ops, "Sigmoid", WithForwardOp(ops.Sigmoid), WithBackwardOp(ops.SigmoidGrad))
}
