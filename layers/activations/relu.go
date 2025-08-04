package activations

import (
	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
)

// ReLU implements the Rectified Linear Unit activation function.
type ReLU[T tensor.Numeric] struct {
	*BaseActivation[T]
}

// NewReLU creates a new ReLU activation function.
func NewReLU[T tensor.Numeric](engine compute.Engine[T], ops numeric.Arithmetic[T]) *ReLU[T] {
	return &ReLU[T]{
		BaseActivation: NewBaseActivation(engine, ops, ops.ReLU, ops.ReLUGrad),
	}
}
