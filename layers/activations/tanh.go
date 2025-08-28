package activations

import (
	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/graph"
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

// BuildTanh constructs a Tanh activation node from attributes.
func BuildTanh[T tensor.Numeric](
	engine compute.Engine[T],
	ops numeric.Arithmetic[T],
	_ string,
	_ map[string]*graph.Parameter[T],
	_ map[string]interface{},
) (graph.Node[T], error) {
	return NewTanh[T](engine, ops), nil
}

// Statically assert that the type implements the graph.Node interface.
var _ graph.Node[float32] = (*Tanh[float32])(nil)
