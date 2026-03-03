package activations

import (
	"math"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/graph"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
)

// NewErf creates an Erf activation layer using the standard error function.
// erf(x) = (2/sqrt(pi)) * integral_0^x exp(-t^2) dt
func NewErf[T tensor.Float](engine compute.Engine[T], ops numeric.Arithmetic[T]) *BaseActivation[T] {
	forwardOp := func(x T) T { return T(math.Erf(float64(x))) }
	backwardOp := func(x T) T {
		return T(2.0 / math.Sqrt(math.Pi) * math.Exp(-float64(x)*float64(x)))
	}
	return NewBaseActivation(engine, ops, "Erf",
		WithForwardOp(forwardOp),
		WithBackwardOp(backwardOp))
}

// BuildErf constructs an Erf activation layer for the registry.
func BuildErf[T tensor.Float](
	engine compute.Engine[T],
	ops numeric.Arithmetic[T],
	_ string,
	_ map[string]*graph.Parameter[T],
	_ map[string]interface{},
) (graph.Node[T], error) {
	return NewErf(engine, ops), nil
}
