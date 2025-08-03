package optimizer

import (
	"github.com/zerfoo/zerfoo/graph"
	"github.com/zerfoo/zerfoo/tensor"
)

// Optimizer defines the interface for optimization algorithms.
type Optimizer[T tensor.Numeric] interface {
	Step(params []*graph.Parameter[T])
	Clip(params []*graph.Parameter[T], threshold float32)
}
