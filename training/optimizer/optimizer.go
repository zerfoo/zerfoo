package optimizer

import (
	"context"

	"github.com/zerfoo/zerfoo/graph"
	"github.com/zerfoo/zerfoo/tensor"
)

// Optimizer defines the interface for optimization algorithms.
type Optimizer[T tensor.Numeric] interface {
	Step(ctx context.Context, params []*graph.Parameter[T]) error
	Clip(ctx context.Context, params []*graph.Parameter[T], threshold float32)
}