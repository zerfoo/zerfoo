package optimizer

import (
	"context"

	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/tensor"
)

// Optimizer defines the interface for optimization algorithms.
type Optimizer[T tensor.Numeric] interface {
	Step(ctx context.Context, params []*graph.Parameter[T]) error
}
