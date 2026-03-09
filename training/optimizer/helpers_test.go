package optimizer

import (
	"context"

	"github.com/zerfoo/zerfoo/graph"
	"github.com/zerfoo/zerfoo/tensor"
)

// noopOptimizer is a test helper that does nothing on Step.
type noopOptimizer[T tensor.Numeric] struct{}

func (n *noopOptimizer[T]) Step(_ context.Context, _ []*graph.Parameter[T]) error {
	return nil
}

// setOptimizer is a test helper that sets all parameter values to a fixed value.
type setOptimizer[T tensor.Numeric] struct {
	value T
}

func (s *setOptimizer[T]) Step(_ context.Context, params []*graph.Parameter[T]) error {
	for _, p := range params {
		data := p.Value.Data()
		for i := range data {
			data[i] = s.value
		}
		p.Value.SetData(data)
	}
	return nil
}
