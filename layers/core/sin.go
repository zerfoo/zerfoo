package core

import (
	"context"
	"fmt"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/ztensor/types"
)

// Sin represents an element-wise sine node.
type Sin[T tensor.Numeric] struct {
	engine compute.Engine[T]
}

func (s *Sin[T]) OpType() string                    { return "Sin" }
func (s *Sin[T]) Attributes() map[string]any        { return nil }
func (s *Sin[T]) OutputShape() []int                { return nil }
func (s *Sin[T]) Parameters() []*graph.Parameter[T] { return nil }

func (s *Sin[T]) Forward(ctx context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if len(inputs) != 1 {
		return nil, fmt.Errorf("Sin requires 1 input, got %d", len(inputs))
	}
	if s.engine != nil {
		return s.engine.Sin(ctx, inputs[0])
	}
	return nil, fmt.Errorf("Sin: engine not set")
}

func (s *Sin[T]) Backward(_ context.Context, _ types.BackwardMode, _ *tensor.TensorNumeric[T], _ ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	return nil, fmt.Errorf("Sin backward not implemented")
}

// BuildSin constructs a Sin node from attributes.
func BuildSin[T tensor.Numeric](
	engine compute.Engine[T], _ numeric.Arithmetic[T], _ string,
	_ map[string]*graph.Parameter[T], _ map[string]any,
) (graph.Node[T], error) {
	return &Sin[T]{engine: engine}, nil
}

var _ graph.Node[float32] = (*Sin[float32])(nil)
