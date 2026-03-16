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

// Sqrt represents an element-wise square root node.
type Sqrt[T tensor.Numeric] struct {
	engine compute.Engine[T]
}

// NewSqrt creates a new Sqrt node.
func NewSqrt[T tensor.Numeric](engine compute.Engine[T]) *Sqrt[T] {
	return &Sqrt[T]{engine: engine}
}

func (s *Sqrt[T]) OpType() string                  { return "Sqrt" }
func (s *Sqrt[T]) Attributes() map[string]any       { return nil }
func (s *Sqrt[T]) OutputShape() []int               { return nil }
func (s *Sqrt[T]) Parameters() []*graph.Parameter[T] { return nil }

func (s *Sqrt[T]) Forward(ctx context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if len(inputs) != 1 {
		return nil, fmt.Errorf("Sqrt requires 1 input, got %d", len(inputs))
	}
	return s.engine.Sqrt(ctx, inputs[0])
}

func (s *Sqrt[T]) Backward(_ context.Context, _ types.BackwardMode, _ *tensor.TensorNumeric[T], _ ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	return nil, fmt.Errorf("Sqrt backward not implemented")
}

// BuildSqrt constructs a Sqrt node from attributes.
func BuildSqrt[T tensor.Numeric](
	engine compute.Engine[T], _ numeric.Arithmetic[T], _ string,
	_ map[string]*graph.Parameter[T], _ map[string]any,
) (graph.Node[T], error) {
	return NewSqrt[T](engine), nil
}

var _ graph.Node[float32] = (*Sqrt[float32])(nil)
