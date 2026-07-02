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

func (s *Sqrt[T]) Backward(ctx context.Context, _ types.BackwardMode, dOut *tensor.TensorNumeric[T], inputs ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	if len(inputs) != 1 {
		return nil, fmt.Errorf("Sqrt backward requires 1 input, got %d", len(inputs))
	}

	a := inputs[0]

	// gradA = dOut * 0.5 / sqrt(a)
	sqrtA, err := s.engine.Sqrt(ctx, a)
	if err != nil {
		return nil, fmt.Errorf("Sqrt backward sqrt(a): %w", err)
	}

	ops := s.engine.Ops()
	half := ops.FromFloat32(0.5)
	halfOverSqrtA, err := s.engine.UnaryOp(ctx, sqrtA, func(x T) T {
		return ops.Div(half, x)
	})
	if err != nil {
		return nil, fmt.Errorf("Sqrt backward 0.5/sqrt(a): %w", err)
	}

	gradA, err := s.engine.Mul(ctx, dOut, halfOverSqrtA)
	if err != nil {
		return nil, fmt.Errorf("Sqrt backward dOut * 0.5/sqrt(a): %w", err)
	}

	return []*tensor.TensorNumeric[T]{gradA}, nil
}

// BuildSqrt constructs a Sqrt node from attributes.
func BuildSqrt[T tensor.Numeric](
	engine compute.Engine[T], _ numeric.Arithmetic[T], _ string,
	_ map[string]*graph.Parameter[T], _ map[string]any,
) (graph.Node[T], error) {
	return NewSqrt[T](engine), nil
}

var _ graph.Node[float32] = (*Sqrt[float32])(nil)
