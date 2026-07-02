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

// LessOrEqual represents an element-wise less-than-or-equal comparison.
// Output is 1 for true, 0 for false.
type LessOrEqual[T tensor.Numeric] struct {
	engine compute.Engine[T]
	ops    numeric.Arithmetic[T]
}

func (l *LessOrEqual[T]) OpType() string                  { return "LessOrEqual" }
func (l *LessOrEqual[T]) Attributes() map[string]any       { return nil }
func (l *LessOrEqual[T]) OutputShape() []int               { return nil }
func (l *LessOrEqual[T]) Parameters() []*graph.Parameter[T] { return nil }

func (l *LessOrEqual[T]) Forward(_ context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if len(inputs) != 2 {
		return nil, fmt.Errorf("LessOrEqual requires 2 inputs, got %d", len(inputs))
	}
	a, b := inputs[0].Data(), inputs[1].Data()
	one := l.ops.One()

	if len(b) == 1 {
		out := make([]T, len(a))
		bv := b[0]
		for i := range a {
			if float64(a[i]) <= float64(bv) {
				out[i] = one
			}
		}
		return tensor.New(inputs[0].Shape(), out)
	}
	if len(a) == 1 {
		out := make([]T, len(b))
		av := a[0]
		for i := range b {
			if float64(av) <= float64(b[i]) {
				out[i] = one
			}
		}
		return tensor.New(inputs[1].Shape(), out)
	}

	if len(a) != len(b) {
		return nil, fmt.Errorf("LessOrEqual: input sizes differ (%d vs %d)", len(a), len(b))
	}
	out := make([]T, len(a))
	for i := range a {
		if float64(a[i]) <= float64(b[i]) {
			out[i] = one
		}
	}
	return tensor.New(inputs[0].Shape(), out)
}

func (l *LessOrEqual[T]) Backward(_ context.Context, _ types.BackwardMode, _ *tensor.TensorNumeric[T], _ ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	return nil, fmt.Errorf("LessOrEqual backward not implemented")
}

// BuildLessOrEqual constructs a LessOrEqual node from attributes.
func BuildLessOrEqual[T tensor.Numeric](
	engine compute.Engine[T], ops numeric.Arithmetic[T], _ string,
	_ map[string]*graph.Parameter[T], _ map[string]any,
) (graph.Node[T], error) {
	return &LessOrEqual[T]{engine: engine, ops: ops}, nil
}

var _ graph.Node[float32] = (*LessOrEqual[float32])(nil)
