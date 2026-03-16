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

// Max computes the element-wise maximum of two tensors.
type Max[T tensor.Numeric] struct {
	engine compute.Engine[T]
}

func (m *Max[T]) OpType() string                   { return "Max" }
func (m *Max[T]) Attributes() map[string]any        { return nil }
func (m *Max[T]) OutputShape() []int                { return nil }
func (m *Max[T]) Parameters() []*graph.Parameter[T] { return nil }

func (m *Max[T]) Forward(_ context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if len(inputs) != 2 {
		return nil, fmt.Errorf("Max requires 2 inputs, got %d", len(inputs))
	}
	a, b := inputs[0].Data(), inputs[1].Data()
	if len(a) != len(b) {
		return nil, fmt.Errorf("Max: input sizes differ (%d vs %d)", len(a), len(b))
	}
	out := make([]T, len(a))
	ops := m.engine.Ops()
	for i := range out {
		if ops.GreaterThan(b[i], a[i]) {
			out[i] = b[i]
		} else {
			out[i] = a[i]
		}
	}
	return tensor.New(inputs[0].Shape(), out)
}

func (m *Max[T]) Backward(_ context.Context, _ types.BackwardMode, _ *tensor.TensorNumeric[T], _ ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	return nil, fmt.Errorf("Max backward not implemented")
}

// BuildMax constructs a Max node from attributes.
func BuildMax[T tensor.Numeric](
	engine compute.Engine[T], _ numeric.Arithmetic[T], _ string,
	_ map[string]*graph.Parameter[T], _ map[string]any,
) (graph.Node[T], error) {
	return &Max[T]{engine: engine}, nil
}

var _ graph.Node[float32] = (*Max[float32])(nil)
