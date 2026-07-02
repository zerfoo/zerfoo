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

// Neg represents an element-wise negation node.
type Neg[T tensor.Numeric] struct {
	engine compute.Engine[T]
	ops    numeric.Arithmetic[T]
}

func (n *Neg[T]) OpType() string                  { return "Neg" }
func (n *Neg[T]) Attributes() map[string]any       { return nil }
func (n *Neg[T]) OutputShape() []int               { return nil }
func (n *Neg[T]) Parameters() []*graph.Parameter[T] { return nil }

func (n *Neg[T]) Forward(ctx context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if len(inputs) != 1 {
		return nil, fmt.Errorf("Neg requires 1 input, got %d", len(inputs))
	}
	data := inputs[0].Data()
	out := make([]T, len(data))
	zero := n.ops.FromFloat32(0)
	for i, v := range data {
		out[i] = n.ops.Sub(zero, v)
	}
	return tensor.New(inputs[0].Shape(), out)
}

func (n *Neg[T]) Backward(_ context.Context, _ types.BackwardMode, _ *tensor.TensorNumeric[T], _ ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	return nil, fmt.Errorf("Neg backward not implemented")
}

// BuildNeg constructs a Neg node from attributes.
func BuildNeg[T tensor.Numeric](
	engine compute.Engine[T], ops numeric.Arithmetic[T], _ string,
	_ map[string]*graph.Parameter[T], _ map[string]any,
) (graph.Node[T], error) {
	return &Neg[T]{engine: engine, ops: ops}, nil
}

var _ graph.Node[float32] = (*Neg[float32])(nil)
