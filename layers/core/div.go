package core //nolint:dupl // Div follows the same binary-op pattern as Add/Sub/Mul/Pow

import (
	"context"
	"fmt"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/ztensor/types"
)

// Div represents an element-wise division node.
type Div[T tensor.Numeric] struct {
	engine compute.Engine[T]
}

// NewDiv creates a new Div node.
func NewDiv[T tensor.Numeric](engine compute.Engine[T]) *Div[T] {
	return &Div[T]{engine: engine}
}

func (d *Div[T]) OpType() string                  { return "Div" }
func (d *Div[T]) Attributes() map[string]any       { return nil }
func (d *Div[T]) OutputShape() []int               { return nil }
func (d *Div[T]) Parameters() []*graph.Parameter[T] { return nil }

func (d *Div[T]) Forward(ctx context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if len(inputs) != 2 {
		return nil, fmt.Errorf("Div requires 2 inputs, got %d", len(inputs))
	}
	return d.engine.Div(ctx, inputs[0], inputs[1])
}

func (d *Div[T]) Backward(_ context.Context, _ types.BackwardMode, _ *tensor.TensorNumeric[T], _ ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	return nil, fmt.Errorf("Div backward not implemented")
}

// BuildDiv constructs a Div node from attributes.
func BuildDiv[T tensor.Numeric](
	engine compute.Engine[T], _ numeric.Arithmetic[T], _ string,
	_ map[string]*graph.Parameter[T], _ map[string]any,
) (graph.Node[T], error) {
	return NewDiv[T](engine), nil
}

var _ graph.Node[float32] = (*Div[float32])(nil)
