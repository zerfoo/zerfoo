package core //nolint:dupl // Pow follows the same binary-op pattern as Add/Sub/Mul/Div

import (
	"context"
	"fmt"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/ztensor/types"
)

// Pow represents an element-wise power node.
type Pow[T tensor.Numeric] struct {
	engine compute.Engine[T]
}

// NewPow creates a new Pow node.
func NewPow[T tensor.Numeric](engine compute.Engine[T]) *Pow[T] {
	return &Pow[T]{engine: engine}
}

func (p *Pow[T]) OpType() string                  { return "Pow" }
func (p *Pow[T]) Attributes() map[string]any       { return nil }
func (p *Pow[T]) OutputShape() []int               { return nil }
func (p *Pow[T]) Parameters() []*graph.Parameter[T] { return nil }

func (p *Pow[T]) Forward(ctx context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if len(inputs) != 2 {
		return nil, fmt.Errorf("Pow requires 2 inputs, got %d", len(inputs))
	}
	return p.engine.Pow(ctx, inputs[0], inputs[1])
}

func (p *Pow[T]) Backward(_ context.Context, _ types.BackwardMode, _ *tensor.TensorNumeric[T], _ ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	return nil, fmt.Errorf("Pow backward not implemented")
}

// BuildPow constructs a Pow node from attributes.
func BuildPow[T tensor.Numeric](
	engine compute.Engine[T], _ numeric.Arithmetic[T], _ string,
	_ map[string]*graph.Parameter[T], _ map[string]any,
) (graph.Node[T], error) {
	return NewPow[T](engine), nil
}

var _ graph.Node[float32] = (*Pow[float32])(nil)
