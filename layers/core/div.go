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

func (d *Div[T]) Backward(ctx context.Context, _ types.BackwardMode, dOut *tensor.TensorNumeric[T], inputs ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	if len(inputs) != 2 {
		return nil, fmt.Errorf("Div backward requires 2 inputs, got %d", len(inputs))
	}

	a := inputs[0]
	b := inputs[1]

	// gradA = dOut / b  (d/da(a/b) = 1/b)
	gradA, err := d.engine.Div(ctx, dOut, b)
	if err != nil {
		return nil, fmt.Errorf("Div backward gradA: %w", err)
	}

	// gradB = -dOut * a / (b * b)  (d/db(a/b) = -a/b²)
	bSquared, err := d.engine.Mul(ctx, b, b)
	if err != nil {
		return nil, fmt.Errorf("Div backward b²: %w", err)
	}

	aOverBSq, err := d.engine.Div(ctx, a, bSquared)
	if err != nil {
		return nil, fmt.Errorf("Div backward a/b²: %w", err)
	}

	gradB, err := d.engine.Mul(ctx, dOut, aOverBSq)
	if err != nil {
		return nil, fmt.Errorf("Div backward dOut*a/b²: %w", err)
	}

	// Negate gradB
	ops := d.engine.Ops()
	negOne := ops.FromFloat32(-1.0)
	gradB, err = d.engine.UnaryOp(ctx, gradB, func(x T) T { return ops.Mul(x, negOne) })
	if err != nil {
		return nil, fmt.Errorf("Div backward negate gradB: %w", err)
	}

	return []*tensor.TensorNumeric[T]{gradA, gradB}, nil
}

// BuildDiv constructs a Div node from attributes.
func BuildDiv[T tensor.Numeric](
	engine compute.Engine[T], _ numeric.Arithmetic[T], _ string,
	_ map[string]*graph.Parameter[T], _ map[string]any,
) (graph.Node[T], error) {
	return NewDiv[T](engine), nil
}

var _ graph.Node[float32] = (*Div[float32])(nil)
