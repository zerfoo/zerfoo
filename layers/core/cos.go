package core

import (
	"context"
	"fmt"
	"math"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/ztensor/types"
)

// Cos represents an element-wise cosine node.
type Cos[T tensor.Numeric] struct {
	engine compute.Engine[T]
}

func (c *Cos[T]) OpType() string                    { return "Cos" }
func (c *Cos[T]) Attributes() map[string]any        { return nil }
func (c *Cos[T]) OutputShape() []int                { return nil }
func (c *Cos[T]) Parameters() []*graph.Parameter[T] { return nil }

func (c *Cos[T]) Forward(_ context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if len(inputs) != 1 {
		return nil, fmt.Errorf("Cos requires 1 input, got %d", len(inputs))
	}
	data := inputs[0].Data()
	out := make([]T, len(data))
	switch d := any(data).(type) {
	case []float32:
		o := any(out).([]float32)
		for i, v := range d {
			o[i] = float32(math.Cos(float64(v)))
		}
	case []float64:
		o := any(out).([]float64)
		for i, v := range d {
			o[i] = math.Cos(v)
		}
	default:
		return nil, fmt.Errorf("Cos: unsupported type %T", data)
	}
	return tensor.New(inputs[0].Shape(), out)
}

func (c *Cos[T]) Backward(_ context.Context, _ types.BackwardMode, _ *tensor.TensorNumeric[T], _ ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	return nil, fmt.Errorf("Cos backward not implemented")
}

// BuildCos constructs a Cos node from attributes.
func BuildCos[T tensor.Numeric](
	engine compute.Engine[T], _ numeric.Arithmetic[T], _ string,
	_ map[string]*graph.Parameter[T], _ map[string]any,
) (graph.Node[T], error) {
	return &Cos[T]{engine: engine}, nil
}

var _ graph.Node[float32] = (*Cos[float32])(nil)
