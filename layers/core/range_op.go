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

// Range generates a sequence of numbers [start, limit) with given delta.
type Range[T tensor.Numeric] struct {
	engine compute.Engine[T]
}

func (r *Range[T]) OpType() string                  { return "Range" }
func (r *Range[T]) Attributes() map[string]any       { return nil }
func (r *Range[T]) OutputShape() []int               { return nil }
func (r *Range[T]) Parameters() []*graph.Parameter[T] { return nil }

func (r *Range[T]) Forward(_ context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if len(inputs) != 3 {
		return nil, fmt.Errorf("Range requires 3 inputs (start, limit, delta), got %d", len(inputs))
	}

	// Extract scalar values from each input, validating that each has at
	// least one element. Inputs may be 0-D scalars (shape []) or 1-D
	// tensors (shape [1]). Data() performs a D2H copy for GPU tensors.
	names := [3]string{"start", "limit", "delta"}
	vals := [3]float64{}
	for i := 0; i < 3; i++ {
		data := inputs[i].Data()
		if len(data) == 0 {
			return nil, fmt.Errorf("Range: %s input (inputs[%d]) has no data (shape=%v)", names[i], i, inputs[i].Shape())
		}
		vals[i] = float64(data[0])
	}
	start, limit, delta := vals[0], vals[1], vals[2]

	if delta == 0 {
		return nil, fmt.Errorf("Range: delta cannot be zero")
	}

	n := int((limit - start) / delta)
	if n < 0 {
		n = 0
	}

	out := make([]T, n)
	val := start
	for i := range out {
		out[i] = T(val)
		val += delta
	}

	return tensor.New([]int{n}, out)
}

func (r *Range[T]) Backward(_ context.Context, _ types.BackwardMode, _ *tensor.TensorNumeric[T], _ ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	return nil, fmt.Errorf("Range backward not implemented")
}

// BuildRange constructs a Range node from attributes.
func BuildRange[T tensor.Numeric](
	engine compute.Engine[T], _ numeric.Arithmetic[T], _ string,
	_ map[string]*graph.Parameter[T], _ map[string]any,
) (graph.Node[T], error) {
	return &Range[T]{engine: engine}, nil
}

var _ graph.Node[float32] = (*Range[float32])(nil)
