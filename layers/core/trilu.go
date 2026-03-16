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

// Trilu extracts the upper or lower triangular part of a 2D matrix (or batch of matrices).
type Trilu[T tensor.Numeric] struct {
	engine compute.Engine[T]
	upper  bool // true = upper triangular, false = lower triangular
}

func (t *Trilu[T]) OpType() string { return "Trilu" }
func (t *Trilu[T]) Attributes() map[string]any {
	return map[string]any{"upper": t.upper}
}
func (t *Trilu[T]) OutputShape() []int               { return nil }
func (t *Trilu[T]) Parameters() []*graph.Parameter[T] { return nil }

func (tr *Trilu[T]) Forward(_ context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if len(inputs) < 1 {
		return nil, fmt.Errorf("Trilu requires at least 1 input, got %d", len(inputs))
	}

	input := inputs[0]
	shape := input.Shape()
	ndim := len(shape)
	if ndim < 2 {
		return nil, fmt.Errorf("Trilu requires at least 2D input, got %dD", ndim)
	}

	k := 0 // diagonal offset
	if len(inputs) >= 2 {
		k = int(inputs[1].Data()[0])
	}

	data := input.Data()
	out := make([]T, len(data))

	rows := shape[ndim-2]
	cols := shape[ndim-1]
	matSize := rows * cols

	// Process each matrix in the batch.
	numMats := len(data) / matSize
	for m := range numMats {
		base := m * matSize
		for r := range rows {
			for c := range cols {
				keep := false
				if tr.upper {
					keep = c >= r+k
				} else {
					keep = c <= r+k
				}
				if keep {
					out[base+r*cols+c] = data[base+r*cols+c]
				}
			}
		}
	}

	return tensor.New(shape, out)
}

func (tr *Trilu[T]) Backward(_ context.Context, _ types.BackwardMode, _ *tensor.TensorNumeric[T], _ ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	return nil, fmt.Errorf("Trilu backward not implemented")
}

// BuildTrilu constructs a Trilu node from attributes.
func BuildTrilu[T tensor.Numeric](
	engine compute.Engine[T], _ numeric.Arithmetic[T], _ string,
	_ map[string]*graph.Parameter[T], attrs map[string]any,
) (graph.Node[T], error) {
	upper := true // ONNX default
	if v, ok := attrs["upper"]; ok {
		switch val := v.(type) {
		case int64:
			upper = val != 0
		case bool:
			upper = val
		}
	}
	return &Trilu[T]{engine: engine, upper: upper}, nil
}

var _ graph.Node[float32] = (*Trilu[float32])(nil)
