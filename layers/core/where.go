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

// Where selects elements from two tensors based on a condition tensor.
// Output[i] = x[i] if condition[i] != 0, else y[i].
type Where[T tensor.Numeric] struct {
	engine compute.Engine[T]
}

func (w *Where[T]) OpType() string                  { return "Where" }
func (w *Where[T]) Attributes() map[string]any       { return nil }
func (w *Where[T]) OutputShape() []int               { return nil }
func (w *Where[T]) Parameters() []*graph.Parameter[T] { return nil }

func (w *Where[T]) Forward(_ context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if len(inputs) != 3 {
		return nil, fmt.Errorf("Where requires 3 inputs (condition, x, y), got %d", len(inputs))
	}
	cond, x, y := inputs[0].Data(), inputs[1].Data(), inputs[2].Data()
	shapeCond, shapeX, shapeY := inputs[0].Shape(), inputs[1].Shape(), inputs[2].Shape()

	// Compute broadcast output shape across all three inputs.
	outShapeCX, padCond, padX, err := validatedBroadcast(shapeCond, shapeX)
	if err != nil {
		return nil, fmt.Errorf("Where: %w", err)
	}
	outShape, padCX, padY, err := validatedBroadcast(outShapeCX, shapeY)
	if err != nil {
		return nil, fmt.Errorf("Where: %w", err)
	}
	ndim := len(outShape)
	outSize := 1
	for _, d := range outShape {
		outSize *= d
	}

	// Re-pad cond and x shapes to the final ndim (may have grown).
	if len(padCond) < ndim {
		padCond = padShapeTo(shapeCond, ndim)
		padX = padShapeTo(shapeX, ndim)
	}
	_ = padCX // intermediate; cond and x already re-padded

	out := make([]T, outSize)
	for i := range out {
		ci := expandSrcIndex(i, ndim, outShape, padCond)
		if cond[ci] != 0 {
			out[i] = x[expandSrcIndex(i, ndim, outShape, padX)]
		} else {
			out[i] = y[expandSrcIndex(i, ndim, outShape, padY)]
		}
	}
	return tensor.New(outShape, out)
}

// padShapeTo left-pads a shape to the given ndim with zeros.
func padShapeTo(shape []int, ndim int) []int {
	padded := make([]int, ndim)
	off := ndim - len(shape)
	for i, d := range shape {
		padded[off+i] = d
	}
	return padded
}

func (w *Where[T]) Backward(_ context.Context, _ types.BackwardMode, _ *tensor.TensorNumeric[T], _ ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	return nil, fmt.Errorf("Where backward not implemented")
}

// BuildWhere constructs a Where node from attributes.
func BuildWhere[T tensor.Numeric](
	engine compute.Engine[T], _ numeric.Arithmetic[T], _ string,
	_ map[string]*graph.Parameter[T], _ map[string]any,
) (graph.Node[T], error) {
	return &Where[T]{engine: engine}, nil
}

var _ graph.Node[float32] = (*Where[float32])(nil)
