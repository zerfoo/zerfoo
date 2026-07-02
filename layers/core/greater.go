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

// Greater represents an element-wise greater-than comparison. Output is 1 for true, 0 for false.
type Greater[T tensor.Numeric] struct {
	engine compute.Engine[T]
	ops    numeric.Arithmetic[T]
}

func (g *Greater[T]) OpType() string                  { return "Greater" }
func (g *Greater[T]) Attributes() map[string]any       { return nil }
func (g *Greater[T]) OutputShape() []int               { return nil }
func (g *Greater[T]) Parameters() []*graph.Parameter[T] { return nil }

func (g *Greater[T]) Forward(_ context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if len(inputs) != 2 {
		return nil, fmt.Errorf("Greater requires 2 inputs, got %d", len(inputs))
	}
	a, b := inputs[0].Data(), inputs[1].Data()
	one := g.ops.One()

	// Scalar broadcasting.
	if len(b) == 1 {
		out := make([]T, len(a))
		bv := float64(b[0])
		for i := range a {
			if float64(a[i]) > bv {
				out[i] = one
			}
		}
		return tensor.New(inputs[0].Shape(), out)
	}
	if len(a) == 1 {
		out := make([]T, len(b))
		av := float64(a[0])
		for i := range b {
			if av > float64(b[i]) {
				out[i] = one
			}
		}
		return tensor.New(inputs[1].Shape(), out)
	}

	// General broadcasting.
	shapeA, shapeB := inputs[0].Shape(), inputs[1].Shape()
	outShape, padA, padB, err := validatedBroadcast(shapeA, shapeB)
	if err != nil {
		return nil, fmt.Errorf("Greater: %w", err)
	}
	ndim := len(outShape)
	outSize := 1
	for _, d := range outShape {
		outSize *= d
	}

	out := make([]T, outSize)
	for i := range out {
		ai := expandSrcIndex(i, ndim, outShape, padA)
		bi := expandSrcIndex(i, ndim, outShape, padB)
		if float64(a[ai]) > float64(b[bi]) {
			out[i] = one
		}
	}
	return tensor.New(outShape, out)
}

func (g *Greater[T]) Backward(_ context.Context, _ types.BackwardMode, _ *tensor.TensorNumeric[T], _ ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	return nil, fmt.Errorf("Greater backward not implemented")
}

// BuildGreater constructs a Greater node from attributes.
func BuildGreater[T tensor.Numeric](
	engine compute.Engine[T], ops numeric.Arithmetic[T], _ string,
	_ map[string]*graph.Parameter[T], _ map[string]any,
) (graph.Node[T], error) {
	return &Greater[T]{engine: engine, ops: ops}, nil
}

var _ graph.Node[float32] = (*Greater[float32])(nil)
