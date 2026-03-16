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

// Or represents an element-wise logical OR. Output is 1 when either input is
// nonzero, 0 otherwise.
type Or[T tensor.Numeric] struct {
	engine compute.Engine[T]
	ops    numeric.Arithmetic[T]
}

func (o *Or[T]) OpType() string                  { return "Or" }
func (o *Or[T]) Attributes() map[string]any       { return nil }
func (o *Or[T]) OutputShape() []int               { return nil }
func (o *Or[T]) Parameters() []*graph.Parameter[T] { return nil }

func (o *Or[T]) Forward(_ context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if len(inputs) != 2 {
		return nil, fmt.Errorf("Or requires 2 inputs, got %d", len(inputs))
	}
	a, b := inputs[0].Data(), inputs[1].Data()
	one := o.ops.One()
	var zero T

	if len(b) == 1 {
		out := make([]T, len(a))
		bv := b[0]
		for i := range a {
			if a[i] != zero || bv != zero {
				out[i] = one
			}
		}
		return tensor.New(inputs[0].Shape(), out)
	}
	if len(a) == 1 {
		out := make([]T, len(b))
		av := a[0]
		for i := range b {
			if av != zero || b[i] != zero {
				out[i] = one
			}
		}
		return tensor.New(inputs[1].Shape(), out)
	}

	// General N-D broadcasting.
	shapeA, shapeB := inputs[0].Shape(), inputs[1].Shape()
	outShape, padA, padB, err := validatedBroadcast(shapeA, shapeB)
	if err != nil {
		return nil, fmt.Errorf("Or: %w", err)
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
		if a[ai] != zero || b[bi] != zero {
			out[i] = one
		}
	}
	return tensor.New(outShape, out)
}

func (o *Or[T]) Backward(_ context.Context, _ types.BackwardMode, _ *tensor.TensorNumeric[T], _ ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	return nil, fmt.Errorf("Or backward not implemented")
}

// BuildOr constructs an Or node from attributes.
func BuildOr[T tensor.Numeric](
	engine compute.Engine[T], ops numeric.Arithmetic[T], _ string,
	_ map[string]*graph.Parameter[T], _ map[string]any,
) (graph.Node[T], error) {
	return &Or[T]{engine: engine, ops: ops}, nil
}

var _ graph.Node[float32] = (*Or[float32])(nil)
