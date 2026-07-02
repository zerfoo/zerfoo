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

// Expand broadcasts a tensor to a target shape.
type Expand[T tensor.Numeric] struct {
	engine compute.Engine[T]
}

func (e *Expand[T]) OpType() string                  { return "Expand" }
func (e *Expand[T]) Attributes() map[string]any       { return nil }
func (e *Expand[T]) OutputShape() []int               { return nil }
func (e *Expand[T]) Parameters() []*graph.Parameter[T] { return nil }

func (e *Expand[T]) Forward(ctx context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if len(inputs) != 2 {
		return nil, fmt.Errorf("Expand requires 2 inputs (input, shape), got %d", len(inputs))
	}

	input := inputs[0]
	shapeData := inputs[1].Data()
	targetShape := make([]int, len(shapeData))
	for i, v := range shapeData {
		targetShape[i] = int(v)
	}

	srcShape := input.Shape()

	// Compute output shape using numpy-style broadcasting.
	outShape := broadcastShape(srcShape, targetShape)

	// GPU path: use broadcast multiply with ones to stay GPU-resident.
	if _, ok := input.GetStorage().(*tensor.GPUStorage[T]); ok {
		outSize := 1
		for _, d := range outShape {
			outSize *= d
		}
		ones := make([]T, outSize)
		var one T
		switch v := any(&one).(type) {
		case *float32:
			*v = 1.0
		case *float64:
			*v = 1.0
		}
		for i := range ones {
			ones[i] = one
		}
		onesTensor, err := tensor.New(outShape, ones)
		if err != nil {
			return nil, fmt.Errorf("Expand: failed to create ones tensor: %w", err)
		}
		return e.engine.Mul(ctx, input, onesTensor)
	}

	// CPU path: direct element-wise expansion.
	data := input.Data()
	outSize := 1
	for _, d := range outShape {
		outSize *= d
	}

	out := make([]T, outSize)

	// Compute strides for source in output coordinate space.
	ndim := len(outShape)
	srcPadded := make([]int, ndim)
	offset := ndim - len(srcShape)
	for i, d := range srcShape {
		srcPadded[offset+i] = d
	}

	for i := range out {
		out[i] = data[expandSrcIndex(i, ndim, outShape, srcPadded)]
	}

	return tensor.New(outShape, out)
}

func expandSrcIndex(flatIdx, ndim int, outShape, srcPadded []int) int {
	srcIdx := 0
	srcStride := 1
	for d := ndim - 1; d >= 0; d-- {
		dimSize := outShape[d]
		coord := flatIdx % dimSize
		flatIdx /= dimSize
		srcDim := srcPadded[d]
		if srcDim > 0 {
			srcCoord := coord
			if srcDim == 1 {
				srcCoord = 0
			}
			srcIdx += srcCoord * srcStride
			srcStride *= srcDim
		}
	}
	return srcIdx
}

func broadcastShape(a, b []int) []int {
	n := len(a)
	if len(b) > n {
		n = len(b)
	}
	out := make([]int, n)
	for i := range n {
		da, db := 1, 1
		if ai := len(a) - 1 - i; ai >= 0 {
			da = a[ai]
		}
		if bi := len(b) - 1 - i; bi >= 0 {
			db = b[bi]
		}
		if da > db {
			out[n-1-i] = da
		} else {
			out[n-1-i] = db
		}
	}
	return out
}

// validatedBroadcast computes the broadcast output shape for two input shapes,
// returning an error if the shapes are not broadcast-compatible. It also returns
// the left-padded source shapes for use with expandSrcIndex.
func validatedBroadcast(shapeA, shapeB []int) (outShape, padA, padB []int, err error) {
	ndim := len(shapeA)
	if len(shapeB) > ndim {
		ndim = len(shapeB)
	}
	outShape = make([]int, ndim)
	padA = make([]int, ndim)
	padB = make([]int, ndim)
	offA := ndim - len(shapeA)
	offB := ndim - len(shapeB)
	for i, d := range shapeA {
		padA[offA+i] = d
	}
	for i, d := range shapeB {
		padB[offB+i] = d
	}
	for i := range ndim {
		da, db := padA[i], padB[i]
		if da == 0 {
			da = 1
		}
		if db == 0 {
			db = 1
		}
		if da != db && da != 1 && db != 1 {
			return nil, nil, nil, fmt.Errorf("shapes %v and %v are not broadcast-compatible", shapeA, shapeB)
		}
		if da > db {
			outShape[i] = da
		} else {
			outShape[i] = db
		}
	}
	return outShape, padA, padB, nil
}

func (e *Expand[T]) Backward(_ context.Context, _ types.BackwardMode, _ *tensor.TensorNumeric[T], _ ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	return nil, fmt.Errorf("Expand backward not implemented")
}

// BuildExpand constructs an Expand node from attributes.
func BuildExpand[T tensor.Numeric](
	engine compute.Engine[T], _ numeric.Arithmetic[T], _ string,
	_ map[string]*graph.Parameter[T], _ map[string]any,
) (graph.Node[T], error) {
	return &Expand[T]{engine: engine}, nil
}

var _ graph.Node[float32] = (*Expand[float32])(nil)
