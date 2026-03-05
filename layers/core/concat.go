// Package core provides core layer implementations for the Zerfoo ML framework.
package core

import (
	"context"
	"fmt"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/graph"
	"github.com/zerfoo/zerfoo/tensor"
	"github.com/zerfoo/zerfoo/types"
)

// Concat is a layer that concatenates multiple tensors along a specified axis.
type Concat[T tensor.Numeric] struct {
	engine      compute.Engine[T]
	axis        int
	outputShape []int
}

// NewConcat creates a new Concat layer.
func NewConcat[T tensor.Numeric](engine compute.Engine[T], axis int) *Concat[T] {
	return &Concat[T]{
		engine: engine,
		axis:   axis,
	}
}

// OutputShape returns the output shape of the Concat layer.
func (c *Concat[T]) OutputShape() []int {
	return c.outputShape
}

// Parameters returns no trainable parameters for the Concat layer.
func (c *Concat[T]) Parameters() []*graph.Parameter[T] {
	return nil
}

// Forward computes the concatenation of input tensors along the specified axis.
func (c *Concat[T]) Forward(_ context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if len(inputs) == 0 {
		panic("Concat layer requires at least 1 input")
	}

	// Handle single input case (no-op)
	if len(inputs) == 1 {
		c.outputShape = inputs[0].Shape()

		return inputs[0], nil
	}

	// Ensure all inputs have the same rank by unsqueezing lower-rank tensors.
	// This handles the common ONNX pattern where Constant [N] tensors are
	// concatenated with Unsqueeze outputs of shape [1, N].
	maxRank := 0
	for _, t := range inputs {
		if r := len(t.Shape()); r > maxRank {
			maxRank = r
		}
	}
	needsAlign := false
	for _, t := range inputs {
		if len(t.Shape()) < maxRank {
			needsAlign = true
			break
		}
	}
	aligned := inputs
	if needsAlign {
		aligned = make([]*tensor.TensorNumeric[T], len(inputs))
		for i, t := range inputs {
			if r := len(t.Shape()); r < maxRank {
				// Prepend dimensions of size 1 to match maxRank.
				newShape := make([]int, maxRank)
				pad := maxRank - r
				for j := range pad {
					newShape[j] = 1
				}
				copy(newShape[pad:], t.Shape())
				reshaped, err := tensor.New[T](newShape, t.Data())
				if err != nil {
					return nil, fmt.Errorf("concat rank alignment failed for input %d: %w", i, err)
				}
				aligned[i] = reshaped
			} else {
				aligned[i] = t
			}
		}
	}

	// Perform actual concatenation via engine
	out, err := c.engine.Concat(context.Background(), aligned, c.axis)
	if err != nil {
		return nil, err
	}

	c.outputShape = out.Shape()

	return out, nil
}

// Backward computes the gradients for the Concat layer.
func (c *Concat[T]) Backward(_ context.Context, mode types.BackwardMode, outputGradient *tensor.TensorNumeric[T], inputs ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	if len(inputs) == 0 {
		panic("Concat layer requires at least 1 input")
	}

	// Handle single input case (gradient passes through unchanged)
	if len(inputs) == 1 {
		return []*tensor.TensorNumeric[T]{outputGradient}, nil
	}

	// Properly split gradient along the concatenation axis according to each input's size
	shape := outputGradient.Shape()

	axis := c.axis
	if axis < 0 {
		axis = len(shape) + axis
	}

	// Compute block size (product of dims after axis) and outer (product before axis)
	blockSize := 1
	for i := axis + 1; i < len(shape); i++ {
		blockSize *= shape[i]
	}

	outer := 1
	for i := range shape[:axis] {
		outer *= shape[i]
	}

	grads := make([]*tensor.TensorNumeric[T], len(inputs))
	// Precompute per-input lengths along axis
	inAxisLens := make([]int, len(inputs))
	totalAxis := 0

	for i, in := range inputs {
		inAxisLens[i] = in.Shape()[axis]
		totalAxis += inAxisLens[i]
	}

	// Sanity: outputGradient axis length must match sum of inputs along axis
	if totalAxis != shape[axis] {
		return nil, fmt.Errorf("Concat backward: mismatch along axis %d: out %d vs sum(inputs) %d", axis, shape[axis], totalAxis)
	}

	// Allocate gradient tensors per input and copy corresponding slices
	for i, in := range inputs {
		g, err := tensor.New[T](in.Shape(), nil)
		if err != nil {
			return nil, err
		}

		grads[i] = g
	}

	outData := outputGradient.Data()
	// runningOffset tracks how many positions along axis we've consumed in out gradient
	runningOffset := 0

	for i, g := range grads {
		part := inAxisLens[i]
		gData := g.Data()

		for o := 0; o < outer; o++ { //nolint:intrange
			for j := 0; j < part; j++ { //nolint:intrange
				srcStart := o*shape[axis]*blockSize + (runningOffset+j)*blockSize
				dstStart := o*part*blockSize + j*blockSize
				copy(gData[dstStart:dstStart+blockSize], outData[srcStart:srcStart+blockSize])
			}
		}

		runningOffset += part
	}

	return grads, nil
}

// OpType returns the operation type of the Concat layer.
func (c *Concat[T]) OpType() string {
	return "Concat"
}

// Attributes returns the attributes of the Concat layer.
func (c *Concat[T]) Attributes() map[string]interface{} {
	return map[string]interface{}{"axis": c.axis}
}

// Statically assert that the type implements the graph.Node interface.
var _ graph.Node[float32] = (*Concat[float32])(nil)
