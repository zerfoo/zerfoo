// Package transpose provides the Transpose layer for the Zerfoo ML framework.
package transpose

import (
	"context"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/ztensor/types"
)

// Transpose represents a transpose operation.
type Transpose[T tensor.Numeric] struct {
	engine      compute.Engine[T]
	perm        []int
	outputShape []int
}

// OpType returns the operation type.
func (t *Transpose[T]) OpType() string {
	return "Transpose"
}

// Attributes returns the attributes.
func (t *Transpose[T]) Attributes() map[string]interface{} {
	return map[string]interface{}{
		"perm": t.perm,
	}
}

// New creates a new Transpose layer.
func New[T tensor.Numeric](engine compute.Engine[T], axes []int) *Transpose[T] {
	return &Transpose[T]{
		engine: engine,
		perm:   axes,
	}
}

// OutputShape returns the output shape of the Transpose layer.
func (t *Transpose[T]) OutputShape() []int {
	return t.outputShape
}

// Parameters returns no trainable parameters for the Transpose layer.
func (t *Transpose[T]) Parameters() []*graph.Parameter[T] {
	return nil
}

// Forward computes the transpose operation.
//
// Result caching was removed because the graph's arena pool may reclaim
// the transposed tensor's GPU memory between forward passes (via
// ResetPool), leaving a stale GPUStorage with devicePtr=nil. Returning
// such a cached tensor caused cuBLAS status 7 (INTERNAL_ERROR) when the
// downstream MatMul tried to use the null device pointer.
func (t *Transpose[T]) Forward(ctx context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	input := inputs[0]
	shape := input.Shape()

	// If perm is nil, use the ONNX default: reverse all axes.
	perm := t.perm
	if perm == nil {
		perm = make([]int, len(shape))
		for i := range perm {
			perm[i] = len(shape) - 1 - i
		}
		t.perm = perm
	}

	outputShape := make([]int, len(shape))
	for i, axis := range perm {
		outputShape[i] = shape[axis]
	}

	t.outputShape = outputShape

	// Q4 pass-through: when the input is a Q4-backed weight tensor and
	// the transpose is a simple 2D swap [1,0], preserve Q4 storage with
	// the transposed shape. Q4Storage lives outside the arena pool so
	// the virtual transpose is safe to reuse.
	if len(shape) == 2 && len(perm) == 2 && perm[0] == 1 && perm[1] == 0 {
		if _, ok := any(input.GetStorage()).(*tensor.Q4Storage); ok {
			return tensor.NewWithStorage[T](outputShape, input.GetStorage())
		}
	}

	return t.engine.Transpose(ctx, input, perm)
}

// Backward computes the gradients for the Transpose layer.
func (t *Transpose[T]) Backward(ctx context.Context, _ types.BackwardMode, outputGradient *tensor.TensorNumeric[T], _ ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	// The gradient w.r.t. the input is the gradient transposed by the inverse permutation.
	inv := make([]int, len(t.perm))
	for i, p := range t.perm {
		inv[p] = i
	}

	gradInput, err := t.engine.Transpose(ctx, outputGradient, inv)
	if err != nil {
		return nil, err
	}
	return []*tensor.TensorNumeric[T]{gradInput}, nil
}

// Statically assert that the type implements the graph.Node interface.
var _ graph.Node[float32] = (*Transpose[float32])(nil)
