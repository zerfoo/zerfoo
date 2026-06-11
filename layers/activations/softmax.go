package activations

import (
	"context"
	"fmt"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/ztensor/types"
)

// Softmax applies the softmax function along a given axis.
//
// Softmax is constrained to floating-point element types: the operation is
// only defined on real-valued inputs, and the backward pass requires the
// arithmetic implied by tensor.Float.
type Softmax[T tensor.Float] struct {
	engine      compute.Engine[T]
	axis        int
	outputShape []int
	output      *tensor.TensorNumeric[T] // cached softmax output for backward
	saver       graph.Saver[T]           // wired by graph Builder (graph.SaverAware); nil outside a Graph
}

// SetSaver implements graph.SaverAware. The cached softmax output is
// expensive to recompute, so it is registered with the graph's
// save-for-backward contract every Forward: arena-backed storage stays
// pinned until this node's Backward has consumed it (ztensor ADR 006).
func (s *Softmax[T]) SetSaver(sv graph.Saver[T]) {
	s.saver = sv
}

// NewSoftmax creates a new Softmax activation layer.
func NewSoftmax[T tensor.Float](engine compute.Engine[T], axis int) *Softmax[T] {
	return &Softmax[T]{engine: engine, axis: axis}
}

// Forward applies softmax to the input tensor along the configured axis.
func (s *Softmax[T]) Forward(ctx context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if len(inputs) != 1 {
		return nil, fmt.Errorf("Softmax expects 1 input, got %d", len(inputs))
	}
	out, err := s.engine.Softmax(ctx, inputs[0], s.axis)
	if err != nil {
		return nil, err
	}
	s.outputShape = out.Shape()
	s.output = out
	if s.saver != nil {
		s.saver.SaveForBackward(out)
	}

	return out, nil
}

// Backward computes the gradient of the softmax function.
//
// For each row along the configured axis:
//
//	dInput = softmaxOutput * (dOutput - sum(dOutput * softmaxOutput, axis, keepdims))
//
// The cached softmax output from the forward pass is reused so the gradient
// works on both CPU and GPU without requiring a custom kernel. This is the
// same math implemented by functional.SoftmaxBackward; the implementation is
// inlined here rather than delegated because layers/functional already
// imports layers/activations (the gelu/sigmoid wrappers), so a delegation
// would introduce an import cycle. A unit test in this package asserts
// numerical equivalence with functional.SoftmaxBackward to keep the two
// implementations honest.
func (s *Softmax[T]) Backward(ctx context.Context, _ types.BackwardMode, dOut *tensor.TensorNumeric[T], _ ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	if s.output == nil {
		return nil, fmt.Errorf("Softmax.Backward called before Forward")
	}
	if dOut == nil {
		return nil, fmt.Errorf("Softmax.Backward: dOut tensor is nil")
	}

	y := s.output
	shape := y.Shape()
	axis := s.axis
	if axis < 0 {
		axis += len(shape)
	}
	if axis < 0 || axis >= len(shape) {
		return nil, fmt.Errorf("Softmax.Backward: axis %d out of range for shape %v", s.axis, shape)
	}

	// prod = dOut * y (elementwise)
	prod, err := s.engine.Mul(ctx, dOut, y)
	if err != nil {
		return nil, fmt.Errorf("Softmax.Backward: mul dOut*y: %w", err)
	}
	// dot = sum(prod, axis, keepDims=true)
	dot, err := s.engine.ReduceSum(ctx, prod, axis, true)
	if err != nil {
		return nil, fmt.Errorf("Softmax.Backward: reduce sum: %w", err)
	}
	// diff = dOut - dot (broadcast over axis)
	diff, err := s.engine.Sub(ctx, dOut, dot)
	if err != nil {
		return nil, fmt.Errorf("Softmax.Backward: sub dot: %w", err)
	}
	// dInput = y * diff
	dInput, err := s.engine.Mul(ctx, y, diff)
	if err != nil {
		return nil, fmt.Errorf("Softmax.Backward: mul y*(dOut-dot): %w", err)
	}
	return []*tensor.TensorNumeric[T]{dInput}, nil
}

// OpType returns "Softmax".
func (s *Softmax[T]) OpType() string { return "Softmax" }

// Attributes returns the softmax configuration.
func (s *Softmax[T]) Attributes() map[string]interface{} {
	return map[string]interface{}{"axis": s.axis}
}

// OutputShape returns the output shape (same as input shape).
func (s *Softmax[T]) OutputShape() []int { return s.outputShape }

// Parameters returns nil (no trainable parameters).
func (s *Softmax[T]) Parameters() []*graph.Parameter[T] { return nil }

// BuildSoftmax constructs a Softmax layer for the registry.
// The optional "axis" attribute (int or int64) selects the softmax axis; defaults to -1.
func BuildSoftmax[T tensor.Float](
	engine compute.Engine[T],
	_ numeric.Arithmetic[T],
	_ string,
	_ map[string]*graph.Parameter[T],
	attributes map[string]interface{},
) (graph.Node[T], error) {
	axis := -1
	if v, ok := attributes["axis"]; ok {
		switch a := v.(type) {
		case int:
			axis = a
		case int64:
			axis = int(a)
		}
	}
	return NewSoftmax(engine, axis), nil
}

// Statically assert that Softmax implements graph.Node.
var _ graph.Node[float32] = (*Softmax[float32])(nil)

// Statically assert that Softmax participates in the save-for-backward contract.
var _ graph.SaverAware[float32] = (*Softmax[float32])(nil)
