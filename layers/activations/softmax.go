package activations

import (
	"context"
	"fmt"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/graph"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
	"github.com/zerfoo/zerfoo/types"
)

// Softmax applies the softmax function along a given axis.
type Softmax[T tensor.Numeric] struct {
	engine      compute.Engine[T]
	axis        int
	outputShape []int
}

// NewSoftmax creates a new Softmax activation layer.
func NewSoftmax[T tensor.Numeric](engine compute.Engine[T], axis int) *Softmax[T] {
	return &Softmax[T]{engine: engine, axis: axis}
}

// Forward applies softmax to the input tensor along the configured axis.
func (s *Softmax[T]) Forward(ctx context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if len(inputs) != 1 {
		return nil, fmt.Errorf("Softmax expects 1 input, got %d", len(inputs))
	}
	s.outputShape = inputs[0].Shape()
	return s.engine.Softmax(ctx, inputs[0], s.axis)
}

// Backward returns nil gradients (not required for inference).
func (s *Softmax[T]) Backward(_ context.Context, _ types.BackwardMode, _ *tensor.TensorNumeric[T], _ ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	return nil, nil
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
func BuildSoftmax[T tensor.Numeric](
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
