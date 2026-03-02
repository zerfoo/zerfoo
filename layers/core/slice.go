package core

import (
	"context"
	"fmt"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/graph"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
	"github.com/zerfoo/zerfoo/types"
)

// Slice extracts a sub-tensor using start/end/axes/steps attributes.
// Steps other than 1 are not supported; steps are accepted but ignored.
type Slice[T tensor.Numeric] struct {
	engine      compute.Engine[T]
	starts      []int64
	ends        []int64
	axes        []int64 // nil means apply to axes 0..len(starts)-1
	steps       []int64 // nil or all-1; non-1 steps are not supported
	outputShape []int
}

// NewSlice creates a new Slice layer.
func NewSlice[T tensor.Numeric](engine compute.Engine[T], starts, ends, axes, steps []int64) *Slice[T] {
	return &Slice[T]{engine: engine, starts: starts, ends: ends, axes: axes, steps: steps}
}

// Forward applies the slice operation to the input tensor.
func (s *Slice[T]) Forward(_ context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if len(inputs) != 1 {
		return nil, fmt.Errorf("Slice expects 1 input, got %d", len(inputs))
	}
	input := inputs[0]
	shape := input.Shape()
	ndim := len(shape)

	// Resolve axes (default: 0..len(starts)-1).
	axes := s.axes
	if axes == nil {
		axes = make([]int64, len(s.starts))
		for i := range axes {
			axes[i] = int64(i)
		}
	}

	// Build the full set of ranges, defaulting to the full dimension.
	ranges := make([][2]int, ndim)
	for d := range ndim {
		ranges[d] = [2]int{0, shape[d]}
	}

	for i, ax := range axes {
		dim := int(ax)
		if dim < 0 {
			dim += ndim
		}
		start := int(s.starts[i])
		if start < 0 {
			start += shape[dim]
		}
		end := int(s.ends[i])
		if end < 0 {
			end += shape[dim]
		}
		if end > shape[dim] {
			end = shape[dim]
		}
		if start < 0 {
			start = 0
		}
		ranges[dim] = [2]int{start, end}
	}

	sliced, err := input.Slice(ranges...)
	if err != nil {
		return nil, fmt.Errorf("Slice.Forward: %w", err)
	}

	// Return a dense copy so downstream nodes are not tied to the view.
	outData := sliced.Data()
	out, err := tensor.New[T](sliced.Shape(), outData)
	if err != nil {
		return nil, fmt.Errorf("Slice.Forward copy: %w", err)
	}
	s.outputShape = out.Shape()
	return out, nil
}

// Backward returns nil (not required for inference).
func (s *Slice[T]) Backward(_ context.Context, _ types.BackwardMode, _ *tensor.TensorNumeric[T], _ ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	return nil, nil
}

// OpType returns "Slice".
func (s *Slice[T]) OpType() string { return "Slice" }

// Attributes returns the slice configuration.
func (s *Slice[T]) Attributes() map[string]interface{} {
	return map[string]interface{}{
		"starts": s.starts,
		"ends":   s.ends,
		"axes":   s.axes,
	}
}

// OutputShape returns the output shape from the last forward call.
func (s *Slice[T]) OutputShape() []int { return s.outputShape }

// Parameters returns nil (no trainable parameters).
func (s *Slice[T]) Parameters() []*graph.Parameter[T] { return nil }

// BuildSlice constructs a Slice layer from ZMF attributes.
// Supported attribute keys: "starts", "ends", "axes", "steps" (all []int64).
func BuildSlice[T tensor.Numeric](
	engine compute.Engine[T],
	_ numeric.Arithmetic[T],
	_ string,
	_ map[string]*graph.Parameter[T],
	attributes map[string]interface{},
) (graph.Node[T], error) {
	starts := extractInt64Slice(attributes, "starts")
	ends := extractInt64Slice(attributes, "ends")
	axes := extractInt64Slice(attributes, "axes")
	steps := extractInt64Slice(attributes, "steps")
	return NewSlice(engine, starts, ends, axes, steps), nil
}

// extractInt64Slice reads a []int64 value from attributes, returning nil if absent.
func extractInt64Slice(attrs map[string]interface{}, key string) []int64 {
	v, ok := attrs[key]
	if !ok {
		return nil
	}
	if s, ok := v.([]int64); ok {
		return s
	}
	return nil
}

// Statically assert that Slice implements graph.Node.
var _ graph.Node[float32] = (*Slice[float32])(nil)
