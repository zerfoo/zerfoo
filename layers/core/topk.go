package core

import (
	"context"
	"errors"
	"fmt"
	"sort"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/graph"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
	"github.com/zerfoo/zerfoo/types"
)

// TopK selects the top-k values along the last axis of the input tensor.
// It returns only the values (not indices) as a single output tensor.
// When largest=true (default) it returns the k largest values in descending order;
// when largest=false it returns the k smallest values in ascending order.
type TopK[T tensor.Numeric] struct {
	engine      compute.Engine[T]
	k           int
	axis        int
	largest     bool
	sorted      bool
	outputShape []int
}

// NewTopK creates a new TopK layer.
func NewTopK[T tensor.Numeric](engine compute.Engine[T], k, axis int, largest, sorted bool) *TopK[T] {
	return &TopK[T]{engine: engine, k: k, axis: axis, largest: largest, sorted: sorted}
}

// Forward selects the top-k values along the last axis of the input.
func (tk *TopK[T]) Forward(_ context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if len(inputs) != 1 {
		return nil, fmt.Errorf("TopK expects 1 input, got %d", len(inputs))
	}
	input := inputs[0]
	data := input.Data()
	n := len(data)

	k := tk.k
	if k > n {
		k = n
	}

	// Build an index slice and sort it by value.
	indices := make([]int, n)
	for i := range indices {
		indices[i] = i
	}

	if tk.largest {
		sort.Slice(indices, func(i, j int) bool {
			return data[indices[i]] > data[indices[j]]
		})
	} else {
		sort.Slice(indices, func(i, j int) bool {
			return data[indices[i]] < data[indices[j]]
		})
	}

	// Select the top-k indices. The sort.Slice above already produces the
	// required order: descending for largest=true, ascending for largest=false.
	// When sorted=false the caller accepts any order; we keep the sorted result.
	topIndices := indices[:k]

	// Gather the values.
	outData := make([]T, k)
	for i, idx := range topIndices {
		outData[i] = data[idx]
	}

	out, err := tensor.New[T]([]int{k}, outData)
	if err != nil {
		return nil, fmt.Errorf("TopK.Forward: %w", err)
	}
	tk.outputShape = out.Shape()
	return out, nil
}

// Backward returns nil (not required for inference).
func (tk *TopK[T]) Backward(_ context.Context, _ types.BackwardMode, _ *tensor.TensorNumeric[T], _ ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	return nil, nil
}

// OpType returns "TopK".
func (tk *TopK[T]) OpType() string { return "TopK" }

// Attributes returns the TopK configuration.
func (tk *TopK[T]) Attributes() map[string]interface{} {
	return map[string]interface{}{"k": tk.k, "axis": tk.axis, "largest": tk.largest}
}

// OutputShape returns the output shape from the last forward call.
func (tk *TopK[T]) OutputShape() []int { return tk.outputShape }

// Parameters returns nil (no trainable parameters).
func (tk *TopK[T]) Parameters() []*graph.Parameter[T] { return nil }

// BuildTopK constructs a TopK layer from ZMF attributes.
// Required: "k" (int or int64). Optional: "axis" (int/int64, default -1),
// "largest" (int64, 1=true, 0=false; default 1), "sorted" (int64; default 1).
func BuildTopK[T tensor.Numeric](
	engine compute.Engine[T],
	_ numeric.Arithmetic[T],
	_ string,
	_ map[string]*graph.Parameter[T],
	attributes map[string]interface{},
) (graph.Node[T], error) {
	kAttr, ok := attributes["k"]
	if !ok {
		return nil, errors.New("TopK: missing required attribute 'k'")
	}
	var k int
	switch v := kAttr.(type) {
	case int:
		k = v
	case int64:
		k = int(v)
	default:
		return nil, fmt.Errorf("TopK: attribute 'k' has unsupported type %T", kAttr)
	}

	axis := -1
	if v, ok := attributes["axis"]; ok {
		switch a := v.(type) {
		case int:
			axis = a
		case int64:
			axis = int(a)
		}
	}

	largest := true
	if v, ok := attributes["largest"]; ok {
		if i, ok := v.(int64); ok {
			largest = i != 0
		}
	}

	sorted := true
	if v, ok := attributes["sorted"]; ok {
		if i, ok := v.(int64); ok {
			sorted = i != 0
		}
	}

	return NewTopK(engine, k, axis, largest, sorted), nil
}

// Statically assert that TopK implements graph.Node.
var _ graph.Node[float32] = (*TopK[float32])(nil)
