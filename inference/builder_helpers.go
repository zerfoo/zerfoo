package inference

import (
	"fmt"

	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/tensor"
)

// tensorLookup wraps a tensor map to provide consistent lookup semantics
// across architecture builders. It replaces the per-builder closure pattern:
//
//	lookup := func(name string) (*tensor.TensorNumeric[float32], error) {
//	    t, ok := tensors[name]
//	    if !ok { return nil, fmt.Errorf("missing tensor %q", name) }
//	    return t, nil
//	}
type tensorLookup[T tensor.Numeric] struct {
	tensors map[string]*tensor.TensorNumeric[T]
}

// newTensorLookup creates a tensorLookup from a tensor map.
func newTensorLookup[T tensor.Numeric](tensors map[string]*tensor.TensorNumeric[T]) tensorLookup[T] {
	return tensorLookup[T]{tensors: tensors}
}

// Lookup returns the tensor with the given name, or an error if not found.
func (tl tensorLookup[T]) Lookup(name string) (*tensor.TensorNumeric[T], error) {
	t, ok := tl.tensors[name]
	if !ok {
		return nil, fmt.Errorf("missing tensor %q", name)
	}
	return t, nil
}

// Optional returns the tensor with the given name and true if it exists,
// or nil and false if not found. This supports the common pattern of
// looking up tensors that may or may not be present (e.g., tied weights,
// optional biases).
func (tl tensorLookup[T]) Optional(name string) (*tensor.TensorNumeric[T], bool) {
	t, ok := tl.tensors[name]
	return t, ok
}

// Has returns true if the tensor map contains the given name.
func (tl tensorLookup[T]) Has(name string) bool {
	_, ok := tl.tensors[name]
	return ok
}

// paramWrapper wraps tensor values into graph parameters. It replaces the
// per-builder closure pattern:
//
//	param := func(name string, t *tensor.TensorNumeric[float32]) *graph.Parameter[float32] {
//	    return &graph.Parameter[float32]{Name: name, Value: t}
//	}
type paramWrapper[T tensor.Numeric] struct{}

// newParamWrapper creates a paramWrapper.
func newParamWrapper[T tensor.Numeric]() paramWrapper[T] {
	return paramWrapper[T]{}
}

// Wrap creates a graph.Parameter from a name and tensor value.
func (pw paramWrapper[T]) Wrap(name string, t *tensor.TensorNumeric[T]) *graph.Parameter[T] {
	return &graph.Parameter[T]{Name: name, Value: t}
}
