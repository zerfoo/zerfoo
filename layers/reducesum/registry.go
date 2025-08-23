package reducesum

import (
	"fmt"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/graph"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
)

// BuildReduceSum constructs a ReduceSum node, parsing axes and keepdims from attributes.
func BuildReduceSum[T tensor.Numeric](
	engine compute.Engine[T],
	_ numeric.Arithmetic[T],
	_ string,
	_ map[string]*graph.Parameter[T],
	attributes map[string]interface{},
) (graph.Node[T], error) {
	var axes []int

	axesAttr, ok := attributes["axes"]
	if !ok {
		// Default behavior: reduce over all axes (empty slice indicates all axes)
		axes = []int{}
	} else {
		switch v := axesAttr.(type) {
		case []int64:
			axes = make([]int, len(v))
			for i, val := range v {
				axes[i] = int(val)
			}
		case []any:
			axes = make([]int, len(v))
			for i, val := range v {
				axes[i] = int(val.(int64))
			}
		default:
			return nil, fmt.Errorf("unsupported type for 'axes' attribute: %T", axesAttr)
		}
	}

	keepDims, ok := attributes["keepdims"].(int64)
	if !ok {
		keepDims = 1
	}

	return New(engine, axes, keepDims == 1), nil
}
