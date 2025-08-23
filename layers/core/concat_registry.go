package core

import (
	"fmt"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/graph"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
)

// BuildConcat constructs a Concat node, extracting the axis from attributes.
func BuildConcat[T tensor.Numeric](
	engine compute.Engine[T],
	_ numeric.Arithmetic[T],
	_ string,
	_ map[string]*graph.Parameter[T],
	attributes map[string]interface{},
) (graph.Node[T], error) {
	axisAttr, ok := attributes["axis"]
	if !ok {
		// Default to axis 0 if not specified
		return NewConcat(engine, 0), nil
	}

	var axis int
	switch v := axisAttr.(type) {
	case int64:
		axis = int(v)
	case int:
		axis = v
	default:
		return nil, fmt.Errorf("unsupported type for 'axis' attribute: %T", axisAttr)
	}

	return NewConcat(engine, axis), nil
}
