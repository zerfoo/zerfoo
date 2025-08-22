package core

import (
	"fmt"
	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/graph"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
)

func BuildReshape[T tensor.Numeric](
	engine compute.Engine[T],
	_ numeric.Arithmetic[T],
	name string,
	_ map[string]*graph.Parameter[T],
	attributes map[string]interface{},
) (graph.Node[T], error) {
	var shape []int
	
	
	shapeAttr, ok := attributes["shape"]
	if !ok {
		// Try to find shape in constant attributes (ZMF pattern)
		// Look for attributes that contain shape information
		for key, value := range attributes {
			if key != "shape" {
				// Try to extract shape from constant attributes
				if shapeValues, isSlice := value.([]int64); isSlice {
					shape = make([]int, len(shapeValues))
					for i, val := range shapeValues {
						shape[i] = int(val)
					}
					break
				} else if shapeValues, isSlice := value.([]interface{}); isSlice {
					shape = make([]int, len(shapeValues))
					for i, val := range shapeValues {
						if intVal, ok := val.(int64); ok {
							shape[i] = int(intVal)
						} else if intVal, ok := val.(int); ok {
							shape[i] = intVal
						}
					}
					break
				}
			}
		}
		
		if len(shape) == 0 {
			// Default behavior: return identity reshape (will be determined at runtime)
			shape = []int{-1} // -1 means infer from input
		}
	} else {
		switch v := shapeAttr.(type) {
		case []int64:
			shape = make([]int, len(v))
			for i, val := range v {
				shape[i] = int(val)
			}
		case []any:
			shape = make([]int, len(v))
			for i, val := range v {
				shape[i] = int(val.(int64))
			}
		default:
			return nil, fmt.Errorf("unsupported type for 'shape' attribute: %T", shapeAttr)
		}
	}

	return NewReshape(engine, shape), nil
}
