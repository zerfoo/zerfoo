package core

import (
	"fmt"
	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/graph"
	"github.com/zerfoo/zerfoo/model"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
)

func init() {
	model.RegisterLayer("Reshape", BuildReshape[float32])
}

func BuildReshape[T tensor.Numeric](
	engine compute.Engine[T],
	_ numeric.Arithmetic[T],
	_ string,
	_ map[string]*graph.Parameter[T],
	attributes map[string]interface{},
) (graph.Node[T], error) {
	var shape []int
	
	shapeAttr, ok := attributes["shape"]
	if !ok {
		// Default behavior: return identity reshape (will be determined at runtime)
		shape = []int{-1} // -1 means infer from input
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
