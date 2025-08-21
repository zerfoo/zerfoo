// Package normalization provides various normalization layers for neural networks.
package normalization

import (
	"fmt"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/graph"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
)

func BuildRMSNorm[T tensor.Numeric](
	engine compute.Engine[T],
	ops numeric.Arithmetic[T],
	name string,
	params map[string]*graph.Parameter[T],
	attributes map[string]interface{},
) (graph.Node[T], error) {
	gain, ok := params[name+"_gain"]
	if !ok {
		return nil, fmt.Errorf("missing required parameter: %s_gain", name)
	}

	epsilonAttr, ok := attributes["epsilon"]
	if !ok {
		return nil, fmt.Errorf("missing required attribute: epsilon")
	}

	var epsilon float64
	switch v := epsilonAttr.(type) {
	case float64:
		epsilon = v
	case float32:
		epsilon = float64(v)
	default:
		return nil, fmt.Errorf("attribute 'epsilon' has incorrect type: expected float64 or float32, got %T", epsilonAttr)
	}

	return NewRMSNormFromParam(engine, ops, ops.FromFloat64(epsilon), gain)
}

func BuildSimplifiedLayerNormalization[T tensor.Numeric](
	engine compute.Engine[T],
	ops numeric.Arithmetic[T],
	name string,
	params map[string]*graph.Parameter[T],
	attributes map[string]interface{},
) (graph.Node[T], error) {
	gain, ok := params[name+"_gain"]
	if !ok {
		return nil, fmt.Errorf("missing required parameter: %s_gain", name)
	}

	epsilonAttr, ok := attributes["epsilon"]
	if !ok {
		return nil, fmt.Errorf("missing required attribute: epsilon")
	}

	var epsilon float64
	switch v := epsilonAttr.(type) {
	case float64:
		epsilon = v
	case float32:
		epsilon = float64(v)
	default:
		return nil, fmt.Errorf("attribute 'epsilon' has incorrect type: expected float64 or float32, got %T", epsilonAttr)
	}

	return NewSimplifiedLayerNormalization(engine, ops, gain.Value, T(epsilon))
}

func BuildSkipSimplifiedLayerNormalization[T tensor.Numeric](
	engine compute.Engine[T],
	ops numeric.Arithmetic[T],
	name string,
	params map[string]*graph.Parameter[T],
	attributes map[string]interface{},
) (graph.Node[T], error) {
	gain, ok := params[name+"_gain"]
	if !ok {
		return nil, fmt.Errorf("missing required parameter: %s_gain", name)
	}

	epsilonAttr, ok := attributes["epsilon"]
	if !ok {
		return nil, fmt.Errorf("missing required attribute: epsilon")
	}

	var epsilon float64
	switch v := epsilonAttr.(type) {
	case float64:
		epsilon = v
	case float32:
		epsilon = float64(v)
	default:
		return nil, fmt.Errorf("attribute 'epsilon' has incorrect type: expected float64 or float32, got %T", epsilonAttr)
	}

	return NewSkipSimplifiedLayerNormalization(engine, ops, gain.Value, T(epsilon))
}
