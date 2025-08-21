// Package normalization provides various normalization layers for neural networks.
package normalization

import (
	"fmt"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/graph"
	"github.com/zerfoo/zerfoo/model"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
)

func init() {
	model.RegisterLayer("RMSNorm", buildRMSNorm[tensor.Float16])
	// Add registrations for other supported types like float32 if needed.
}

func buildRMSNorm[T tensor.Numeric](
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

	epsilon, ok := epsilonAttr.(float64)
	if !ok {
		return nil, fmt.Errorf("attribute 'epsilon' has incorrect type: expected float64, got %T", epsilonAttr)
	}

	return NewRMSNormFromParam(engine, ops, ops.FromFloat64(epsilon), gain)
}
