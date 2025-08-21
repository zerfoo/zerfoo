// Package model provides the core structures and loading mechanisms for Zerfoo models.
package model

import (
	"fmt"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/graph"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
	"github.com/zerfoo/zerfoo/zmf"
)

// BuildFromZMF constructs a Zerfoo model from a ZMF graph definition.
// This function iterates through the nodes in the graph, instantiates the
// corresponding layers using a registered builder, and connects them.
func BuildFromZMF[T tensor.Numeric](
	engine compute.Engine[T],
	ops numeric.Arithmetic[T],
	model *zmf.Model,
) (map[string]graph.Node[T], error) {
	if model == nil || model.Graph == nil {
		return nil, fmt.Errorf("cannot build model from nil or empty ZMF graph")
	}

	// Convert ZMF parameters to graph.Parameter
	params, err := convertParameters[T](model.Graph.Parameters)
	if err != nil {
		return nil, err
	}

	instantiatedNodes := make(map[string]graph.Node[T])

	// Iterate through the nodes in the ZMF graph and build them.
	// We assume the nodes are in a valid topological order for now.
	for _, nodeProto := range model.Graph.Nodes {
		builder, err := GetLayerBuilder[T](nodeProto.OpType)
		if err != nil {
			return nil, err
		}

		// Convert ZMF attributes to a more usable map[string]interface{}
		attributes := make(map[string]interface{})
		for name, attr := range nodeProto.Attributes {
			switch v := attr.Value.(type) {
			case *zmf.Attribute_F:
				attributes[name] = v.F
			case *zmf.Attribute_I:
				attributes[name] = int(v.I) // Cast to int for convenience
			case *zmf.Attribute_S:
				attributes[name] = v.S
			}
		}

		node, err := builder(engine, ops, nodeProto.Name, params, attributes)
		if err != nil {
			return nil, fmt.Errorf("failed to build node '%s' of type '%s': %w", nodeProto.Name, nodeProto.OpType, err)
		}

		instantiatedNodes[nodeProto.Name] = node
	}

	return instantiatedNodes, nil
}

// convertParameters converts the ZMF Tensor map to a map of graph.Parameter.
func convertParameters[T tensor.Numeric](zmfParams map[string]*zmf.Tensor) (map[string]*graph.Parameter[T], error) {
	params := make(map[string]*graph.Parameter[T])
	for name, tensorProto := range zmfParams {
		tensor, err := DecodeTensor[T](tensorProto)
		if err != nil {
			return nil, fmt.Errorf("failed to decode tensor for parameter '%s': %w", name, err)
		}

		param, err := graph.NewParameter[T](name, tensor, nil)
		if err != nil {
			return nil, fmt.Errorf("failed to create parameter '%s': %w", name, err)
		}
		params[name] = param
	}
	return params, nil
}
