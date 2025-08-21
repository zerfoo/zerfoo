// Package model provides the core structures and loading mechanisms for Zerfoo models.
package model

import (
	"fmt"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/graph"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
	"github.com/zerfoo/zmf"
)

// BuildFromZMF constructs a Zerfoo computation graph from a ZMF model definition.
// This function iterates through the nodes in the graph, instantiates the
// corresponding layers using a registered builder, and connects them into an
// executable graph.
func BuildFromZMF[T tensor.Numeric](
	engine compute.Engine[T],
	ops numeric.Arithmetic[T],
	model *zmf.Model,
) (*graph.Graph[T], error) {
	if model == nil || model.Graph == nil {
		return nil, fmt.Errorf("cannot build model from nil or empty ZMF graph")
	}

	params, err := convertParameters[T](model.Graph.Parameters)
	if err != nil {
		return nil, err
	}

	builder := graph.NewBuilder[T](engine)
	instantiatedNodes := make(map[string]graph.Node[T])

	// 1. Handle Graph Inputs
	// These are the entry points to the graph. We create special input nodes for them.
	for _, inputProto := range model.Graph.Inputs {
		dims := make([]int, len(inputProto.Shape))
		for i, dim := range inputProto.Shape {
			dims[i] = int(dim) // Convert int64 to int
		}
		instantiatedNodes[inputProto.Name] = builder.Input(dims)
	}

	// 2. First pass: Instantiate all layer nodes
	for _, nodeProto := range model.Graph.Nodes {
		// Skip if a node with this name already exists (e.g., it's an input)
		if _, exists := instantiatedNodes[nodeProto.Name]; exists {
			continue
		}
		layerBuilder, err := GetLayerBuilder[T](nodeProto.OpType)
		if err != nil {
			return nil, err
		}
		attributes := convertAttributes(nodeProto.Attributes)
		node, err := layerBuilder(engine, ops, nodeProto.Name, params, attributes)
		if err != nil {
			return nil, fmt.Errorf("failed to build node '%s' of type '%s': %w", nodeProto.Name, nodeProto.OpType, err)
		}
		instantiatedNodes[nodeProto.Name] = node
	}

	// 3. Second pass: Connect the nodes
	for _, nodeProto := range model.Graph.Nodes {
		currentNode := instantiatedNodes[nodeProto.Name]
		inputNodes := make([]graph.Node[T], len(nodeProto.Inputs))
		for i, inputName := range nodeProto.Inputs {
			inputNode, ok := instantiatedNodes[inputName]
			if !ok {
				return nil, fmt.Errorf("input node '%s' for node '%s' not found", inputName, nodeProto.Name)
			}
			inputNodes[i] = inputNode
		}
		builder.AddNode(currentNode, inputNodes...)
	}

	// 4. Identify the final output node of the graph.
	if len(model.Graph.Outputs) == 0 {
		return nil, fmt.Errorf("graph has no defined outputs")
	}
	outputNodeName := model.Graph.Outputs[0].Name
	outputNode, ok := instantiatedNodes[outputNodeName]
	if !ok {
		return nil, fmt.Errorf("output node '%s' not found in instantiated nodes", outputNodeName)
	}

	return builder.Build(outputNode)
}

// convertParameters converts the ZMF Tensor map to a map of graph.Parameter.
func convertParameters[T tensor.Numeric](zmfParams map[string]*zmf.Tensor) (map[string]*graph.Parameter[T], error) {
	params := make(map[string]*graph.Parameter[T])
	for name, tensorProto := range zmfParams {
		tensorValue, err := DecodeTensor[T](tensorProto)
		if err != nil {
			return nil, fmt.Errorf("failed to decode tensor for parameter '%s': %w", name, err)
		}
		newTensorFn := func(shape []int, data []T) (*tensor.Tensor[T], error) {
			return tensor.New(shape, data)
		}
		param, err := graph.NewParameter[T](name, tensorValue, newTensorFn)
		if err != nil {
			return nil, fmt.Errorf("failed to create parameter '%s': %w", name, err)
		}
		params[name] = param
	}
	return params, nil
}

// convertAttributes converts ZMF attributes to a more usable map[string]interface{}.
func convertAttributes(zmfAttributes map[string]*zmf.Attribute) map[string]interface{} {
	attributes := make(map[string]interface{})
	for name, attr := range zmfAttributes {
		switch v := attr.Value.(type) {
		case *zmf.Attribute_F:
			attributes[name] = v.F
		case *zmf.Attribute_I:
			attributes[name] = int(v.I) // Cast to int for convenience
		case *zmf.Attribute_S:
			attributes[name] = v.S
		}
	}
	return attributes
}
