// Package model provides the core structures and loading mechanisms for Zerfoo models.
package model

import (
	"context"
	"fmt"
	"strconv"
	"strings"

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

	// 1.5. Handle Parameters as nodes (only add them if they don't conflict with layer nodes)
	// We'll add parameters on-demand during the connection phase to avoid conflicts

	// 2. First pass: Instantiate all layer nodes
	for _, nodeProto := range model.Graph.Nodes {
		// Skip if a node with this name already exists (e.g., it's an input or parameter)
		if _, exists := instantiatedNodes[nodeProto.Name]; exists {
			fmt.Printf("DEBUG: Skipping node '%s' (already exists)\n", nodeProto.Name)
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
		
		// Filter out empty input names first
		validInputNames := make([]string, 0, len(nodeProto.Inputs))
		for _, inputName := range nodeProto.Inputs {
			if inputName != "" {
				validInputNames = append(validInputNames, inputName)
			}
		}
		
		inputNodes := make([]graph.Node[T], len(validInputNames))
		
		for i, inputName := range validInputNames {
			inputNode, ok := instantiatedNodes[inputName]
			if !ok {
				// Try to resolve output suffix (e.g., "/path/to/node/output_0" -> "/path/to/node")
				resolvedName := resolveOutputSuffix(inputName)
				if resolvedName != inputName {
					inputNode, ok = instantiatedNodes[resolvedName]
					if !ok {
						// Try with common layer suffixes (LayerNorm, etc.)
						layerSuffixes := []string{"/LayerNorm", "/SimplifiedLayerNormalization", "/SkipLayerNorm"}
						for _, suffix := range layerSuffixes {
							candidateName := resolvedName + suffix
							if candidate, exists := instantiatedNodes[candidateName]; exists {
								inputNode = candidate
								ok = true
								break
							}
						}
					}
				}
				if !ok {
					// Try to create a parameter node if this input refers to a parameter
					if param, paramExists := params[inputName]; paramExists {
						paramNode := &parameterNode[T]{value: param.Value}
						instantiatedNodes[inputName] = paramNode
						inputNode = paramNode
					} else if param, paramExists := params[resolvedName]; paramExists {
						// Try parameter lookup with resolved name
						paramNode := &parameterNode[T]{value: param.Value}
						instantiatedNodes[resolvedName] = paramNode
						inputNode = paramNode
					} else {
						// Handle special cases like transposed parameters
						baseParamName := strings.TrimSuffix(inputName, "_transposed")
						if baseParamName != inputName {
							if param, paramExists := params[baseParamName]; paramExists {
								paramNode := &parameterNode[T]{value: param.Value}
								instantiatedNodes[inputName] = paramNode
								inputNode = paramNode
							} else {
								return nil, fmt.Errorf("input node '%s' (resolved: '%s') for node '%s' not found", inputName, resolvedName, nodeProto.Name)
							}
						} else {
							return nil, fmt.Errorf("input node '%s' (resolved: '%s') for node '%s' not found", inputName, resolvedName, nodeProto.Name)
						}
					}
				}
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
		// Try to resolve output suffix for the output name
		resolvedOutputName := resolveOutputSuffix(outputNodeName)
		outputNode, ok = instantiatedNodes[resolvedOutputName]
		if !ok {
			// For Gemma models, 'logits' typically maps to the last MatMul node (lm_head)
			if outputNodeName == "logits" {
				if lmHeadNode, exists := instantiatedNodes["/lm_head/MatMul"]; exists {
					outputNode = lmHeadNode
					ok = true
				}
			}
		}
		if !ok {
			return nil, fmt.Errorf("output node '%s' not found in instantiated nodes", outputNodeName)
		}
	}

	return builder.Build(outputNode)
}

// parameterNode is a special node type for parameters that are referenced as inputs.
type parameterNode[T tensor.Numeric] struct {
	value *tensor.TensorNumeric[T]
}

func (n *parameterNode[T]) OutputShape() []int {
	return n.value.Shape()
}

func (n *parameterNode[T]) Forward(ctx context.Context, _ ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	return n.value, nil
}

func (n *parameterNode[T]) Backward(ctx context.Context, outputGradient *tensor.TensorNumeric[T], _ ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	// Parameters don't propagate gradients to inputs since they have no inputs
	return nil, nil
}

func (n *parameterNode[T]) Parameters() []*graph.Parameter[T] {
	return nil
}

// resolveOutputSuffix removes output suffixes like "/output_0", "/output_1", ":0", ":1" etc.
func resolveOutputSuffix(name string) string {
	// Handle /output_N pattern
	if idx := strings.LastIndex(name, "/output_"); idx != -1 {
		return name[:idx]
	}
	// Handle :N pattern
	if idx := strings.LastIndex(name, ":"); idx != -1 {
		// Check if what follows is a number
		if suffix := name[idx+1:]; len(suffix) > 0 {
			if _, err := strconv.Atoi(suffix); err == nil {
				return name[:idx]
			}
		}
	}
	return name
}

// getNodeNames returns a slice of all node names for debugging
func getNodeNames[T tensor.Numeric](nodes map[string]graph.Node[T]) []string {
	names := make([]string, 0, len(nodes))
	for name := range nodes {
		names = append(names, name)
	}
	return names
}

// convertParameters converts the ZMF Tensor map to a map of graph.Parameter.
func convertParameters[T tensor.Numeric](zmfParams map[string]*zmf.Tensor) (map[string]*graph.Parameter[T], error) {
	params := make(map[string]*graph.Parameter[T])
	for name, tensorProto := range zmfParams {
		// Skip non-float tensors, as they are constants handled as attributes.
		if tensorProto.Dtype != zmf.Tensor_FLOAT32 && tensorProto.Dtype != zmf.Tensor_FLOAT64 && tensorProto.Dtype != zmf.Tensor_FLOAT16 && tensorProto.Dtype != zmf.Tensor_BFLOAT16 {
			continue
		}

		tensorValue, err := DecodeTensor[T](tensorProto)
		if err != nil {
			return nil, fmt.Errorf("failed to decode tensor for parameter '%s': %w", name, err)
		}
		newTensorFn := func(shape []int, data []T) (*tensor.TensorNumeric[T], error) {
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
