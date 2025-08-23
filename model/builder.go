// Package model provides the core structures and loading mechanisms for Zerfoo models.
package model

import (
	"context"
	"errors"
	"fmt"
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
//nolint:gocyclo // High-level orchestration with many cases; splitting would harm clarity.
func BuildFromZMF[T tensor.Numeric](
	engine compute.Engine[T],
	ops numeric.Arithmetic[T],
	model *zmf.Model,
) (*graph.Graph[T], error) {
	if model == nil || model.Graph == nil {
		return nil, errors.New("cannot build model from nil or empty ZMF graph")
	}

	params, err := convertParameters[T](model.Graph.Parameters)
	if err != nil {
		return nil, err
	}

	builder := graph.NewBuilder[T](engine)
	instantiatedNodes := make(map[string]graph.Node[T])

	// 1. Handle Graph Inputs
	// These are the entry points to the graph. We create special input nodes for them.
	// Only create input nodes for actual model inputs, not parameters
	for _, inputProto := range model.Graph.Inputs {
		// Skip parameters - they should be embedded in layers, not treated as inputs
		if _, isParam := params[inputProto.Name]; isParam {
			continue
		}
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

		// Special handling for layers with embedded weights/parameters
		actualInputNames := validInputNames
		if gatherLayer, isGather := currentNode.(interface{ HasEmbeddedWeights() bool }); isGather && gatherLayer.HasEmbeddedWeights() {
			// Skip the first input (weights) since it's embedded in the layer
			if len(actualInputNames) > 1 {
				actualInputNames = actualInputNames[1:]
			}
		} else {
			switch nodeProto.OpType {
			case "MatMul":
				if strings.Contains(nodeProto.Name, "lm_head") && len(actualInputNames) > 1 {
					weightInputName := actualInputNames[1]
					if strings.Contains(weightInputName, "embed_tokens") {
						if embedParam, exists := params[weightInputName]; exists {
							transposedTensor, err := engine.Transpose(context.Background(), embedParam.Value, []int{1, 0})
							if err != nil {
								return nil, fmt.Errorf("failed to transpose embedding weights for lm_head: %w", err)
							}
							transposedParam := &graph.Parameter[T]{
								Name:  weightInputName + "_transposed",
								Value: transposedTensor,
							}
							params[weightInputName+"_transposed"] = transposedParam
							actualInputNames[1] = weightInputName + "_transposed"
						}
					}
				}
			case "SimplifiedLayerNormalization", "SkipSimplifiedLayerNormalization":
				if len(actualInputNames) > 1 {
					actualInputNames = actualInputNames[:1]
				}
			case "Reshape":
				if len(actualInputNames) > 1 {
					shapeInputName := actualInputNames[1]
					if shapeParam, exists := params[shapeInputName]; exists {
						shapeValues := make([]int64, shapeParam.Value.Size())
						for i := 0; i < shapeParam.Value.Size(); i++ { //nolint:intrange // classic loop for generic tensor access
							val, err := shapeParam.Value.At(i)
							if err != nil {
								return nil, fmt.Errorf("failed to extract shape value at index %d: %w", i, err)
							}
							shapeValues[i] = int64(val)
						}
						if nodeProto.Attributes == nil {
							nodeProto.Attributes = make(map[string]*zmf.Attribute)
						}
						intsAttr := &zmf.Ints{Val: shapeValues}
						attr := &zmf.Attribute{Value: &zmf.Attribute_Ints{Ints: intsAttr}}
						nodeProto.Attributes["shape"] = attr
					}
					actualInputNames = actualInputNames[:1]
				}
			}
		}

		// Connect inputs
		inputNodes := make([]graph.Node[T], len(actualInputNames))
		for i, inputName := range actualInputNames {
			inputNode, ok := instantiatedNodes[inputName]
			if !ok {
				// Try to resolve with output suffix
				resolvedName := resolveOutputSuffix(inputName, instantiatedNodes)
				if resolvedName != "" {
					inputNode, ok = instantiatedNodes[resolvedName]
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
		return nil, errors.New("graph has no defined outputs")
	}
	outputNodeName := model.Graph.Outputs[0].Name

	outputNode, ok := instantiatedNodes[outputNodeName]
	if !ok {
		// Try to resolve output suffix for the output name
		resolvedOutputName := resolveOutputSuffix(outputNodeName, instantiatedNodes)
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

func (p *parameterNode[T]) OpType() string {
	return "Parameter"
}

func (p *parameterNode[T]) Attributes() map[string]interface{} {
	return make(map[string]interface{})
}

func (p *parameterNode[T]) OutputShape() []int {
	return p.value.Shape()
}

func (p *parameterNode[T]) Forward(_ context.Context, _ ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	return p.value, nil
}

func (p *parameterNode[T]) Backward(_ context.Context, _ *tensor.TensorNumeric[T], _ ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	// Parameters don't propagate gradients to inputs since they have no inputs
	return nil, nil
}

func (p *parameterNode[T]) Parameters() []*graph.Parameter[T] {
	return nil
}

// resolveOutputSuffix removes output suffixes like "/output_0", "/output_1", ":0", ":1" etc.
// and tries to find the actual node name in the nodeMap.
func resolveOutputSuffix[T tensor.Numeric](name string, nodeMap map[string]graph.Node[T]) string {
	// Try stripping common suffixes first, including numbered outputs.
	suffixes := []string{"/output_0", ":0", "/output_1", ":1", "/output_2", ":2", "/output_3", ":3"}
	for _, suffix := range suffixes {
		if strings.HasSuffix(name, suffix) {
			baseName := strings.TrimSuffix(name, suffix)
			if _, exists := nodeMap[baseName]; exists {
				return baseName
			}

			// For patterns like "/model/layers.0/input_layernorm/output_0",
			// try to find the actual layer node by appending common layer suffixes
			layerSuffixes := []string{"/LayerNorm", "/SimplifiedLayerNormalization", "/SkipLayerNorm", "/MatMul", "/Gather", "/Shape", "/Cast", "/Reshape", "/Mul", "/Sub", "/Add", "/Concat", "/Unsqueeze", "/FastGelu"}
			for _, layerSuffix := range layerSuffixes {
				candidateName := baseName + layerSuffix
				if _, exists := nodeMap[candidateName]; exists {
					return candidateName
				}
			}
		}
	}

	// Try common layer name variations (for backward compatibility).
	layerSuffixes := []string{"/LayerNorm", "/SimplifiedLayerNormalization", "/SkipLayerNorm"}
	for _, suffix := range layerSuffixes {
		if strings.HasSuffix(name, suffix) {
			baseName := strings.TrimSuffix(name, suffix)
			if _, exists := nodeMap[baseName]; exists {
				return baseName
			}
		}
	}

	return ""
}

// getNodeNames returns a slice of all node names for debugging.
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
		newTensorFn := tensor.New[T]
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
		case *zmf.Attribute_Ints:
			// Convert int64 array to int array for convenience
			intValues := make([]int64, len(v.Ints.Val))
			copy(intValues, v.Ints.Val)
			attributes[name] = intValues
		case *zmf.Attribute_Floats:
			// Convert float array
			floatValues := make([]float32, len(v.Floats.Val))
			copy(floatValues, v.Floats.Val)
			attributes[name] = floatValues
		case *zmf.Attribute_Strings:
			// Convert string array
			stringValues := make([]string, len(v.Strings.Val))
			copy(stringValues, v.Strings.Val)
			attributes[name] = stringValues
		}
	}

	return attributes
}
