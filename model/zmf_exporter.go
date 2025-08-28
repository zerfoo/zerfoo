// Package model provides the core structures and loading mechanisms for Zerfoo models.
package model

import (
	"fmt"
	"os"

	"github.com/zerfoo/zerfoo/graph"
	"github.com/zerfoo/zerfoo/tensor"
	"github.com/zerfoo/zmf"
	"google.golang.org/protobuf/proto"
)

// NodeSerializer defines the interface for serializing specific node types to ZMF format.
type NodeSerializer[T tensor.Numeric] interface {
	// SerializeNode converts a graph node to ZMF node representation.
	SerializeNode(node graph.Node[T], metadata map[string]interface{}) (*zmf.Node, error)
}

// NodeDeserializer defines the interface for deserializing specific node types from ZMF format.
type NodeDeserializer[T tensor.Numeric] interface {
	// DeserializeNode converts a ZMF node to a graph node.
	DeserializeNode(zmfNode *zmf.Node, params map[string]*graph.Parameter[T]) (graph.Node[T], error)
}

// ZMFExporter implements the Exporter interface for ZMF format.
type ZMFExporter[T tensor.Numeric] struct {
	serializers map[string]NodeSerializer[T]
}

// NewZMFExporter creates a new ZMF exporter with default serializers.
func NewZMFExporter[T tensor.Numeric]() *ZMFExporter[T] {
	return &ZMFExporter[T]{
		serializers: make(map[string]NodeSerializer[T]),
	}
}

// RegisterNodeSerializer registers a custom serializer for a specific node type.
func (e *ZMFExporter[T]) RegisterNodeSerializer(opType string, serializer NodeSerializer[T]) {
	e.serializers[opType] = serializer
}

// Export saves the given model to ZMF format at the specified path.
func (e *ZMFExporter[T]) Export(model *Model[T], path string) error {
	zmfModel, err := e.convertModelToZMF(model)
	if err != nil {
		return fmt.Errorf("failed to convert model to ZMF: %w", err)
	}

	data, err := proto.Marshal(zmfModel)
	if err != nil {
		return fmt.Errorf("failed to marshal ZMF model: %w", err)
	}

	err = os.WriteFile(path, data, 0644)
	if err != nil {
		return fmt.Errorf("failed to write ZMF file: %w", err)
	}

	return nil
}

// convertModelToZMF converts a Zerfoo model to ZMF protobuf format.
func (e *ZMFExporter[T]) convertModelToZMF(model *Model[T]) (*zmf.Model, error) {
	zmfModel := &zmf.Model{
		ZmfVersion: "1.0.0",
		Graph:      &zmf.Graph{},
	}

	if model.ZMFVersion != "" {
		zmfModel.ZmfVersion = model.ZMFVersion
	}

	// Get all nodes using the new introspection API
	allNodes := model.Graph.GetAllNodes()
	dependencies := model.Graph.GetDependencies()

	// Convert parameters
	allParams := model.Graph.Parameters()
	zmfParams := make(map[string]*zmf.Tensor)
	for _, param := range allParams {
		zmfParam, err := e.convertParameterToZMF(param)
		if err != nil {
			return nil, fmt.Errorf("failed to convert parameter: %w", err)
		}
		zmfParams[param.Name] = zmfParam
	}
	zmfModel.Graph.Parameters = zmfParams

	// Convert nodes using introspection metadata
	zmfNodes := make([]*zmf.Node, 0, len(allNodes))
	nodeNameMap := make(map[graph.Node[T]]string) // Map nodes to their names for dependency resolution

	for i, node := range allNodes {
		metadata := model.Graph.GetNodeMetadata(node)
		nodeName := fmt.Sprintf("node_%d", i)
		nodeNameMap[node] = nodeName

		zmfNode, err := e.convertNodeToZMF(node, nodeName, metadata)
		if err != nil {
			return nil, fmt.Errorf("failed to convert node %s: %w", nodeName, err)
		}

		// Add dependencies
		nodeDeps := dependencies[node]
		for _, dep := range nodeDeps {
			if depName, exists := nodeNameMap[dep]; exists {
				zmfNode.Inputs = append(zmfNode.Inputs, depName)
			}
		}

		zmfNodes = append(zmfNodes, zmfNode)
	}
	zmfModel.Graph.Nodes = zmfNodes

	// Set inputs and outputs
	inputs := model.Graph.Inputs()
	for _, input := range inputs {
		if inputName, exists := nodeNameMap[input]; exists {
			zmfModel.Graph.Inputs = append(zmfModel.Graph.Inputs, &zmf.ValueInfo{
				Name:  inputName,
				Shape: int32SliceToInt64Slice(input.OutputShape()),
			})
		}
	}

	output := model.Graph.Output()
	if outputName, exists := nodeNameMap[output]; exists {
		zmfModel.Graph.Outputs = append(zmfModel.Graph.Outputs, &zmf.ValueInfo{
			Name:  outputName,
			Shape: int32SliceToInt64Slice(output.OutputShape()),
		})
	}

	return zmfModel, nil
}

// convertNodeToZMF converts a graph node to ZMF format using registered serializers.
func (e *ZMFExporter[T]) convertNodeToZMF(node graph.Node[T], name string, metadata map[string]interface{}) (*zmf.Node, error) {
	opType := metadata["op_type"].(string)

	// Use custom serializer if available
	if serializer, exists := e.serializers[opType]; exists {
		return serializer.SerializeNode(node, metadata)
	}

	// Default serialization
	zmfNode := &zmf.Node{
		Name:   name,
		OpType: opType,
	}

	// Convert attributes
	if attrs, ok := metadata["attributes"].(map[string]interface{}); ok && attrs != nil {
		zmfNode.Attributes = make(map[string]*zmf.Attribute)
		for key, value := range attrs {
			attr := &zmf.Attribute{}
			switch v := value.(type) {
			case string:
				attr.Value = &zmf.Attribute_S{S: v}
			case int:
				attr.Value = &zmf.Attribute_I{I: int64(v)}
			case int64:
				attr.Value = &zmf.Attribute_I{I: v}
			case float32:
				attr.Value = &zmf.Attribute_F{F: v}
			case float64:
				attr.Value = &zmf.Attribute_F{F: float32(v)}
			case bool:
				attr.Value = &zmf.Attribute_B{B: v}
			default:
				attr.Value = &zmf.Attribute_S{S: fmt.Sprintf("%v", value)}
			}
			zmfNode.Attributes[key] = attr
		}
	}

	// Add parameter references as inputs
	params := node.Parameters()
	for _, param := range params {
		zmfNode.Inputs = append(zmfNode.Inputs, param.Name)
	}

	return zmfNode, nil
}

// convertParameterToZMF converts a graph parameter to ZMF format.
func (e *ZMFExporter[T]) convertParameterToZMF(param *graph.Parameter[T]) (*zmf.Tensor, error) {
	tensor := param.Value
	
	zmfTensor := &zmf.Tensor{
		Dtype: getZMFDataType[T](),
		Shape: int32SliceToInt64Slice(tensor.Shape()),
	}

	// Serialize tensor data
	data, err := serializeTensorData(tensor)
	if err != nil {
		return nil, fmt.Errorf("failed to serialize tensor data: %w", err)
	}
	zmfTensor.Data = data

	return zmfTensor, nil
}

// Helper functions
func int32SliceToInt64Slice(slice []int) []int64 {
	result := make([]int64, len(slice))
	for i, v := range slice {
		result[i] = int64(v)
	}
	return result
}

func getZMFDataType[T tensor.Numeric]() zmf.Tensor_DataType {
	var zero T
	switch any(zero).(type) {
	case float32:
		return zmf.Tensor_FLOAT32
	case float64:
		return zmf.Tensor_FLOAT64
	case int:
		return zmf.Tensor_INT32
	case int32:
		return zmf.Tensor_INT32
	case int64:
		return zmf.Tensor_INT64
	default:
		return zmf.Tensor_FLOAT32 // Default fallback
	}
}

func serializeTensorData[T tensor.Numeric](t *tensor.TensorNumeric[T]) ([]byte, error) {
	// This is a simplified implementation - in practice you'd want proper binary serialization
	data := t.Data()
	result := make([]byte, len(data)*4) // Assuming 4 bytes per element for simplicity
	
	for i, val := range data {
		// Convert to bytes - this is a placeholder implementation
		bytes := []byte(fmt.Sprintf("%v", val))
		if i*4+len(bytes) <= len(result) {
			copy(result[i*4:], bytes)
		}
	}
	
	return result, nil
}
