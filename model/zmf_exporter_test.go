package model

import (
	"context"
	"os"
	"testing"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/graph"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
	"github.com/zerfoo/zerfoo/types"
	"github.com/zerfoo/zmf"
)

// mockNode for testing
type mockNode struct {
	name        string
	opType      string
	outputShape []int
	attributes  map[string]interface{}
	params      []*graph.Parameter[int]
}

func (m *mockNode) OpType() string                      { return m.opType }
func (m *mockNode) OutputShape() []int                  { return m.outputShape }
func (m *mockNode) Attributes() map[string]interface{}  { return m.attributes }
func (m *mockNode) Parameters() []*graph.Parameter[int] { return m.params }
func (m *mockNode) Forward(_ context.Context, inputs ...*tensor.TensorNumeric[int]) (*tensor.TensorNumeric[int], error) {
	return inputs[0], nil
}

func (m *mockNode) Backward(_ context.Context, mode types.BackwardMode, outputGradient *tensor.TensorNumeric[int], _ ...*tensor.TensorNumeric[int]) ([]*tensor.TensorNumeric[int], error) {
	return []*tensor.TensorNumeric[int]{outputGradient}, nil
}

// mockNodeSerializer for testing custom serialization
type mockNodeSerializer struct{}

func (s *mockNodeSerializer) SerializeNode(node graph.Node[int], metadata map[string]interface{}) (*zmf.Node, error) {
	return &zmf.Node{
		Name:   "custom_serialized_node",
		OpType: "CustomOp",
		Attributes: map[string]*zmf.Attribute{
			"custom": {Value: &zmf.Attribute_S{S: "true"}},
		},
	}, nil
}

func TestNewZMFExporter(t *testing.T) {
	exporter := NewZMFExporter[int]()
	if exporter == nil {
		t.Fatal("expected non-nil exporter")
	}
	if exporter.serializers == nil {
		t.Error("expected initialized serializers map")
	}
}

func TestZMFExporter_RegisterNodeSerializer(t *testing.T) {
	exporter := NewZMFExporter[int]()
	serializer := &mockNodeSerializer{}

	exporter.RegisterNodeSerializer("TestOp", serializer)

	if len(exporter.serializers) != 1 {
		t.Errorf("expected 1 serializer, got %d", len(exporter.serializers))
	}

	registered, exists := exporter.serializers["TestOp"]
	if !exists {
		t.Error("expected TestOp serializer to be registered")
	}
	if registered != serializer {
		t.Error("registered serializer doesn't match original")
	}
}

func TestZMFExporter_Export(t *testing.T) {
	// Create a simple model for testing
	engine := compute.NewCPUEngine[int](numeric.IntOps{})
	builder := graph.NewBuilder[int](engine)

	// Create input node
	inputNode := builder.Input([]int{2, 2})

	// Create a mock node with parameters
	paramTensor, _ := tensor.New[int]([]int{2, 2}, []int{1, 2, 3, 4})
	param, _ := graph.NewParameter("test_param", paramTensor, tensor.New[int])

	mockNode := &mockNode{
		name:        "test_node",
		opType:      "MockOp",
		outputShape: []int{2, 2},
		attributes: map[string]interface{}{
			"test_attr": "test_value",
		},
		params: []*graph.Parameter[int]{param},
	}

	builder.AddNode(mockNode, inputNode)

	// Build graph
	g, err := builder.Build(mockNode)
	if err != nil {
		t.Fatalf("failed to build graph: %v", err)
	}

	// Create model (without embedding for simplicity)
	model := &Model[int]{
		Graph:      g,
		ZMFVersion: "1.0.0",
	}

	// Create exporter and export
	exporter := NewZMFExporter[int]()
	tempFile := "test_model.zmf"
	defer func() {
		if err := os.Remove(tempFile); err != nil {
			t.Logf("failed to remove temp file %s: %v", tempFile, err)
		}
	}()

	err = exporter.Export(model, tempFile)
	if err != nil {
		t.Fatalf("failed to export model: %v", err)
	}

	// Verify file was created
	if _, err := os.Stat(tempFile); os.IsNotExist(err) {
		t.Error("exported file does not exist")
	}

	// Load and verify the exported model
	zmfModel, err := LoadZMF(tempFile)
	if err != nil {
		t.Fatalf("failed to load exported ZMF: %v", err)
	}

	if zmfModel.ZmfVersion != "1.0.0" {
		t.Errorf("expected ZMF version '1.0.0', got '%s'", zmfModel.ZmfVersion)
	}

	if len(zmfModel.Graph.Nodes) != 2 { // input + mock node
		t.Errorf("expected 2 nodes, got %d", len(zmfModel.Graph.Nodes))
	}

	if len(zmfModel.Graph.Parameters) != 1 {
		t.Errorf("expected 1 parameter, got %d", len(zmfModel.Graph.Parameters))
	}
}

func TestZMFExporter_CustomSerializer(t *testing.T) {
	// Create a simple model
	engine := compute.NewCPUEngine[int](numeric.IntOps{})
	builder := graph.NewBuilder[int](engine)

	inputNode := builder.Input([]int{2, 2})

	mockNode := &mockNode{
		name:        "custom_node",
		opType:      "CustomOp",
		outputShape: []int{2, 2},
		attributes:  map[string]interface{}{},
		params:      []*graph.Parameter[int]{},
	}

	builder.AddNode(mockNode, inputNode)

	g, err := builder.Build(mockNode)
	if err != nil {
		t.Fatalf("failed to build graph: %v", err)
	}

	model := &Model[int]{
		Graph: g,
	}

	// Create exporter with custom serializer
	exporter := NewZMFExporter[int]()
	exporter.RegisterNodeSerializer("CustomOp", &mockNodeSerializer{})

	tempFile := "test_custom_model.zmf"
	defer func() {
		if err := os.Remove(tempFile); err != nil {
			t.Logf("failed to remove temp file %s: %v", tempFile, err)
		}
	}()

	err = exporter.Export(model, tempFile)
	if err != nil {
		t.Fatalf("failed to export model with custom serializer: %v", err)
	}

	// Verify custom serialization was used
	zmfModel, err := LoadZMF(tempFile)
	if err != nil {
		t.Fatalf("failed to load exported ZMF: %v", err)
	}

	// Find the custom node
	var customNode *zmf.Node
	for _, node := range zmfModel.Graph.Nodes {
		if node.OpType == "CustomOp" {
			customNode = node
			break
		}
	}

	if customNode == nil {
		t.Fatal("custom node not found in exported model")
	}

	if customNode.Name != "custom_serialized_node" {
		t.Errorf("expected custom node name 'custom_serialized_node', got '%s'", customNode.Name)
	}

	if customNode.Attributes["custom"].GetS() != "true" {
		t.Errorf("expected custom attribute 'custom'='true', got '%s'", customNode.Attributes["custom"].GetS())
	}
}

func TestConvertParameterToZMF(t *testing.T) {
	exporter := NewZMFExporter[int]()

	// Create a test parameter
	tensorData := []int{1, 2, 3, 4}
	testTensor, err := tensor.New[int]([]int{2, 2}, tensorData)
	if err != nil {
		t.Fatalf("failed to create test tensor: %v", err)
	}

	param, err := graph.NewParameter("test_param", testTensor, tensor.New[int])
	if err != nil {
		t.Fatalf("failed to create test parameter: %v", err)
	}

	zmfParam, err := exporter.convertParameterToZMF(param)
	if err != nil {
		t.Fatalf("failed to convert parameter to ZMF: %v", err)
	}

	// Note: zmfParam is now a Tensor, not a Parameter, so we don't check the name here
	// The name is stored as the key in the parameters map

	if len(zmfParam.Shape) != 2 || zmfParam.Shape[0] != 2 || zmfParam.Shape[1] != 2 {
		t.Errorf("expected shape [2, 2], got %v", zmfParam.Shape)
	}

	if zmfParam.Dtype != zmf.Tensor_INT32 {
		t.Errorf("expected data type INT32, got %v", zmfParam.Dtype)
	}
}

func TestGetZMFDataType(t *testing.T) {
	tests := []struct {
		name     string
		expected zmf.Tensor_DataType
	}{
		{"float32", zmf.Tensor_FLOAT32},
		{"float64", zmf.Tensor_FLOAT64},
		{"int32", zmf.Tensor_INT32},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			switch tt.name {
			case "float32":
				result := getZMFDataType[float32]()
				if result != tt.expected {
					t.Errorf("expected %v, got %v", tt.expected, result)
				}
			case "float64":
				result := getZMFDataType[float64]()
				if result != tt.expected {
					t.Errorf("expected %v, got %v", tt.expected, result)
				}
			case "int32":
				result := getZMFDataType[int32]()
				if result != tt.expected {
					t.Errorf("expected %v, got %v", tt.expected, result)
				}
			}
		})
	}
}

func TestInt32SliceToInt64Slice(t *testing.T) {
	input := []int{1, 2, 3, 4}
	expected := []int64{1, 2, 3, 4}

	result := int32SliceToInt64Slice(input)

	if len(result) != len(expected) {
		t.Errorf("expected length %d, got %d", len(expected), len(result))
	}

	for i, v := range result {
		if v != expected[i] {
			t.Errorf("expected %d at index %d, got %d", expected[i], i, v)
		}
	}
}
