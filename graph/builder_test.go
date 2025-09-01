package graph

import (
	"context"
	"testing"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
	"github.com/zerfoo/zerfoo/types"
)

type mockNode struct {
	name         string
	outputShape  []int
	forwardFunc  func(ctx context.Context, inputs ...*tensor.TensorNumeric[int]) (*tensor.TensorNumeric[int], error)
	backwardFunc func(ctx context.Context, mode types.BackwardMode, outputGradient *tensor.TensorNumeric[int]) ([]*tensor.TensorNumeric[int], error)
	params       []*Parameter[int]
	capturedMode types.BackwardMode
}

func (m *mockNode) OutputShape() []int { return m.outputShape }
func (m *mockNode) Forward(ctx context.Context, inputs ...*tensor.TensorNumeric[int]) (*tensor.TensorNumeric[int], error) {
	if m.forwardFunc != nil {
		return m.forwardFunc(ctx, inputs...)
	}

	return inputs[0], nil
}

func (m *mockNode) Backward(ctx context.Context, mode types.BackwardMode, outputGradient *tensor.TensorNumeric[int], _ ...*tensor.TensorNumeric[int]) ([]*tensor.TensorNumeric[int], error) {
	m.capturedMode = mode
	if m.backwardFunc != nil {
		return m.backwardFunc(ctx, mode, outputGradient)
	}

	return []*tensor.TensorNumeric[int]{outputGradient}, nil
}

func (m *mockNode) Parameters() []*Parameter[int] {
	return m.params
}

func (m *mockNode) OpType() string {
	return "mock"
}

func (m *mockNode) Attributes() map[string]interface{} {
	attrs := make(map[string]interface{})
	attrs["name"] = m.name
	return attrs
}

func TestBuilder_Build(t *testing.T) {
	var engine compute.Engine[int] = compute.NewCPUEngine[int](numeric.IntOps{})

	builder := NewBuilder[int](engine)

	inputNode := builder.Input([]int{2, 2})

	node1 := &mockNode{
		name: "node1",
	}
	node2 := &mockNode{
		name: "node2",
	}

	builder.AddNode(node1, inputNode)
	builder.AddNode(node2, node1)

	graph, err := builder.Build(node2)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(builder.Parameters()) != 0 {
		t.Errorf("expected 0 params, got %d", len(builder.Parameters()))
	}

	input, _ := tensor.New[int]([]int{2, 2}, []int{1, 2, 3, 4})

	output, _ := graph.Forward(context.Background(), input)
	if output.Data()[0] != 1 {
		t.Errorf("expected 1, got %d", output.Data()[0])
	}

	initialGradient, _ := tensor.New[int]([]int{2, 2}, []int{1, 1, 1, 1})
	_ = graph.Backward(context.Background(), types.FullBackprop, initialGradient)
}

func TestGraph_Backward_Mode(t *testing.T) {
	var engine compute.Engine[int] = compute.NewCPUEngine[int](numeric.IntOps{})
	builder := NewBuilder[int](engine)

	input := builder.Input([]int{1})
	mock := &mockNode{name: "mock"}
	builder.AddNode(mock, input)

	graph, err := builder.Build(mock)
	if err != nil {
		t.Fatalf("Build() failed: %v", err)
	}

	initialGradient, _ := tensor.New[int]([]int{1}, []int{1})

	t.Run("FullBackprop", func(t *testing.T) {
		err := graph.Backward(context.Background(), types.FullBackprop, initialGradient)
		if err != nil {
			t.Fatalf("Backward() failed: %v", err)
		}
		if mock.capturedMode != types.FullBackprop {
			t.Errorf("expected mode %v, got %v", types.FullBackprop, mock.capturedMode)
		}
	})

	t.Run("OneStepApproximation", func(t *testing.T) {
		err := graph.Backward(context.Background(), types.OneStepApproximation, initialGradient)
		if err != nil {
			t.Fatalf("Backward() failed: %v", err)
		}
		if mock.capturedMode != types.OneStepApproximation {
			t.Errorf("expected mode %v, got %v", types.OneStepApproximation, mock.capturedMode)
		}
	})
}

func TestBuilder_Input(t *testing.T) {
	var engine compute.Engine[int] = compute.NewCPUEngine[int](numeric.IntOps{})

	builder := NewBuilder[int](engine)

	inputNode := builder.Input([]int{2, 2})
	if inputNode == nil {
		t.Fatal("input node should not be nil")
	}

	if inputNode.OutputShape()[0] != 2 {
		t.Errorf("expected output shape to be [2, 2], got %v", inputNode.OutputShape())
	}

	input, _ := tensor.New[int]([]int{2, 2}, []int{1, 2, 3, 4})

	output, _ := inputNode.Forward(context.Background(), input)
	if output != nil {
		t.Errorf("expected nil, got %v", output)
	}

	_, _ = inputNode.Backward(context.Background(), types.FullBackprop, nil)
	if inputNode.Parameters() != nil {
		t.Errorf("expected nil parameters, got %v", inputNode.Parameters())
	}
}

func TestBuilder_Build_Error(t *testing.T) {
	var engine compute.Engine[int] = compute.NewCPUEngine[int](numeric.IntOps{})

	builder := NewBuilder[int](engine)
	builder.nodes = []Node[int]{&mockNode{name: "a"}, &mockNode{name: "b"}}
	// This test is no longer valid as topologicalSortFn is not a field anymore.
	// I will create a cycle to test the error.
	node1 := &mockNode{name: "node1"}
	node2 := &mockNode{name: "node2"}
	builder.AddNode(node1, node2)
	builder.AddNode(node2, node1)

	_, err := builder.Build(node1)
	if err == nil {
		t.Fatal("expected error, got nil")
	}
}

func TestGraph_GetTopologicalOrder_CycleError(t *testing.T) {
	var engine compute.Engine[int] = compute.NewCPUEngine[int](numeric.IntOps{})
	builder := NewBuilder[int](engine)

	node1 := &mockNode{name: "node1"}
	node2 := &mockNode{name: "node2"}
	// Create a cycle
	builder.AddNode(node1, node2)
	builder.AddNode(node2, node1)

	// This should fail at build time, but let's test the topological sort directly
	nodes := []Node[int]{node1, node2}
	deps := map[Node[int]][]Node[int]{
		node1: {node2},
		node2: {node1},
	}

	_, err := topologicalSort(nodes, deps)
	if err == nil {
		t.Error("expected cycle detection error, got nil")
	}
}

func TestGraph_GetNodeMetadata(t *testing.T) {
	var engine compute.Engine[int] = compute.NewCPUEngine[int](numeric.IntOps{})
	builder := NewBuilder[int](engine)

	inputNode := builder.Input([]int{2, 2})
	mockNode := &mockNode{
		name:        "test_node",
		outputShape: []int{3, 3},
		params:      []*Parameter[int]{{}, {}}, // 2 parameters
	}
	builder.AddNode(mockNode, inputNode)

	graph, err := builder.Build(mockNode)
	if err != nil {
		t.Fatalf("Build() failed: %v", err)
	}

	metadata := graph.GetNodeMetadata(mockNode)
	if metadata["op_type"] != "mock" {
		t.Errorf("expected op_type 'mock', got %v", metadata["op_type"])
	}
	if len(metadata["output_shape"].([]int)) != 2 || metadata["output_shape"].([]int)[0] != 3 {
		t.Errorf("expected output_shape [3, 3], got %v", metadata["output_shape"])
	}
	if metadata["parameter_count"] != 2 {
		t.Errorf("expected parameter_count 2, got %v", metadata["parameter_count"])
	}
	attrs := metadata["attributes"].(map[string]interface{})
	if attrs["name"] != "test_node" {
		t.Errorf("expected name 'test_node', got %v", attrs["name"])
	}
}

func TestGraph_GetDependencies(t *testing.T) {
	var engine compute.Engine[int] = compute.NewCPUEngine[int](numeric.IntOps{})
	builder := NewBuilder[int](engine)

	inputNode := builder.Input([]int{2, 2})
	node1 := &mockNode{name: "node1"}
	node2 := &mockNode{name: "node2"}
	builder.AddNode(node1, inputNode)
	builder.AddNode(node2, node1)

	graph, err := builder.Build(node2)
	if err != nil {
		t.Fatalf("Build() failed: %v", err)
	}

	deps := graph.GetDependencies()
	if len(deps) == 0 {
		t.Error("expected dependencies, got empty map")
	}

	// Verify node1 depends on inputNode
	node1Deps, exists := deps[node1]
	if !exists {
		t.Error("expected node1 to have dependencies")
	}
	if len(node1Deps) != 1 || node1Deps[0] != inputNode {
		t.Errorf("expected node1 to depend on inputNode, got %v", node1Deps)
	}

	// Verify node2 depends on node1
	node2Deps, exists := deps[node2]
	if !exists {
		t.Error("expected node2 to have dependencies")
	}
	if len(node2Deps) != 1 || node2Deps[0] != node1 {
		t.Errorf("expected node2 to depend on node1, got %v", node2Deps)
	}

	// Test that returned map is a copy (modifying it shouldn't affect original)
	deps[node1] = nil
	originalDeps := graph.GetDependencies()
	if len(originalDeps[node1]) != 1 {
		t.Error("dependencies map should be a copy, original was modified")
	}
}

func TestGraph_GetAllNodes(t *testing.T) {
	var engine compute.Engine[int] = compute.NewCPUEngine[int](numeric.IntOps{})
	builder := NewBuilder[int](engine)

	inputNode := builder.Input([]int{2, 2})
	node1 := &mockNode{name: "node1"}
	node2 := &mockNode{name: "node2"}
	builder.AddNode(node1, inputNode)
	builder.AddNode(node2, node1)

	graph, err := builder.Build(node2)
	if err != nil {
		t.Fatalf("Build() failed: %v", err)
	}

	allNodes := graph.GetAllNodes()
	if len(allNodes) != 3 { // inputNode, node1, node2
		t.Errorf("expected 3 nodes, got %d", len(allNodes))
	}

	// Test that returned slice is a copy
	allNodes[0] = nil
	originalNodes := graph.GetAllNodes()
	if originalNodes[0] == nil {
		t.Error("nodes slice should be a copy, original was modified")
	}
}

func TestGraph_GetTopologicalOrder(t *testing.T) {
	var engine compute.Engine[int] = compute.NewCPUEngine[int](numeric.IntOps{})
	builder := NewBuilder[int](engine)

	inputNode := builder.Input([]int{2, 2})
	node1 := &mockNode{name: "node1"}
	node2 := &mockNode{name: "node2"}
	builder.AddNode(node1, inputNode)
	builder.AddNode(node2, node1)

	graph, err := builder.Build(node2)
	if err != nil {
		t.Fatalf("Build() failed: %v", err)
	}

	topOrder, err := graph.GetTopologicalOrder()
	if err != nil {
		t.Fatalf("GetTopologicalOrder() failed: %v", err)
	}

	if len(topOrder) != 3 {
		t.Errorf("expected 3 nodes in topological order, got %d", len(topOrder))
	}

	// Verify order: inputNode should come before node1, node1 before node2
	inputIndex, node1Index, node2Index := -1, -1, -1
	for i, node := range topOrder {
		switch node {
		case inputNode:
			inputIndex = i
		case node1:
			node1Index = i
		case node2:
			node2Index = i
		}
	}

	if inputIndex == -1 || node1Index == -1 || node2Index == -1 {
		t.Error("all nodes should be present in topological order")
	}
	if inputIndex >= node1Index || node1Index >= node2Index {
		t.Error("nodes should be in correct topological order")
	}
}
