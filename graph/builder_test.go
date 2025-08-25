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
	return nil
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
