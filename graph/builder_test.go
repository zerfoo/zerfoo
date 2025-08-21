package graph

import (
	"context"
	"testing"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
	_ "github.com/zerfoo/zerfoo/layers/core"
	_ "github.com/zerfoo/zerfoo/layers/gather"
	_ "github.com/zerfoo/zerfoo/layers/transpose"
)

type mockNode struct {
	name         string
	outputShape  []int
	forwardFunc  func(ctx context.Context, inputs ...*tensor.TensorNumeric[int]) (*tensor.TensorNumeric[int], error)
	backwardFunc func(ctx context.Context, outputGradient *tensor.TensorNumeric[int]) ([]*tensor.TensorNumeric[int], error)
	params       []*Parameter[int]
}

func (m *mockNode) OutputShape() []int { return m.outputShape }
func (m *mockNode) Forward(ctx context.Context, inputs ...*tensor.TensorNumeric[int]) (*tensor.TensorNumeric[int], error) {
	if m.forwardFunc != nil {
		return m.forwardFunc(ctx, inputs...)
	}

	return inputs[0], nil
}

func (m *mockNode) Backward(ctx context.Context, outputGradient *tensor.TensorNumeric[int], inputs ...*tensor.TensorNumeric[int]) ([]*tensor.TensorNumeric[int], error) {
	if m.backwardFunc != nil {
		return m.backwardFunc(ctx, outputGradient)
	}

	return []*tensor.TensorNumeric[int]{outputGradient}, nil
}

func (m *mockNode) Parameters() []*Parameter[int] {
	return m.params
}

func TestBuilder_Build(t *testing.T) {
	var engine compute.Engine[int] = compute.NewCPUEngine[int](numeric.IntOps{})
	builder := NewBuilder[int](engine)

	inputNode := builder.Input([]int{2, 2})

	node1 := &mockNode{
		name: "node1",
		forwardFunc: func(ctx context.Context, inputs ...*tensor.TensorNumeric[int]) (*tensor.TensorNumeric[int], error) {
			return inputs[0], nil
		},
	}
	node1.backwardFunc = func(ctx context.Context, outputGradient *tensor.TensorNumeric[int]) ([]*tensor.TensorNumeric[int], error) {
		return []*tensor.TensorNumeric[int]{outputGradient}, nil
	}
	node2 := &mockNode{
		name: "node2",
		forwardFunc: func(ctx context.Context, inputs ...*tensor.TensorNumeric[int]) (*tensor.TensorNumeric[int], error) {
			return inputs[0], nil
		},
	}
	node2.backwardFunc = func(ctx context.Context, outputGradient *tensor.TensorNumeric[int]) ([]*tensor.TensorNumeric[int], error) {
		return []*tensor.TensorNumeric[int]{outputGradient}, nil
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

	_ = graph.Backward(context.Background(), input)
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
	_, _ = inputNode.Backward(context.Background(), nil)
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