// Package model_test provides tests for the model package.
package model_test

import (
	"context"
	"testing"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/graph"
	"github.com/zerfoo/zerfoo/model"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
	"github.com/zerfoo/zmf"
)

// mockNode is a simple implementation of graph.Node for testing.
type mockNode[T tensor.Numeric] struct {
	name       string
	forwardFn  func(inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error)
	parameters []*graph.Parameter[T]
}

func (m *mockNode[T]) OutputShape() []int { return []int{1} }
func (m *mockNode[T]) Forward(_ context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if m.forwardFn != nil {
		return m.forwardFn(inputs...)
	}

	return tensor.New[T]([]int{1}, []T{1})
}

func (m *mockNode[T]) Backward(_ context.Context, _ *tensor.TensorNumeric[T], _ ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	return nil, nil
}
func (m *mockNode[T]) Parameters() []*graph.Parameter[T]  { return m.parameters }
func (m *mockNode[T]) OpType() string                     { return m.name }
func (m *mockNode[T]) Attributes() map[string]interface{} { return nil }

// TestBuildFromZMF_ConnectedGraph tests building a graph with multiple connected nodes.
func TestBuildFromZMF_ConnectedGraph(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	// 1. Register mock layer builders for our test op types.
	model.RegisterLayer("OpA", func(e compute.Engine[float32], o numeric.Arithmetic[float32], n string, p map[string]*graph.Parameter[float32], a map[string]interface{}) (graph.Node[float32], error) {
		// This node adds 1 to its input
		return &mockNode[float32]{
			name: n,
			forwardFn: func(inputs ...*tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
				outData := inputs[0].Data()[0] + 1

				return tensor.New[float32]([]int{1}, []float32{outData})
			},
		}, nil
	})
	model.RegisterLayer("OpB", func(e compute.Engine[float32], o numeric.Arithmetic[float32], n string, p map[string]*graph.Parameter[float32], a map[string]interface{}) (graph.Node[float32], error) {
		// This node multiplies its input by 2
		return &mockNode[float32]{
			name: n,
			forwardFn: func(inputs ...*tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
				outData := inputs[0].Data()[0] * 2

				return tensor.New[float32]([]int{1}, []float32{outData})
			},
		}, nil
	})

	// 2. Define a ZMF model with a simple graph: input -> nodeA -> nodeB
	zmfModel := &zmf.Model{
		Graph: &zmf.Graph{
			Inputs: []*zmf.ValueInfo{
				{Name: "graph_input", Shape: []int64{1}},
			},
			Nodes: []*zmf.Node{
				{Name: "nodeA", OpType: "OpA", Inputs: []string{"graph_input"}},
				{Name: "nodeB", OpType: "OpB", Inputs: []string{"nodeA"}},
			},
			Outputs: []*zmf.ValueInfo{
				{Name: "nodeB"},
			},
		},
	}

	// 3. Call the function to be tested.
	builtGraph, err := model.BuildFromZMF[float32](engine, ops, zmfModel)
	if err != nil {
		t.Fatalf("BuildFromZMF failed: %v", err)
	}

	// 4. Verify the result by running the forward pass.
	if builtGraph == nil {
		t.Fatal("BuildFromZMF returned a nil graph")
	}

	inputTensor, _ := tensor.New[float32]([]int{1}, []float32{10})
	expectedOutput := float32((10 + 1) * 2) // 22

	output, err := builtGraph.Forward(context.Background(), inputTensor)
	if err != nil {
		t.Fatalf("Graph forward pass failed: %v", err)
	}

	if output.Data()[0] != expectedOutput {
		t.Errorf("Expected output %f, got %f", expectedOutput, output.Data()[0])
	}
}
