// Package graph provides a computational graph abstraction.
package graph

import (
	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/tensor"
)

// Builder provides a fluent API for constructing a computation graph.
type Builder[T tensor.Numeric] struct {
	engine       compute.Engine[T]
	nodes        []Node[T]
	dependencies map[Node[T]][]Node[T]
	inputs       []Node[T]
}

// NewBuilder creates a new graph builder.
func NewBuilder[T tensor.Numeric](engine compute.Engine[T]) *Builder[T] {
	return &Builder[T]{
		engine:       engine,
		dependencies: make(map[Node[T]][]Node[T]),
	}
}

// AddNode adds a new node to the graph with the given inputs.
func (b *Builder[T]) AddNode(node Node[T], inputs ...Node[T]) Node[T] {
	b.nodes = append(b.nodes, node)
	b.dependencies[node] = inputs

	return node
}

// Input creates a new input node.
func (b *Builder[T]) Input(shape []int) Node[T] {
	inputNode := &inputNode[T]{shape: shape}
	b.nodes = append(b.nodes, inputNode)
	b.inputs = append(b.inputs, inputNode)

	return inputNode
}

// Build constructs the final graph.
func (b *Builder[T]) Build(outputNode Node[T]) (*Graph[T], error) {
	sortedNodes, err := topologicalSort[T](b.nodes, b.dependencies)
	if err != nil {
		return nil, err
	}

	g := &Graph[T]{
		engine:       b.engine,
		nodes:        sortedNodes,
		dependencies: b.dependencies,
		inputs:       b.inputs,
		output:       outputNode,
	}

	return g, nil
}

// Parameters returns all the trainable parameters in the graph.
func (b *Builder[T]) Parameters() []*Parameter[T] {
	var params []*Parameter[T]
	for _, node := range b.nodes {
		params = append(params, node.Parameters()...)
	}

	return params
}
