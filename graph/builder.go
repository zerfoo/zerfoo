package graph

import (
	"context"
	"fmt"
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

// Build constructs the final graph and returns forward and backward functions.
func (b *Builder[T]) Build(outputNode Node[T]) (func(inputs ...*tensor.Tensor[T]) (*tensor.Tensor[T], error), func(initialGradient *tensor.Tensor[T]) error, error) {
	sortedNodes, err := topologicalSort[T](b.nodes, b.dependencies)
	if err != nil {
		return nil, nil, err
	}

	g := &Graph[T]{
		engine:       b.engine,
		nodes:        sortedNodes,
		dependencies: b.dependencies,
		inputs:       b.inputs,
		output:       outputNode,
	}

	return g.Forward, g.Backward, nil
}

// Parameters returns all the trainable parameters in the graph.
func (b *Builder[T]) Parameters() []*Parameter[T] {
	var params []*Parameter[T]
	for _, node := range b.nodes {
		params = append(params, node.Parameters()...)
	}
	return params
}

// Graph represents a computation graph with a defined execution order.
type Graph[T tensor.Numeric] struct {
	engine       compute.Engine[T]
	nodes        []Node[T]
	dependencies map[Node[T]][]Node[T]
	inputs       []Node[T]
	output       Node[T]
}

// Forward executes the forward pass of the entire graph.
func (g *Graph[T]) Forward(inputs ...*tensor.Tensor[T]) (*tensor.Tensor[T], error) {
	memo := make(map[Node[T]]*tensor.Tensor[T])
	for i, n := range g.inputs {
		memo[n] = inputs[i]
	}

	for _, n := range g.nodes {
		if _, ok := n.(*inputNode[T]); ok {
			continue
		}
		nodeInputs := make([]*tensor.Tensor[T], len(g.dependencies[n]))
		for i, dep := range g.dependencies[n] {
			nodeInputs[i] = memo[dep]
		}
		output, err := n.Forward(nodeInputs...)
		if err != nil {
			return nil, err
		}
		memo[n] = output
	}
	return memo[g.output], nil
}

// Backward executes the backward pass of the entire graph.
func (g *Graph[T]) Backward(initialGradient *tensor.Tensor[T]) error {
	grads := make(map[Node[T]]*tensor.Tensor[T])
	grads[g.output] = initialGradient

	for i := len(g.nodes) - 1; i >= 0; i-- {
		node := g.nodes[i]
		if grad, ok := grads[node]; ok {
			inputGrads, err := node.Backward(grad)
			if err != nil {
				return err
			}
			for j, dep := range g.dependencies[node] {
				if existingGrad, ok := grads[dep]; !ok {
					grads[dep] = inputGrads[j]
				} else {
					// Accumulate gradients if multiple paths converge to the same node
					addedGrad, err := g.engine.Add(context.Background(), existingGrad, inputGrads[j])
					if err != nil {
						return fmt.Errorf("error accumulating gradients: %v", err)
					}
					grads[dep] = addedGrad
				}
			}
		}
	}
	return nil
}

// Parameters returns all the trainable parameters in the graph.
func (g *Graph[T]) Parameters() []*Parameter[T] {
	var params []*Parameter[T]
	for _, node := range g.nodes {
		params = append(params, node.Parameters()...)
	}
	return params
}

// inputNode is a special node type for graph inputs.
type inputNode[T tensor.Numeric] struct {
	shape []int
}

func (n *inputNode[T]) OutputShape() []int {
	return n.shape
}

func (n *inputNode[T]) Forward(inputs ...*tensor.Tensor[T]) (*tensor.Tensor[T], error) {
	return nil, nil
}
func (n *inputNode[T]) Backward(outputGradient *tensor.Tensor[T]) ([]*tensor.Tensor[T], error) {
	return nil, nil
}
func (n *inputNode[T]) Parameters() []*Parameter[T] { return nil }

func topologicalSort[T tensor.Numeric](nodes []Node[T], deps map[Node[T]][]Node[T]) ([]Node[T], error) {
	var sorted []Node[T]
	visited := make(map[Node[T]]bool)
	recursionStack := make(map[Node[T]]bool)

	var visit func(node Node[T]) error
	visit = func(node Node[T]) error {
		if recursionStack[node] {
			return fmt.Errorf("cycle detected in graph")
		}
		if visited[node] {
			return nil
		}

		recursionStack[node] = true
		visited[node] = true

		for _, dep := range deps[node] {
			if err := visit(dep); err != nil {
				return err
			}
		}

		sorted = append(sorted, node)
		delete(recursionStack, node)
		return nil
	}

	for _, node := range nodes {
		if err := visit(node); err != nil {
			return nil, err
		}
	}
	return sorted, nil
}
