package graph

import (
	"context"
	"errors"
	"fmt"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/tensor"
)

// Graph represents a computation graph with a defined execution order.
type Graph[T tensor.Numeric] struct {
	engine       compute.Engine[T]
	nodes        []Node[T]
	dependencies map[Node[T]][]Node[T]
	inputs       []Node[T]
	output       Node[T]
}

// Forward executes the forward pass of the entire graph.
func (g *Graph[T]) Forward(ctx context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	memo := make(map[Node[T]]*tensor.TensorNumeric[T])
	for i, n := range g.inputs {
		memo[n] = inputs[i]
	}

	for _, n := range g.nodes {
		if _, ok := n.(*inputNode[T]); ok {
			continue
		}
		nodeInputs := make([]*tensor.TensorNumeric[T], len(g.dependencies[n]))
		for i, dep := range g.dependencies[n] {
			nodeInputs[i] = memo[dep]
		}
		output, err := n.Forward(ctx, nodeInputs...)
		if err != nil {
			return nil, err
		}
		memo[n] = output
	}

	return memo[g.output], nil
}

// Backward executes the backward pass of the entire graph.
func (g *Graph[T]) Backward(ctx context.Context, initialGradient *tensor.TensorNumeric[T]) error {
	grads := make(map[Node[T]]*tensor.TensorNumeric[T])
	grads[g.output] = initialGradient

	for i := len(g.nodes) - 1; i >= 0; i-- {
		node := g.nodes[i]
		if grad, ok := grads[node]; ok {
			inputGrads, err := node.Backward(ctx, grad)
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
						return fmt.Errorf("error accumulating gradients: %w", err)
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

// Inputs returns the input nodes of the graph.
func (g *Graph[T]) Inputs() []Node[T] {
	return g.inputs
}

// Output returns the output node of the graph.
func (g *Graph[T]) Output() Node[T] {
	return g.output
}

// inputNode is a special node type for graph inputs.
type inputNode[T tensor.Numeric] struct {
	shape []int
}

func (n *inputNode[T]) OpType() string {
	return "Input"
}

func (n *inputNode[T]) Attributes() map[string]interface{} {
	return make(map[string]interface{})
}

func (n *inputNode[T]) OutputShape() []int {
	return n.shape
}

func (n *inputNode[T]) Forward(ctx context.Context, _ ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	return nil, nil
}

func (n *inputNode[T]) Backward(ctx context.Context, _ *tensor.TensorNumeric[T], _ ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
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
			return errors.New("cycle detected in graph")
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
