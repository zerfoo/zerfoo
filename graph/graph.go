package graph

import (
	"context"
	"errors"
	"fmt"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/tensor"
	"github.com/zerfoo/zerfoo/types"
)

// Graph represents a computation graph with a defined execution order.
type Graph[T tensor.Numeric] struct {
	engine       compute.Engine[T]
	nodes        []Node[T]
	dependencies map[Node[T]][]Node[T]
	inputs       []Node[T]
	output       Node[T]
	memo         map[Node[T]]*tensor.TensorNumeric[T]
}

// Forward executes the forward pass of the entire graph.
func (g *Graph[T]) Forward(ctx context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if len(inputs) != len(g.inputs) {
		return nil, fmt.Errorf("expected %d inputs, got %d", len(g.inputs), len(inputs))
	}

	g.memo = make(map[Node[T]]*tensor.TensorNumeric[T])
	for i, n := range g.inputs {
		g.memo[n] = inputs[i]
	}

	for _, n := range g.nodes {
		if _, ok := n.(*inputNode[T]); ok {
			continue
		}

		nodeInputs := make([]*tensor.TensorNumeric[T], len(g.dependencies[n]))
		for i, dep := range g.dependencies[n] {
			nodeInputs[i] = g.memo[dep]
		}

		output, err := n.Forward(ctx, nodeInputs...)
		if err != nil {
			return nil, err
		}

		g.memo[n] = output
	}

	return g.memo[g.output], nil
}

// Backward executes the backward pass of the entire graph.
func (g *Graph[T]) Backward(ctx context.Context, mode types.BackwardMode, initialGradient *tensor.TensorNumeric[T]) error {
	grads := make(map[Node[T]]*tensor.TensorNumeric[T])
	grads[g.output] = initialGradient

	for i := len(g.nodes) - 1; i >= 0; i-- {
		node := g.nodes[i]
		if grad, ok := grads[node]; ok {
			nodeInputs := make([]*tensor.TensorNumeric[T], len(g.dependencies[node]))
			for j, dep := range g.dependencies[node] {
				nodeInputs[j] = g.memo[dep]
			}

			inputGrads, err := node.Backward(ctx, mode, grad, nodeInputs...)
			if err != nil {
				return err
			}

			for j, dep := range g.dependencies[node] {
				if existingGrad, ok := grads[dep]; !ok {
					grads[dep] = inputGrads[j]
				} else {
					// Accumulate gradients if multiple paths converge to the same node
					addedGrad, err := g.engine.Add(ctx, existingGrad, inputGrads[j])
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

// Nodes returns all the nodes in the graph.
func (g *Graph[T]) Nodes() []Node[T] {
	return g.nodes
}

// Dependencies returns the dependencies of a given node.
func (g *Graph[T]) Dependencies(n Node[T]) []Node[T] {
	return g.dependencies[n]
}

// GetNodeMetadata returns metadata for a specific node including its type, attributes, and shape.
func (g *Graph[T]) GetNodeMetadata(n Node[T]) map[string]interface{} {
	metadata := make(map[string]interface{})
	metadata["op_type"] = n.OpType()
	metadata["output_shape"] = n.OutputShape()
	metadata["attributes"] = n.Attributes()
	metadata["parameter_count"] = len(n.Parameters())
	return metadata
}

// GetDependencies returns the dependency map for all nodes in the graph.
func (g *Graph[T]) GetDependencies() map[Node[T]][]Node[T] {
	// Return a copy to prevent external modification
	deps := make(map[Node[T]][]Node[T])
	for node, nodeDeps := range g.dependencies {
		depsCopy := make([]Node[T], len(nodeDeps))
		copy(depsCopy, nodeDeps)
		deps[node] = depsCopy
	}
	return deps
}

// GetAllNodes returns all nodes in the graph in their current order.
func (g *Graph[T]) GetAllNodes() []Node[T] {
	// Return a copy to prevent external modification
	nodes := make([]Node[T], len(g.nodes))
	copy(nodes, g.nodes)
	return nodes
}

// GetTopologicalOrder returns the nodes in topological order for execution.
func (g *Graph[T]) GetTopologicalOrder() ([]Node[T], error) {
	return topologicalSort(g.nodes, g.dependencies)
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

func (n *inputNode[T]) Forward(_ context.Context, _ ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	return nil, nil
}

func (n *inputNode[T]) Backward(_ context.Context, mode types.BackwardMode, _ *tensor.TensorNumeric[T], _ ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	return nil, nil
}

func (n *inputNode[T]) Parameters() []*Parameter[T] { return nil }

// Statically assert that the type implements the interface.
var _ Node[float32] = (*inputNode[float32])(nil)

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
