// Package modeldsl provides a declarative DSL for defining custom model
// architectures using Go structs. Model definitions are validated and compiled
// into runnable graphs that support forward inference.
package dsl

import (
	"errors"
	"fmt"
)

// LayerType identifies the kind of neural network layer.
type LayerType string

// LayerType constants use PascalCase to match the registration keys
// in layers/registry.RegisterAll. The registry is the canonical source
// of layer names; these constants mirror it for DSL use.
const (
	LayerLinear    LayerType = "Linear"
	LayerRMSNorm   LayerType = "RMSNorm"
	LayerSiLU      LayerType = "SiLU"
	LayerSoftmax   LayerType = "Softmax"
	LayerAttention LayerType = "Attention"
)

// LayerDef defines a single layer in the model.
type LayerDef struct {
	Name   string
	Type   LayerType
	Params map[string]any
}

// ConnectionDef specifies a directed edge from one layer to another.
type ConnectionDef struct {
	From string
	To   string
}

// ModelDef is the top-level model definition.
type ModelDef struct {
	Name        string
	Layers      []LayerDef
	Connections []ConnectionDef
}

// Parse validates a ModelDef and builds a ModelGraph.
func Parse(def *ModelDef) (*ModelGraph, error) {
	if def == nil {
		return nil, errors.New("modeldsl: model definition is nil")
	}
	if def.Name == "" {
		return nil, errors.New("modeldsl: model name is required")
	}
	if len(def.Layers) == 0 {
		return nil, errors.New("modeldsl: at least one layer is required")
	}

	layerIndex := make(map[string]int, len(def.Layers))
	for i, l := range def.Layers {
		if l.Name == "" {
			return nil, fmt.Errorf("modeldsl: layer %d has empty name", i)
		}
		if _, dup := layerIndex[l.Name]; dup {
			return nil, fmt.Errorf("modeldsl: duplicate layer name %q", l.Name)
		}
		if err := validateLayerType(l.Type); err != nil {
			return nil, fmt.Errorf("modeldsl: layer %q: %w", l.Name, err)
		}
		layerIndex[l.Name] = i
	}

	// Build adjacency lists.
	children := make(map[string][]string, len(def.Layers))
	parents := make(map[string][]string, len(def.Layers))
	for _, c := range def.Connections {
		if _, ok := layerIndex[c.From]; !ok {
			return nil, fmt.Errorf("modeldsl: connection references unknown layer %q", c.From)
		}
		if _, ok := layerIndex[c.To]; !ok {
			return nil, fmt.Errorf("modeldsl: connection references unknown layer %q", c.To)
		}
		children[c.From] = append(children[c.From], c.To)
		parents[c.To] = append(parents[c.To], c.From)
	}

	// Topological sort to detect cycles and determine execution order.
	order, err := topoSort(def.Layers, children, parents)
	if err != nil {
		return nil, err
	}

	// Identify input layers (no parents) and output layers (no children).
	var inputs, outputs []string
	for _, l := range def.Layers {
		if len(parents[l.Name]) == 0 {
			inputs = append(inputs, l.Name)
		}
		if len(children[l.Name]) == 0 {
			outputs = append(outputs, l.Name)
		}
	}

	return &ModelGraph{
		name:       def.Name,
		layers:     def.Layers,
		layerIndex: layerIndex,
		children:   children,
		parents:    parents,
		order:      order,
		inputs:     inputs,
		outputs:    outputs,
	}, nil
}

func validateLayerType(t LayerType) error {
	switch t {
	case LayerLinear, LayerRMSNorm, LayerSiLU, LayerSoftmax, LayerAttention:
		return nil
	default:
		return fmt.Errorf("unsupported layer type %q", t)
	}
}

func topoSort(layers []LayerDef, children, parents map[string][]string) ([]string, error) {
	inDegree := make(map[string]int, len(layers))
	for _, l := range layers {
		inDegree[l.Name] = len(parents[l.Name])
	}

	var queue []string
	for _, l := range layers {
		if inDegree[l.Name] == 0 {
			queue = append(queue, l.Name)
		}
	}

	var order []string
	for len(queue) > 0 {
		name := queue[0]
		queue = queue[1:]
		order = append(order, name)
		for _, child := range children[name] {
			inDegree[child]--
			if inDegree[child] == 0 {
				queue = append(queue, child)
			}
		}
	}

	if len(order) != len(layers) {
		return nil, errors.New("modeldsl: cycle detected in layer connections")
	}
	return order, nil
}
