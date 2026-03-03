// Package model provides the core structures and loading mechanisms for Zerfoo models.
package model

import (
	"fmt"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/graph"
	"github.com/zerfoo/zerfoo/log"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
)

// LayerBuilder is a function that constructs a graph.Node (a layer) from ZMF parameters.
type LayerBuilder[T tensor.Numeric] func(
	engine compute.Engine[T],
	ops numeric.Arithmetic[T],
	name string,
	params map[string]*graph.Parameter[T],
	attributes map[string]interface{},
) (graph.Node[T], error)

// registry maps ZMF op_type strings to their corresponding LayerBuilder functions.
var registry = make(map[string]interface{})

// pkgLogger is the package-level logger for model operations.
// Override via SetLogger before calling RegisterLayer if you need output.
var pkgLogger log.Logger = log.Nop()

// SetLogger sets the package-level logger for model operations.
func SetLogger(l log.Logger) {
	if l == nil {
		l = log.Nop()
	}
	pkgLogger = l
}

// RegisterLayer adds a new layer builder to the registry.
// It is intended to be called at initialization time (e.g., in an init() function).
func RegisterLayer[T tensor.Numeric](opType string, builder LayerBuilder[T]) {
	if _, exists := registry[opType]; exists {
		pkgLogger.Warn("overwriting existing layer builder", "op_type", opType)
	}

	registry[opType] = builder
}

// UnregisterLayer removes a layer builder from the registry.
func UnregisterLayer(opType string) {
	delete(registry, opType)
}

// GetLayerBuilder retrieves a layer builder from the registry for a given op_type.
func GetLayerBuilder[T tensor.Numeric](opType string) (LayerBuilder[T], error) {
	builder, exists := registry[opType]
	if !exists {
		return nil, fmt.Errorf("unrecognized op_type: '%s'", opType)
	}

	typedBuilder, ok := builder.(LayerBuilder[T])
	if !ok {
		return nil, fmt.Errorf("layer builder for op_type '%s' has an incorrect type", opType)
	}

	return typedBuilder, nil
}
