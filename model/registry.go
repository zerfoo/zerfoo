// Package model provides the core structures and loading mechanisms for Zerfoo models.
package model

import (
	"fmt"
	"sort"
	"sync"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/log"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
)

// LayerBuilder is a function that constructs a graph.Node (a layer) from serialized parameters.
type LayerBuilder[T tensor.Numeric] func(
	engine compute.Engine[T],
	ops numeric.Arithmetic[T],
	name string,
	params map[string]*graph.Parameter[T],
	attributes map[string]interface{},
) (graph.Node[T], error)

// registry maps op_type strings to their corresponding LayerBuilder functions.
var registryMu sync.RWMutex

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
	registryMu.Lock()
	defer registryMu.Unlock()
	if _, exists := registry[opType]; exists {
		pkgLogger.Warn("overwriting existing layer builder", "op_type", opType)
	}

	registry[opType] = builder
}

// UnregisterLayer removes a layer builder from the registry.
func UnregisterLayer(opType string) {
	registryMu.Lock()
	defer registryMu.Unlock()
	delete(registry, opType)
}

// GetLayerBuilder retrieves a layer builder from the registry for a given op_type.
func GetLayerBuilder[T tensor.Numeric](opType string) (LayerBuilder[T], error) {
	registryMu.RLock()
	defer registryMu.RUnlock()
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

// ListLayerBuilders returns registered operator names in stable order.
func ListLayerBuilders() []string {
	registryMu.RLock()
	defer registryMu.RUnlock()
	names := make([]string, 0, len(registry))
	for name := range registry {
		names = append(names, name)
	}
	sort.Strings(names)
	return names
}
