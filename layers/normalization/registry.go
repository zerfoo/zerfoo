// Package normalization provides various normalization layers for neural networks.
package normalization

import (
	"errors"
	"fmt"
	"strings"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/graph"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
)

// BuildRMSNorm constructs an RMSNorm node from the provided parameter and epsilon attribute.
func BuildRMSNorm[T tensor.Numeric](
	engine compute.Engine[T],
	ops numeric.Arithmetic[T],
	name string,
	params map[string]*graph.Parameter[T],
	attributes map[string]interface{},
) (graph.Node[T], error) {
	gain, ok := params[name+"_gain"]
	if !ok {
		return nil, fmt.Errorf("missing required parameter: %s_gain", name)
	}

	epsilonAttr, ok := attributes["epsilon"]
	if !ok {
		return nil, errors.New("missing required attribute: epsilon")
	}

	var epsilon float64
	switch v := epsilonAttr.(type) {
	case float64:
		epsilon = v
	case float32:
		epsilon = float64(v)
	default:
		return nil, fmt.Errorf("attribute 'epsilon' has incorrect type: expected float64 or float32, got %T", epsilonAttr)
	}

	return NewRMSNormFromParam(engine, ops, ops.FromFloat64(epsilon), gain)
}

// BuildSimplifiedLayerNormalization constructs a SimplifiedLayerNormalization node,
// attempting multiple common naming patterns to resolve the gain/weight parameter.
func BuildSimplifiedLayerNormalization[T tensor.Numeric](
	engine compute.Engine[T],
	ops numeric.Arithmetic[T],
	name string,
	params map[string]*graph.Parameter[T],
	attributes map[string]interface{},
) (graph.Node[T], error) {
	// Try multiple parameter naming patterns for LayerNorm gain/weight
	var gain *graph.Parameter[T]
	var ok bool

	// Pattern 1: {name}_gain (original expected format)
	gain, ok = params[name+"_gain"]
	if !ok {
		// Pattern 2: Convert path-like name to dot notation + .weight
		// e.g., "/model/layers.0/input_layernorm/LayerNorm" -> "model.layers.0.input_layernorm.weight"
		dotName := strings.ReplaceAll(strings.TrimPrefix(name, "/"), "/", ".")
		if strings.HasSuffix(dotName, ".LayerNorm") {
			dotName = strings.TrimSuffix(dotName, ".LayerNorm") + ".weight"
		}
		if strings.HasSuffix(dotName, ".SimplifiedLayerNormalization") {
			dotName = strings.TrimSuffix(dotName, ".SimplifiedLayerNormalization") + ".weight"
		}
		gain, ok = params[dotName]
	}
	if !ok {
		// Pattern 3: Try just the weight suffix
		weightName := strings.ReplaceAll(strings.TrimPrefix(name, "/"), "/", ".") + ".weight"
		gain, ok = params[weightName]
	}
	if !ok {
		// Pattern 4: Try layernorm.weight pattern
		// e.g., "/model/layers.0/attn/q_norm/SimplifiedLayerNormalization" -> "model.layers.0.attn.q_norm.layernorm.weight"
		layernormName := strings.ReplaceAll(strings.TrimPrefix(name, "/"), "/", ".")
		if strings.HasSuffix(layernormName, ".SimplifiedLayerNormalization") {
			layernormName = strings.TrimSuffix(layernormName, ".SimplifiedLayerNormalization") + ".layernorm.weight"
		}
		gain, ok = params[layernormName]
	}

	if !ok {
		return nil, fmt.Errorf("missing required parameter for LayerNorm. Tried: %s_gain, and weight patterns", name)
	}

	epsilonAttr, ok := attributes["epsilon"]
	if !ok {
		return nil, errors.New("missing required attribute: epsilon")
	}

	var epsilon float64
	switch v := epsilonAttr.(type) {
	case float64:
		epsilon = v
	case float32:
		epsilon = float64(v)
	default:
		return nil, fmt.Errorf("attribute 'epsilon' has incorrect type: expected float64 or float32, got %T", epsilonAttr)
	}

	return NewSimplifiedLayerNormalization(engine, ops, gain.Value, T(epsilon))
}

// BuildSkipSimplifiedLayerNormalization constructs a SkipSimplifiedLayerNormalization node,
// resolving the gain/weight parameter using several naming conventions.
func BuildSkipSimplifiedLayerNormalization[T tensor.Numeric](
	engine compute.Engine[T],
	ops numeric.Arithmetic[T],
	name string,
	params map[string]*graph.Parameter[T],
	attributes map[string]interface{},
) (graph.Node[T], error) {
	// Try multiple parameter naming patterns for LayerNorm gain/weight
	var gain *graph.Parameter[T]
	var ok bool

	// Pattern 1: {name}_gain (original expected format)
	gain, ok = params[name+"_gain"]
	if !ok {
		// Pattern 2: Convert path-like name to dot notation + .weight
		// e.g., "/model/layers.0/pre_feedforward_layernorm/SkipLayerNorm" -> "model.layers.0.pre_feedforward_layernorm.weight"
		dotName := strings.ReplaceAll(strings.TrimPrefix(name, "/"), "/", ".")
		if strings.HasSuffix(dotName, ".SkipLayerNorm") {
			dotName = strings.TrimSuffix(dotName, ".SkipLayerNorm") + ".weight"
		}
		if strings.HasSuffix(dotName, ".SkipSimplifiedLayerNormalization") {
			dotName = strings.TrimSuffix(dotName, ".SkipSimplifiedLayerNormalization") + ".weight"
		}
		gain, ok = params[dotName]
	}
	if !ok {
		// Pattern 3: Try just the weight suffix
		weightName := strings.ReplaceAll(strings.TrimPrefix(name, "/"), "/", ".") + ".weight"
		gain, ok = params[weightName]
	}
	if !ok {
		// Pattern 4: Try layernorm.weight pattern
		layernormName := strings.ReplaceAll(strings.TrimPrefix(name, "/"), "/", ".")
		if strings.HasSuffix(layernormName, ".SkipLayerNorm") {
			layernormName = strings.TrimSuffix(layernormName, ".SkipLayerNorm") + ".layernorm.weight"
		}
		if strings.HasSuffix(layernormName, ".SkipSimplifiedLayerNormalization") {
			layernormName = strings.TrimSuffix(layernormName, ".SkipSimplifiedLayerNormalization") + ".layernorm.weight"
		}
		gain, ok = params[layernormName]
	}

	if !ok {
		return nil, fmt.Errorf("missing required parameter for SkipLayerNorm. Tried: %s_gain, and weight patterns", name)
	}

	epsilonAttr, ok := attributes["epsilon"]
	if !ok {
		return nil, errors.New("missing required attribute: epsilon")
	}

	var epsilon float64
	switch v := epsilonAttr.(type) {
	case float64:
		epsilon = v
	case float32:
		epsilon = float64(v)
	default:
		return nil, fmt.Errorf("attribute 'epsilon' has incorrect type: expected float64 or float32, got %T", epsilonAttr)
	}

	return NewSkipSimplifiedLayerNormalization(engine, ops, gain.Value, T(epsilon))
}
