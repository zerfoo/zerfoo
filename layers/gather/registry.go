package gather

import (
	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/graph"
	"github.com/zerfoo/zerfoo/model"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
)

func init() {
	model.RegisterLayer("Gather", BuildGather[float32])
}

func BuildGather[T tensor.Numeric](
	engine compute.Engine[T],
	_ numeric.Arithmetic[T],
	name string,
	params map[string]*graph.Parameter[T],
	attributes map[string]interface{},
) (graph.Node[T], error) {
	// Try to find the embedding weights parameter
	// Look for common patterns in parameter naming
	var weights *graph.Parameter[T]
	var ok bool
	
	// Pattern 1: Try exact match with expected parameter name
	if weightsName, exists := attributes["weights_param"]; exists {
		if paramName, isString := weightsName.(string); isString {
			weights, ok = params[paramName]
		}
	}
	
	// Pattern 2: Try common embedding weight patterns
	if !ok {
		candidates := []string{
			"model.embed_tokens.weight",
			"embed_tokens.weight", 
			name + ".weight",
			name + "_weight",
		}
		
		for _, candidate := range candidates {
			if weights, ok = params[candidate]; ok {
				break
			}
		}
	}
	
	// If we found weights, create a Gather layer with embedded weights
	if ok {
		return NewWithWeights(engine, weights.Value), nil
	}
	
	// Fallback: create basic Gather layer that expects weights as input
	return New(engine), nil
}
