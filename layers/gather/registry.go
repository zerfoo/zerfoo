package gather

import (
	"strings"

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
	ops numeric.Arithmetic[T],
	name string,
	params map[string]*graph.Parameter[T],
	attributes map[string]interface{},
) (graph.Node[T], error) {
	// For embedding layers like /model/embed_tokens/Gather, use embedded weights
	// For other Gather operations, use the standard two-input approach
	if strings.Contains(name, "embed_tokens") {
		// Try to find embedding weights parameter from multiple naming patterns
		var embeddingWeights *graph.Parameter[T]
		
		// Common patterns for embedding weights
		weightPatterns := []string{
			"model.embed_tokens.weight",
			name + ".weight",
			strings.TrimSuffix(name, "/Gather") + ".weight", 
			name + "_weight",
		}
		
		for _, pattern := range weightPatterns {
			if param, exists := params[pattern]; exists {
				embeddingWeights = param
				break
			}
		}
		
		if embeddingWeights != nil {
			// Create Gather layer with embedded weights (expects only indices input)
			return NewWithWeights[T](engine, embeddingWeights.Value), nil
		}
	}
	
	// Create basic Gather layer (expects weights and indices inputs)
	return New[T](engine), nil
}
