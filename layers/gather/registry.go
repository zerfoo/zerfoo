package gather

import (
	"fmt"
	"strings"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/graph"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
)

func BuildGather[T tensor.Numeric](
	engine compute.Engine[T],
	ops numeric.Arithmetic[T],
	name string,
	params map[string]*graph.Parameter[T],
	attributes map[string]interface{},
) (graph.Node[T], error) {
	// Try to find any weights parameter that could be used with this Gather layer
	var embeddingWeights *graph.Parameter[T]

	// Common patterns for weights parameters
	weightPatterns := []string{
		"model.embed_tokens.weight", // For embedding layers
		name + ".weight",
		strings.TrimSuffix(name, "/Gather") + ".weight",
		name + "_weight",
	}

	// Also try to find any parameter that contains "weight" in the name
	for paramName, param := range params {
		if strings.Contains(paramName, "weight") {
			weightPatterns = append(weightPatterns, paramName)
			embeddingWeights = param

			break
		}
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

	// Create a dummy weight tensor - this is a workaround for Gemma's pattern
	// In a real implementation, we'd need to handle this more elegantly
	dummyShape := []int{1, 1} // Minimal shape
	dummyTensor, err := tensor.New[T](dummyShape, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create dummy tensor: %w", err)
	}

	return NewWithWeights[T](engine, dummyTensor), nil
}
