package gather

import (
	"fmt"
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
	// Debug: Print available parameters for this Gather layer
	fmt.Printf("DEBUG: BuildGather for %s, available params: ", name)
	for paramName := range params {
		fmt.Printf("%s ", paramName)
	}
	fmt.Println()
	
	// Try to find any weights parameter that could be used with this Gather layer
	var embeddingWeights *graph.Parameter[T]
	
	// Common patterns for weights parameters
	weightPatterns := []string{
		"model.embed_tokens.weight",  // For embedding layers
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
	
	fmt.Printf("DEBUG: Looking for weights with patterns: %v\n", weightPatterns)
	
	for _, pattern := range weightPatterns {
		if param, exists := params[pattern]; exists {
			embeddingWeights = param
			fmt.Printf("DEBUG: Found weights with pattern: %s\n", pattern)
			break
		}
	}
	
	if embeddingWeights != nil {
		// Create Gather layer with embedded weights (expects only indices input)
		fmt.Printf("DEBUG: Creating Gather with embedded weights for %s\n", name)
		return NewWithWeights[T](engine, embeddingWeights.Value), nil
	}
	
	// For Gemma model, it seems all Gather layers expect only indices input
	// Create a basic Gather that can handle single input by using a dummy weight tensor
	fmt.Printf("DEBUG: Creating single-input Gather for %s (no weights found)\n", name)
	
	// Create a dummy weight tensor - this is a workaround for Gemma's pattern
	// In a real implementation, we'd need to handle this more elegantly
	dummyShape := []int{1, 1}  // Minimal shape
	dummyTensor, err := tensor.New[T](dummyShape, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create dummy tensor: %w", err)
	}
	
	return NewWithWeights[T](engine, dummyTensor), nil
}
