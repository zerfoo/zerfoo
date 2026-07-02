// Package model provides adapter implementations for bridging existing and new model interfaces.
package model

import (
	"context"
	"fmt"
	"time"

	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/ztensor/types"
)

// StandardModelInstance adapts the existing Model struct to implement ModelInstance interface.
type StandardModelInstance[T tensor.Numeric] struct {
	model    *Model[T]
	training bool
	metadata ModelMetadata
}

// NewStandardModelInstance creates a new StandardModelInstance adapter.
func NewStandardModelInstance[T tensor.Numeric](model *Model[T]) *StandardModelInstance[T] {
	metadata := ModelMetadata{
		Name:         "Standard Model",
		Version:      "1.0.0",
		Architecture: "standard",
		Framework:    "zerfoo",
		CreatedAt:    time.Now().Format(time.RFC3339),
		Parameters:   0, // Will be updated below if graph is not nil
		Tags:         []string{"zerfoo", "standard"},
		Extensions:   make(map[string]interface{}),
	}

	// Determine input and output shapes from graph
	if model.Graph != nil {
		// Get input shapes from graph inputs
		inputs := model.Graph.Inputs()
		metadata.InputShape = make([][]int, len(inputs))
		for i, input := range inputs {
			metadata.InputShape[i] = input.OutputShape()
		}

		// Get output shape from graph output
		if output := model.Graph.Output(); output != nil {
			metadata.OutputShape = output.OutputShape()
		}

		// Update parameter count
		metadata.Parameters = int64(len(model.Graph.Parameters()))
	}

	return &StandardModelInstance[T]{
		model:    model,
		training: false,
		metadata: metadata,
	}
}

// Forward implements ModelInstance.Forward
func (s *StandardModelInstance[T]) Forward(ctx context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	return s.model.Forward(ctx, inputs...)
}

// Backward implements ModelInstance.Backward.
// The first variadic input is the gradient of the loss with respect to the
// model output (initial gradient). Exactly one gradient tensor must be provided.
func (s *StandardModelInstance[T]) Backward(ctx context.Context, inputs ...*tensor.TensorNumeric[T]) error {
	if s.model.Graph == nil {
		return fmt.Errorf("model has no computation graph")
	}
	if len(inputs) != 1 {
		return fmt.Errorf("Backward requires exactly 1 gradient tensor, got %d", len(inputs))
	}
	return s.model.Graph.Backward(ctx, types.FullBackprop, inputs[0])
}

// GetGraph implements ModelInstance.GetGraph
func (s *StandardModelInstance[T]) GetGraph() *graph.Graph[T] {
	return s.model.Graph
}

// GetMetadata implements ModelInstance.GetMetadata
func (s *StandardModelInstance[T]) GetMetadata() ModelMetadata {
	// Update modified time
	s.metadata.ModifiedAt = time.Now().Format(time.RFC3339)
	return s.metadata
}

// Parameters implements ModelInstance.Parameters
func (s *StandardModelInstance[T]) Parameters() []*graph.Parameter[T] {
	if s.model.Graph == nil {
		return make([]*graph.Parameter[T], 0)
	}
	return s.model.Graph.Parameters()
}

// SetTrainingMode implements ModelInstance.SetTrainingMode
func (s *StandardModelInstance[T]) SetTrainingMode(training bool) {
	s.training = training
}

// IsTraining implements ModelInstance.IsTraining
func (s *StandardModelInstance[T]) IsTraining() bool {
	return s.training
}

// StandardModelProvider provides standard model creation capabilities.
type StandardModelProvider[T tensor.Numeric] struct {
	capabilities ModelCapabilities
	providerInfo ProviderInfo
}

// NewStandardModelProvider creates a new StandardModelProvider.
func NewStandardModelProvider[T tensor.Numeric]() *StandardModelProvider[T] {
	capabilities := ModelCapabilities{
		SupportedTypes:      []string{"standard"},
		SupportedPrecisions: []string{"float32", "float64"},
		SupportsTraining:    true,
		SupportsInference:   true,
		SupportsBatching:    true,
		SupportsStreaming:   false,
		MaxBatchSize:        1000,
		MaxSequenceLength:   8192,
	}

	providerInfo := ProviderInfo{
		Name:         "Standard Zerfoo Model Provider",
		Version:      "1.0.0",
		Description:  "Provides standard Zerfoo model instances with embedding and graph components",
		SupportedOps: []string{"Forward", "Backward", "Parameter Access"},
		Website:      "https://github.com/zerfoo/zerfoo",
		License:      "Apache-2.0",
	}

	return &StandardModelProvider[T]{
		capabilities: capabilities,
		providerInfo: providerInfo,
	}
}

// CreateModel implements ModelProvider.CreateModel
func (p *StandardModelProvider[T]) CreateModel(ctx context.Context, config ModelConfig) (ModelInstance[T], error) {
	// For now, return an error as we need specific implementation based on config
	return nil, fmt.Errorf("CreateModel not implemented for StandardModelProvider - use CreateFromGraph instead")
}

// CreateFromGraph implements ModelProvider.CreateFromGraph
func (p *StandardModelProvider[T]) CreateFromGraph(ctx context.Context, g *graph.Graph[T], config ModelConfig) (ModelInstance[T], error) {
	// Create a Model instance from the graph
	model := &Model[T]{
		Graph: g,
	}

	instance := NewStandardModelInstance(model)
	instance.SetTrainingMode(config.TrainingMode)

	// Update metadata from config
	if config.Extensions != nil {
		instance.metadata.Extensions = config.Extensions
	}

	return instance, nil
}

// GetCapabilities implements ModelProvider.GetCapabilities
func (p *StandardModelProvider[T]) GetCapabilities() ModelCapabilities {
	return p.capabilities
}

// GetProviderInfo implements ModelProvider.GetProviderInfo
func (p *StandardModelProvider[T]) GetProviderInfo() ProviderInfo {
	return p.providerInfo
}

// BasicModelValidator provides basic model validation functionality.
type BasicModelValidator[T tensor.Numeric] struct {
	validatorInfo ValidatorInfo
}

// NewBasicModelValidator creates a new BasicModelValidator.
func NewBasicModelValidator[T tensor.Numeric]() *BasicModelValidator[T] {
	validatorInfo := ValidatorInfo{
		Name:        "Basic Model Validator",
		Version:     "1.0.0",
		Description: "Provides basic model validation including graph consistency and parameter checks",
		CheckTypes:  []string{"graph_consistency", "parameter_validation", "shape_validation"},
		Strictness:  "medium",
	}

	return &BasicModelValidator[T]{
		validatorInfo: validatorInfo,
	}
}

// ValidateModel implements ModelValidator.ValidateModel
func (v *BasicModelValidator[T]) ValidateModel(ctx context.Context, model ModelInstance[T]) (*ValidationResult, error) {
	result := &ValidationResult{
		IsValid:    true,
		Errors:     make([]ValidationError, 0),
		Warnings:   make([]ValidationWarning, 0),
		Metrics:    make(map[string]float64),
		Extensions: make(map[string]interface{}),
	}

	// Validate graph consistency
	if err := v.ValidateArchitecture(ctx, model); err != nil {
		result.IsValid = false
		result.Errors = append(result.Errors, ValidationError{
			Type:      "architecture_error",
			Message:   err.Error(),
			Component: "graph",
			Severity:  "high",
		})
	}

	// Basic metrics
	result.Metrics["parameter_count"] = float64(len(model.Parameters()))
	result.Metrics["input_count"] = float64(len(model.GetMetadata().InputShape))

	if result.IsValid {
		result.Summary = "Model passed all validation checks"
	} else {
		result.Summary = fmt.Sprintf("Model failed validation with %d errors", len(result.Errors))
	}

	return result, nil
}

// ValidateInputs implements ModelValidator.ValidateInputs
func (v *BasicModelValidator[T]) ValidateInputs(ctx context.Context, model ModelInstance[T], inputs ...*tensor.TensorNumeric[T]) error {
	metadata := model.GetMetadata()

	if len(inputs) != len(metadata.InputShape) {
		return fmt.Errorf("expected %d inputs, got %d", len(metadata.InputShape), len(inputs))
	}

	for i, input := range inputs {
		if i >= len(metadata.InputShape) {
			break
		}

		expectedShape := metadata.InputShape[i]
		actualShape := input.Shape()

		// Check if shapes are compatible (allowing for dynamic batch dimension)
		if len(expectedShape) != len(actualShape) {
			return fmt.Errorf("input %d: expected %d dimensions, got %d", i, len(expectedShape), len(actualShape))
		}

		// Check non-batch dimensions (skip first dimension which is typically batch size)
		for j := 1; j < len(expectedShape); j++ {
			if expectedShape[j] != actualShape[j] && expectedShape[j] > 0 {
				return fmt.Errorf("input %d: expected shape %v, got %v", i, expectedShape, actualShape)
			}
		}
	}

	return nil
}

// ValidateArchitecture implements ModelValidator.ValidateArchitecture
func (v *BasicModelValidator[T]) ValidateArchitecture(ctx context.Context, model ModelInstance[T]) error {
	g := model.GetGraph()
	if g == nil {
		return fmt.Errorf("model has no computation graph")
	}

	// Check that graph has inputs and output
	inputs := g.Inputs()
	if len(inputs) == 0 {
		return fmt.Errorf("model graph has no inputs")
	}

	output := g.Output()
	if output == nil {
		return fmt.Errorf("model graph has no output")
	}

	// Basic parameter validation
	params := g.Parameters()
	if len(params) == 0 {
		// This is a warning, not an error - some models might not have parameters
		return nil
	}

	// Check for nil parameters
	for i, param := range params {
		if param == nil {
			return fmt.Errorf("parameter %d is nil", i)
		}
		if param.Value == nil {
			return fmt.Errorf("parameter %d has nil value", i)
		}
	}

	return nil
}

// GetValidatorInfo implements ModelValidator.GetValidatorInfo
func (v *BasicModelValidator[T]) GetValidatorInfo() ValidatorInfo {
	return v.validatorInfo
}

// Ensure adapters implement their respective interfaces
var _ ModelInstance[float32] = (*StandardModelInstance[float32])(nil)
var _ ModelProvider[float32] = (*StandardModelProvider[float32])(nil)
var _ ModelValidator[float32] = (*BasicModelValidator[float32])(nil)
