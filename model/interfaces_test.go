package model

import (
	"context"
	"testing"
	"time"

	"github.com/zerfoo/zerfoo/graph"
	"github.com/zerfoo/zerfoo/tensor"
)

// Mock implementations for testing

// MockModelInstance implements ModelInstance for testing
type MockModelInstance[T tensor.Numeric] struct {
	graph     *graph.Graph[T]
	training  bool
	metadata  ModelMetadata
	params    []*graph.Parameter[T]
}

func NewMockModelInstance[T tensor.Numeric]() *MockModelInstance[T] {
	metadata := ModelMetadata{
		Name:         "MockModel",
		Version:      "1.0.0",
		Architecture: "mock",
		Framework:    "test",
		CreatedAt:    time.Now().Format(time.RFC3339),
		Parameters:   10,
		InputShape:   [][]int{{1, 32}, {1, 16}},
		OutputShape:  []int{1, 10},
		Tags:         []string{"test", "mock"},
		Extensions:   make(map[string]interface{}),
	}
	
	return &MockModelInstance[T]{
		training: false,
		metadata: metadata,
		params:   make([]*graph.Parameter[T], 0),
	}
}

func (m *MockModelInstance[T]) Forward(ctx context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	// Create a mock output tensor
	output, err := tensor.New[T]([]int{1, 10}, nil)
	return output, err
}

func (m *MockModelInstance[T]) Backward(ctx context.Context, inputs ...*tensor.TensorNumeric[T]) error {
	return nil
}

func (m *MockModelInstance[T]) GetGraph() *graph.Graph[T] {
	return m.graph
}

func (m *MockModelInstance[T]) GetMetadata() ModelMetadata {
	m.metadata.ModifiedAt = time.Now().Format(time.RFC3339)
	return m.metadata
}

func (m *MockModelInstance[T]) Parameters() []*graph.Parameter[T] {
	return m.params
}

func (m *MockModelInstance[T]) SetTrainingMode(training bool) {
	m.training = training
}

func (m *MockModelInstance[T]) IsTraining() bool {
	return m.training
}

// MockModelProvider implements ModelProvider for testing
type MockModelProvider[T tensor.Numeric] struct {
	capabilities ModelCapabilities
	providerInfo ProviderInfo
}

func NewMockModelProvider[T tensor.Numeric]() *MockModelProvider[T] {
	capabilities := ModelCapabilities{
		SupportedTypes:      []string{"mock", "test"},
		SupportedPrecisions: []string{"float32", "float64"},
		SupportsTraining:    true,
		SupportsInference:   true,
		SupportsBatching:    true,
		SupportsStreaming:   false,
		MaxBatchSize:        100,
		MaxSequenceLength:   512,
	}
	
	providerInfo := ProviderInfo{
		Name:         "Mock Model Provider",
		Version:      "1.0.0",
		Description:  "Mock provider for testing",
		SupportedOps: []string{"Forward", "Backward"},
		License:      "MIT",
	}
	
	return &MockModelProvider[T]{
		capabilities: capabilities,
		providerInfo: providerInfo,
	}
}

func (m *MockModelProvider[T]) CreateModel(ctx context.Context, config ModelConfig) (ModelInstance[T], error) {
	instance := NewMockModelInstance[T]()
	instance.SetTrainingMode(config.TrainingMode)
	if config.Extensions != nil {
		instance.metadata.Extensions = config.Extensions
	}
	return instance, nil
}

func (m *MockModelProvider[T]) CreateFromGraph(ctx context.Context, g *graph.Graph[T], config ModelConfig) (ModelInstance[T], error) {
	instance := NewMockModelInstance[T]()
	instance.graph = g
	instance.SetTrainingMode(config.TrainingMode)
	return instance, nil
}

func (m *MockModelProvider[T]) GetCapabilities() ModelCapabilities {
	return m.capabilities
}

func (m *MockModelProvider[T]) GetProviderInfo() ProviderInfo {
	return m.providerInfo
}

// Test functions

func TestModelRegistry_ModelProviderRegistration(t *testing.T) {
	registry := NewModelRegistry[float32]()
	
	// Test registration
	factory := func(ctx context.Context, config map[string]interface{}) (ModelProvider[float32], error) {
		return NewMockModelProvider[float32](), nil
	}
	
	err := registry.RegisterModelProvider("mock", factory)
	if err != nil {
		t.Fatalf("Failed to register model provider: %v", err)
	}
	
	// Test duplicate registration
	err = registry.RegisterModelProvider("mock", factory)
	if err == nil {
		t.Error("Expected error for duplicate registration, got nil")
	}
	
	// Test retrieval
	ctx := context.Background()
	provider, err := registry.GetModelProvider(ctx, "mock", nil)
	if err != nil {
		t.Fatalf("Failed to get model provider: %v", err)
	}
	
	if provider == nil {
		t.Error("Expected provider, got nil")
	}
	
	// Test listing
	providers := registry.ListModelProviders()
	if len(providers) != 1 || providers[0] != "mock" {
		t.Errorf("Expected ['mock'], got %v", providers)
	}
}

func TestModelRegistry_AllComponentTypes(t *testing.T) {
	registry := NewModelRegistry[float32]()
	
	// Register one of each component type
	providerFactory := func(ctx context.Context, config map[string]interface{}) (ModelProvider[float32], error) {
		return NewMockModelProvider[float32](), nil
	}
	serializerFactory := func(ctx context.Context, config map[string]interface{}) (ModelSerializer[float32], error) {
		return nil, nil // Mock implementation
	}
	loaderFactory := func(ctx context.Context, config map[string]interface{}) (ModelLoader[float32], error) {
		return nil, nil // Mock implementation
	}
	
	registry.RegisterModelProvider("test_provider", providerFactory)
	registry.RegisterModelSerializer("test_serializer", serializerFactory)
	registry.RegisterModelLoader("test_loader", loaderFactory)
	
	// Test summary
	summary := registry.Summary()
	if summary["providers"] != 1 {
		t.Errorf("Expected 1 provider, got %d", summary["providers"])
	}
	if summary["serializers"] != 1 {
		t.Errorf("Expected 1 serializer, got %d", summary["serializers"])
	}
	if summary["loaders"] != 1 {
		t.Errorf("Expected 1 loader, got %d", summary["loaders"])
	}
	if summary["exporters"] != 0 {
		t.Errorf("Expected 0 exporters, got %d", summary["exporters"])
	}
	
	// Test GetAllRegistrations
	registrations := registry.GetAllRegistrations()
	if len(registrations["providers"]) != 1 || registrations["providers"][0] != "test_provider" {
		t.Errorf("Expected providers ['test_provider'], got %v", registrations["providers"])
	}
}

func TestModelRegistry_ThreadSafety(t *testing.T) {
	registry := NewModelRegistry[float32]()
	
	// Test concurrent registration and retrieval
	done := make(chan bool, 2)
	
	// Goroutine 1: Register providers
	go func() {
		defer func() { done <- true }()
		for i := 0; i < 50; i++ {
			factory := func(ctx context.Context, config map[string]interface{}) (ModelProvider[float32], error) {
				return NewMockModelProvider[float32](), nil
			}
			registry.RegisterModelProvider("provider_"+string(rune('a'+i%26)), factory)
		}
	}()
	
	// Goroutine 2: List providers
	go func() {
		defer func() { done <- true }()
		for i := 0; i < 50; i++ {
			registry.ListModelProviders()
		}
	}()
	
	// Wait for both goroutines
	<-done
	<-done
	
	// Verify final state
	providers := registry.ListModelProviders()
	if len(providers) == 0 {
		t.Error("Expected some providers to be registered")
	}
}

func TestMockModelInstance(t *testing.T) {
	instance := NewMockModelInstance[float32]()
	ctx := context.Background()
	
	// Test metadata
	metadata := instance.GetMetadata()
	if metadata.Name != "MockModel" {
		t.Errorf("Expected name 'MockModel', got %s", metadata.Name)
	}
	
	if len(metadata.InputShape) != 2 {
		t.Errorf("Expected 2 input shapes, got %d", len(metadata.InputShape))
	}
	
	// Test training mode
	if instance.IsTraining() {
		t.Error("Expected model to start in inference mode")
	}
	
	instance.SetTrainingMode(true)
	if !instance.IsTraining() {
		t.Error("Expected model to be in training mode")
	}
	
	// Test forward pass
	input, err := tensor.New[float32]([]int{1, 32}, nil)
	if err != nil {
		t.Fatalf("Failed to create input tensor: %v", err)
	}
	
	output, err := instance.Forward(ctx, input)
	if err != nil {
		t.Fatalf("Forward pass failed: %v", err)
	}
	
	if output == nil {
		t.Error("Expected output tensor, got nil")
	}
	
	// Test backward pass
	err = instance.Backward(ctx, input)
	if err != nil {
		t.Errorf("Backward pass failed: %v", err)
	}
	
	// Test parameters
	params := instance.Parameters()
	if params == nil {
		t.Error("Expected parameters slice, got nil")
	}
}

func TestMockModelProvider(t *testing.T) {
	provider := NewMockModelProvider[float32]()
	ctx := context.Background()
	
	// Test capabilities
	capabilities := provider.GetCapabilities()
	if len(capabilities.SupportedTypes) == 0 {
		t.Error("Expected supported types, got empty slice")
	}
	
	if !capabilities.SupportsTraining {
		t.Error("Expected provider to support training")
	}
	
	// Test provider info
	info := provider.GetProviderInfo()
	if info.Name != "Mock Model Provider" {
		t.Errorf("Expected 'Mock Model Provider', got %s", info.Name)
	}
	
	// Test model creation
	config := ModelConfig{
		Type:         "mock",
		TrainingMode: true,
		Extensions: map[string]interface{}{
			"test": "value",
		},
	}
	
	instance, err := provider.CreateModel(ctx, config)
	if err != nil {
		t.Fatalf("Failed to create model: %v", err)
	}
	
	if instance == nil {
		t.Error("Expected model instance, got nil")
	}
	
	if !instance.IsTraining() {
		t.Error("Expected created model to be in training mode")
	}
	
	// Test extensions
	metadata := instance.GetMetadata()
	if metadata.Extensions["test"] != "value" {
		t.Error("Expected extension to be preserved in model metadata")
	}
}

func TestStandardModelInstance_Integration(t *testing.T) {
	// Create a simple mock model
	model := &Model[float32]{
		ZMFVersion: "1.0",
	}
	
	instance := NewStandardModelInstance(model)
	
	// Test metadata generation
	metadata := instance.GetMetadata()
	if metadata.Framework != "zerfoo" {
		t.Errorf("Expected framework 'zerfoo', got %s", metadata.Framework)
	}
	
	if metadata.Version != "1.0" {
		t.Errorf("Expected version '1.0', got %s", metadata.Version)
	}
	
	// Test training mode
	instance.SetTrainingMode(true)
	if !instance.IsTraining() {
		t.Error("Expected instance to be in training mode")
	}
}

func TestModelConfig_Extensions(t *testing.T) {
	config := ModelConfig{
		Type:         "standard",
		Architecture: map[string]interface{}{"layers": 3},
		Parameters:   map[string]interface{}{"learning_rate": 0.001},
		TrainingMode: true,
		BatchSize:    32,
		Extensions: map[string]interface{}{
			"numerai": map[string]interface{}{
				"tournament_id": "main",
				"era_limit":     120,
			},
			"custom": map[string]interface{}{
				"feature_count": 1000,
				"target_type":   "regression",
			},
		},
	}
	
	// Test extension access
	if config.Extensions["numerai"] == nil {
		t.Error("Expected numerai extension to be present")
	}
	
	numeraiConfig, ok := config.Extensions["numerai"].(map[string]interface{})
	if !ok {
		t.Error("Expected numerai config to be map[string]interface{}")
	}
	
	tournamentID, ok := numeraiConfig["tournament_id"].(string)
	if !ok || tournamentID != "main" {
		t.Errorf("Expected tournament_id 'main', got %v", tournamentID)
	}
	
	// Test custom extension
	customConfig, ok := config.Extensions["custom"].(map[string]interface{})
	if !ok {
		t.Error("Expected custom config to be map[string]interface{}")
	}
	
	featureCount, ok := customConfig["feature_count"].(int)
	if !ok || featureCount != 1000 {
		t.Errorf("Expected feature_count 1000, got %v", featureCount)
	}
}

func TestBasicModelValidator(t *testing.T) {
	validator := NewBasicModelValidator[float32]()
	ctx := context.Background()
	
	// Test validator info
	info := validator.GetValidatorInfo()
	if info.Name != "Basic Model Validator" {
		t.Errorf("Expected 'Basic Model Validator', got %s", info.Name)
	}
	
	if len(info.CheckTypes) == 0 {
		t.Error("Expected some check types, got empty slice")
	}
	
	// Test input validation
	instance := NewMockModelInstance[float32]()
	
	// Create inputs with correct shape
	input1, err := tensor.New[float32]([]int{1, 32}, nil)
	if err != nil {
		t.Fatalf("Failed to create input1: %v", err)
	}
	
	input2, err := tensor.New[float32]([]int{1, 16}, nil)
	if err != nil {
		t.Fatalf("Failed to create input2: %v", err)
	}
	
	err = validator.ValidateInputs(ctx, instance, input1, input2)
	if err != nil {
		t.Errorf("Input validation failed: %v", err)
	}
	
	// Test with wrong number of inputs
	err = validator.ValidateInputs(ctx, instance, input1)
	if err == nil {
		t.Error("Expected error for wrong number of inputs")
	}
	
	// Test with wrong input shape
	wrongInput, err := tensor.New[float32]([]int{1, 64}, nil)
	if err != nil {
		t.Fatalf("Failed to create wrong input: %v", err)
	}
	
	err = validator.ValidateInputs(ctx, instance, wrongInput, input2)
	if err == nil {
		t.Error("Expected error for wrong input shape")
	}
}

func TestRegistryClear(t *testing.T) {
	registry := NewModelRegistry[float32]()
	
	// Add some components
	factory := func(ctx context.Context, config map[string]interface{}) (ModelProvider[float32], error) {
		return NewMockModelProvider[float32](), nil
	}
	
	registry.RegisterModelProvider("test", factory)
	
	summary := registry.Summary()
	if summary["providers"] != 1 {
		t.Errorf("Expected 1 provider before clear, got %d", summary["providers"])
	}
	
	// Clear registry
	registry.Clear()
	
	summaryAfter := registry.Summary()
	for component, count := range summaryAfter {
		if count != 0 {
			t.Errorf("Expected %s count to be 0 after clear, got %d", component, count)
		}
	}
}