package training

import (
	"context"
	"fmt"
	"testing"

	"github.com/zerfoo/zerfoo/graph"
	"github.com/zerfoo/zerfoo/tensor"
)

// Mock implementations for testing

// MockDataProvider implements DataProvider for testing
type MockDataProvider[T tensor.Numeric] struct {
	trainingBatches   []*Batch[T]
	validationBatches []*Batch[T]
	metadata          map[string]interface{}
	closed            bool
}

func NewMockDataProvider[T tensor.Numeric](trainBatches, validBatches []*Batch[T]) *MockDataProvider[T] {
	return &MockDataProvider[T]{
		trainingBatches:   trainBatches,
		validationBatches: validBatches,
		metadata:          make(map[string]interface{}),
	}
}

func (m *MockDataProvider[T]) GetTrainingData(ctx context.Context, config BatchConfig) (DataIterator[T], error) {
	return NewDataIteratorAdapter[T](m.trainingBatches), nil
}

func (m *MockDataProvider[T]) GetValidationData(ctx context.Context, config BatchConfig) (DataIterator[T], error) {
	return NewDataIteratorAdapter[T](m.validationBatches), nil
}

func (m *MockDataProvider[T]) GetMetadata() map[string]interface{} {
	return m.metadata
}

func (m *MockDataProvider[T]) Close() error {
	m.closed = true
	return nil
}

// MockModelProvider implements ModelProvider for testing
type MockModelProvider[T tensor.Numeric] struct {
	modelGraph *graph.Graph[T]
	modelInfo  ModelInfo
}

func NewMockModelProvider[T tensor.Numeric](g *graph.Graph[T]) *MockModelProvider[T] {
	return &MockModelProvider[T]{
		modelGraph: g,
		modelInfo: ModelInfo{
			Name:         "MockModel",
			Version:      "1.0",
			Architecture: "mock",
		},
	}
}

func (m *MockModelProvider[T]) CreateModel(ctx context.Context, config ModelConfig) (*graph.Graph[T], error) {
	return m.modelGraph, nil
}

func (m *MockModelProvider[T]) LoadModel(ctx context.Context, path string) (*graph.Graph[T], error) {
	return m.modelGraph, nil
}

func (m *MockModelProvider[T]) SaveModel(ctx context.Context, model *graph.Graph[T], path string) error {
	return nil
}

func (m *MockModelProvider[T]) GetModelInfo() ModelInfo {
	return m.modelInfo
}

// MockTrainingWorkflow implements TrainingWorkflow for testing
type MockTrainingWorkflow[T tensor.Numeric] struct {
	initialized bool
	metrics     map[string]interface{}
}

func NewMockTrainingWorkflow[T tensor.Numeric]() *MockTrainingWorkflow[T] {
	return &MockTrainingWorkflow[T]{
		metrics: make(map[string]interface{}),
	}
}

func (m *MockTrainingWorkflow[T]) Initialize(ctx context.Context, config WorkflowConfig) error {
	m.initialized = true
	return nil
}

func (m *MockTrainingWorkflow[T]) Train(ctx context.Context, dataset DataProvider[T], model ModelProvider[T]) (*TrainingResult[T], error) {
	var finalLoss, bestLoss T
	
	// Use type switch to handle different numeric types
	switch any(finalLoss).(type) {
	case float32:
		finalLoss = any(float32(0.5)).(T)
		bestLoss = any(float32(0.3)).(T)
	case float64:
		finalLoss = any(float64(0.5)).(T)
		bestLoss = any(float64(0.3)).(T)
	default:
		// For other types (like int), use zero values
		finalLoss = T(0)
		bestLoss = T(0)
	}
	
	return &TrainingResult[T]{
		FinalLoss:    finalLoss,
		BestLoss:     bestLoss,
		BestEpoch:    10,
		TotalEpochs:  20,
		TrainingTime: 100.0,
		Metrics:      map[string]float64{"accuracy": 0.95},
		Extensions:   make(map[string]interface{}),
	}, nil
}

func (m *MockTrainingWorkflow[T]) Validate(ctx context.Context, dataset DataProvider[T], model ModelProvider[T]) (*ValidationResult[T], error) {
	var loss T
	switch any(loss).(type) {
	case float32:
		loss = any(float32(0.4)).(T)
	case float64:
		loss = any(float64(0.4)).(T)
	default:
		loss = T(0)
	}
	
	return &ValidationResult[T]{
		Loss:           loss,
		Metrics:        map[string]float64{"accuracy": 0.93},
		SampleCount:    1000,
		ValidationTime: 20.0,
		Extensions:     make(map[string]interface{}),
	}, nil
}

func (m *MockTrainingWorkflow[T]) GetMetrics() map[string]interface{} {
	return m.metrics
}

func (m *MockTrainingWorkflow[T]) Shutdown(ctx context.Context) error {
	m.initialized = false
	return nil
}

// Test functions

func TestPluginRegistry_WorkflowRegistration(t *testing.T) {
	registry := NewPluginRegistry[float32]()
	
	// Test registration
	factory := func(ctx context.Context, config map[string]interface{}) (TrainingWorkflow[float32], error) {
		return NewMockTrainingWorkflow[float32](), nil
	}
	
	err := registry.RegisterWorkflow("mock", factory)
	if err != nil {
		t.Fatalf("Failed to register workflow: %v", err)
	}
	
	// Test duplicate registration
	err = registry.RegisterWorkflow("mock", factory)
	if err == nil {
		t.Error("Expected error for duplicate registration, got nil")
	}
	
	// Test retrieval
	ctx := context.Background()
	workflow, err := registry.GetWorkflow(ctx, "mock", nil)
	if err != nil {
		t.Fatalf("Failed to get workflow: %v", err)
	}
	
	if workflow == nil {
		t.Error("Expected workflow, got nil")
	}
	
	// Test non-existent workflow
	_, err = registry.GetWorkflow(ctx, "nonexistent", nil)
	if err == nil {
		t.Error("Expected error for non-existent workflow, got nil")
	}
	
	// Test listing
	workflows := registry.ListWorkflows()
	if len(workflows) != 1 || workflows[0] != "mock" {
		t.Errorf("Expected ['mock'], got %v", workflows)
	}
}

func TestPluginRegistry_DataProviderRegistration(t *testing.T) {
	registry := NewPluginRegistry[float32]()
	
	factory := func(ctx context.Context, config map[string]interface{}) (DataProvider[float32], error) {
		return NewMockDataProvider[float32](nil, nil), nil
	}
	
	// Test registration and retrieval
	err := registry.RegisterDataProvider("mock", factory)
	if err != nil {
		t.Fatalf("Failed to register data provider: %v", err)
	}
	
	ctx := context.Background()
	provider, err := registry.GetDataProvider(ctx, "mock", nil)
	if err != nil {
		t.Fatalf("Failed to get data provider: %v", err)
	}
	
	if provider == nil {
		t.Error("Expected data provider, got nil")
	}
	
	// Test cleanup
	err = provider.Close()
	if err != nil {
		t.Errorf("Failed to close data provider: %v", err)
	}
}

func TestDataIteratorAdapter(t *testing.T) {
	// Create test batches
	batch1 := &Batch[float32]{
		Inputs:  make(map[graph.Node[float32]]*tensor.TensorNumeric[float32]),
		Targets: nil,
	}
	batch2 := &Batch[float32]{
		Inputs:  make(map[graph.Node[float32]]*tensor.TensorNumeric[float32]),
		Targets: nil,
	}
	
	batches := []*Batch[float32]{batch1, batch2}
	iterator := NewDataIteratorAdapter[float32](batches)
	
	ctx := context.Background()
	
	// Test iteration
	count := 0
	for iterator.Next(ctx) {
		batch := iterator.Batch()
		if batch == nil {
			t.Error("Expected batch, got nil")
		}
		count++
	}
	
	if count != 2 {
		t.Errorf("Expected 2 batches, got %d", count)
	}
	
	// Test reset
	err := iterator.Reset()
	if err != nil {
		t.Errorf("Failed to reset iterator: %v", err)
	}
	
	// Test iteration after reset
	if !iterator.Next(ctx) {
		t.Error("Expected next to return true after reset")
	}
	
	// Test close
	err = iterator.Close()
	if err != nil {
		t.Errorf("Failed to close iterator: %v", err)
	}
}

func TestMockDataProvider(t *testing.T) {
	trainBatches := []*Batch[float32]{
		{Inputs: make(map[graph.Node[float32]]*tensor.TensorNumeric[float32])},
	}
	validBatches := []*Batch[float32]{
		{Inputs: make(map[graph.Node[float32]]*tensor.TensorNumeric[float32])},
	}
	
	provider := NewMockDataProvider[float32](trainBatches, validBatches)
	ctx := context.Background()
	config := BatchConfig{BatchSize: 32}
	
	// Test training data
	trainIter, err := provider.GetTrainingData(ctx, config)
	if err != nil {
		t.Fatalf("Failed to get training data: %v", err)
	}
	
	if !trainIter.Next(ctx) {
		t.Error("Expected training data to have at least one batch")
	}
	
	err = trainIter.Close()
	if err != nil {
		t.Errorf("Failed to close training iterator: %v", err)
	}
	
	// Test validation data
	validIter, err := provider.GetValidationData(ctx, config)
	if err != nil {
		t.Fatalf("Failed to get validation data: %v", err)
	}
	
	if !validIter.Next(ctx) {
		t.Error("Expected validation data to have at least one batch")
	}
	
	err = validIter.Close()
	if err != nil {
		t.Errorf("Failed to close validation iterator: %v", err)
	}
	
	// Test metadata
	metadata := provider.GetMetadata()
	if metadata == nil {
		t.Error("Expected metadata, got nil")
	}
	
	// Test close
	err = provider.Close()
	if err != nil {
		t.Errorf("Failed to close provider: %v", err)
	}
	
	// Verify close state - access the closed field directly since we know the concrete type
	mockProvider := provider
	if !mockProvider.closed {
		t.Error("Expected provider to be closed")
	}
}

func TestMockTrainingWorkflow(t *testing.T) {
	workflow := NewMockTrainingWorkflow[float32]()
	ctx := context.Background()
	
	// Test initialization
	config := WorkflowConfig{
		NumEpochs:    10,
		LearningRate: 0.001,
	}
	
	err := workflow.Initialize(ctx, config)
	if err != nil {
		t.Fatalf("Failed to initialize workflow: %v", err)
	}
	
	mockWorkflow := workflow
	if !mockWorkflow.initialized {
		t.Error("Expected workflow to be initialized")
	}
	
	// Test training
	dataProvider := NewMockDataProvider[float32](nil, nil)
	modelProvider := NewMockModelProvider[float32](nil)
	
	result, err := workflow.Train(ctx, dataProvider, modelProvider)
	if err != nil {
		t.Fatalf("Failed to train: %v", err)
	}
	
	if result == nil {
		t.Error("Expected training result, got nil")
	}
	
	if result.TotalEpochs != 20 {
		t.Errorf("Expected 20 total epochs, got %d", result.TotalEpochs)
	}
	
	// Test validation
	validResult, err := workflow.Validate(ctx, dataProvider, modelProvider)
	if err != nil {
		t.Fatalf("Failed to validate: %v", err)
	}
	
	if validResult == nil {
		t.Error("Expected validation result, got nil")
	}
	
	if validResult.SampleCount != 1000 {
		t.Errorf("Expected 1000 samples, got %d", validResult.SampleCount)
	}
	
	// Test metrics
	metrics := workflow.GetMetrics()
	if metrics == nil {
		t.Error("Expected metrics, got nil")
	}
	
	// Test shutdown
	err = workflow.Shutdown(ctx)
	if err != nil {
		t.Errorf("Failed to shutdown: %v", err)
	}
	
	if mockWorkflow.initialized {
		t.Error("Expected workflow to be shut down")
	}
}

func TestPluginRegistryThreadSafety(t *testing.T) {
	registry := NewPluginRegistry[float32]()
	
	// Test concurrent registration and retrieval
	done := make(chan bool, 2)
	
	// Goroutine 1: Register workflows
	go func() {
		defer func() { done <- true }()
		for i := 0; i < 100; i++ {
			factory := func(ctx context.Context, config map[string]interface{}) (TrainingWorkflow[float32], error) {
				return NewMockTrainingWorkflow[float32](), nil
			}
			registry.RegisterWorkflow(fmt.Sprintf("workflow_%d", i), factory)
		}
	}()
	
	// Goroutine 2: List workflows
	go func() {
		defer func() { done <- true }()
		for i := 0; i < 100; i++ {
			registry.ListWorkflows()
		}
	}()
	
	// Wait for both goroutines
	<-done
	<-done
	
	// Verify final state
	workflows := registry.ListWorkflows()
	if len(workflows) != 100 {
		t.Errorf("Expected 100 workflows, got %d", len(workflows))
	}
}

func TestConfigurationExtensions(t *testing.T) {
	config := WorkflowConfig{
		NumEpochs:    100,
		LearningRate: 0.01,
		Extensions: map[string]interface{}{
			"numerai": map[string]interface{}{
				"tournament_id": "main",
				"era_limit":     120,
			},
			"custom": map[string]interface{}{
				"feature_engineering": true,
				"ensemble_size":       5,
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
	
	// Test batch config extensions
	batchConfig := BatchConfig{
		BatchSize: 32,
		Extensions: map[string]interface{}{
			"data_augmentation": true,
			"prefetch_size":     4,
		},
	}
	
	if batchConfig.Extensions["data_augmentation"] != true {
		t.Error("Expected data_augmentation to be true")
	}
}

func TestRegistrySummary(t *testing.T) {
	registry := NewPluginRegistry[float32]()
	
	// Register components
	workflowFactory := func(ctx context.Context, config map[string]interface{}) (TrainingWorkflow[float32], error) {
		return NewMockTrainingWorkflow[float32](), nil
	}
	dataProviderFactory := func(ctx context.Context, config map[string]interface{}) (DataProvider[float32], error) {
		return NewMockDataProvider[float32](nil, nil), nil
	}
	
	registry.RegisterWorkflow("test_workflow", workflowFactory)
	registry.RegisterDataProvider("test_data", dataProviderFactory)
	
	// Test summary
	summary := registry.Summary()
	
	if summary["workflows"] != 1 {
		t.Errorf("Expected 1 workflow, got %d", summary["workflows"])
	}
	
	if summary["dataProviders"] != 1 {
		t.Errorf("Expected 1 data provider, got %d", summary["dataProviders"])
	}
	
	if summary["modelProviders"] != 0 {
		t.Errorf("Expected 0 model providers, got %d", summary["modelProviders"])
	}
	
	// Test clear
	registry.Clear()
	
	summaryAfterClear := registry.Summary()
	for component, count := range summaryAfterClear {
		if count != 0 {
			t.Errorf("Expected %s count to be 0 after clear, got %d", component, count)
		}
	}
}