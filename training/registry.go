// Package training provides a plugin registry for training components.
package training

import (
	"context"
	"fmt"
	"sync"

	"github.com/zerfoo/zerfoo/tensor"
)

// PluginRegistry manages registered training components and provides factory functions.
type PluginRegistry[T tensor.Numeric] struct {
	mu                 sync.RWMutex
	workflows          map[string]WorkflowFactory[T]
	dataProviders      map[string]DataProviderFactory[T]
	modelProviders     map[string]ModelProviderFactory[T]
	sequenceProviders  map[string]SequenceProviderFactory[T]
	metricComputers    map[string]MetricComputerFactory[T]
	crossValidators    map[string]CrossValidatorFactory[T]
}

// Factory function types for creating plugin instances

// WorkflowFactory creates TrainingWorkflow instances
type WorkflowFactory[T tensor.Numeric] func(ctx context.Context, config map[string]interface{}) (TrainingWorkflow[T], error)

// DataProviderFactory creates DataProvider instances
type DataProviderFactory[T tensor.Numeric] func(ctx context.Context, config map[string]interface{}) (DataProvider[T], error)

// ModelProviderFactory creates ModelProvider instances
type ModelProviderFactory[T tensor.Numeric] func(ctx context.Context, config map[string]interface{}) (ModelProvider[T], error)

// SequenceProviderFactory creates SequenceProvider instances
type SequenceProviderFactory[T tensor.Numeric] func(ctx context.Context, config map[string]interface{}) (SequenceProvider[T], error)

// MetricComputerFactory creates MetricComputer instances
type MetricComputerFactory[T tensor.Numeric] func(ctx context.Context, config map[string]interface{}) (MetricComputer[T], error)

// CrossValidatorFactory creates CrossValidator instances
type CrossValidatorFactory[T tensor.Numeric] func(ctx context.Context, config map[string]interface{}) (CrossValidator[T], error)

// Global registry instances for common numeric types
var (
	Float32Registry = NewPluginRegistry[float32]()
	Float64Registry = NewPluginRegistry[float64]()
)

// NewPluginRegistry creates a new plugin registry.
func NewPluginRegistry[T tensor.Numeric]() *PluginRegistry[T] {
	return &PluginRegistry[T]{
		workflows:         make(map[string]WorkflowFactory[T]),
		dataProviders:     make(map[string]DataProviderFactory[T]),
		modelProviders:    make(map[string]ModelProviderFactory[T]),
		sequenceProviders: make(map[string]SequenceProviderFactory[T]),
		metricComputers:   make(map[string]MetricComputerFactory[T]),
		crossValidators:   make(map[string]CrossValidatorFactory[T]),
	}
}

// Workflow registration methods

// RegisterWorkflow registers a training workflow factory.
func (r *PluginRegistry[T]) RegisterWorkflow(name string, factory WorkflowFactory[T]) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	
	if _, exists := r.workflows[name]; exists {
		return fmt.Errorf("workflow '%s' is already registered", name)
	}
	
	r.workflows[name] = factory
	return nil
}

// GetWorkflow retrieves a registered workflow factory and creates an instance.
func (r *PluginRegistry[T]) GetWorkflow(ctx context.Context, name string, config map[string]interface{}) (TrainingWorkflow[T], error) {
	r.mu.RLock()
	factory, exists := r.workflows[name]
	r.mu.RUnlock()
	
	if !exists {
		return nil, fmt.Errorf("workflow '%s' not registered", name)
	}
	
	return factory(ctx, config)
}

// ListWorkflows returns all registered workflow names.
func (r *PluginRegistry[T]) ListWorkflows() []string {
	r.mu.RLock()
	defer r.mu.RUnlock()
	
	names := make([]string, 0, len(r.workflows))
	for name := range r.workflows {
		names = append(names, name)
	}
	return names
}

// Data provider registration methods

// RegisterDataProvider registers a data provider factory.
func (r *PluginRegistry[T]) RegisterDataProvider(name string, factory DataProviderFactory[T]) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	
	if _, exists := r.dataProviders[name]; exists {
		return fmt.Errorf("data provider '%s' is already registered", name)
	}
	
	r.dataProviders[name] = factory
	return nil
}

// GetDataProvider retrieves a registered data provider factory and creates an instance.
func (r *PluginRegistry[T]) GetDataProvider(ctx context.Context, name string, config map[string]interface{}) (DataProvider[T], error) {
	r.mu.RLock()
	factory, exists := r.dataProviders[name]
	r.mu.RUnlock()
	
	if !exists {
		return nil, fmt.Errorf("data provider '%s' not registered", name)
	}
	
	return factory(ctx, config)
}

// ListDataProviders returns all registered data provider names.
func (r *PluginRegistry[T]) ListDataProviders() []string {
	r.mu.RLock()
	defer r.mu.RUnlock()
	
	names := make([]string, 0, len(r.dataProviders))
	for name := range r.dataProviders {
		names = append(names, name)
	}
	return names
}

// Model provider registration methods

// RegisterModelProvider registers a model provider factory.
func (r *PluginRegistry[T]) RegisterModelProvider(name string, factory ModelProviderFactory[T]) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	
	if _, exists := r.modelProviders[name]; exists {
		return fmt.Errorf("model provider '%s' is already registered", name)
	}
	
	r.modelProviders[name] = factory
	return nil
}

// GetModelProvider retrieves a registered model provider factory and creates an instance.
func (r *PluginRegistry[T]) GetModelProvider(ctx context.Context, name string, config map[string]interface{}) (ModelProvider[T], error) {
	r.mu.RLock()
	factory, exists := r.modelProviders[name]
	r.mu.RUnlock()
	
	if !exists {
		return nil, fmt.Errorf("model provider '%s' not registered", name)
	}
	
	return factory(ctx, config)
}

// ListModelProviders returns all registered model provider names.
func (r *PluginRegistry[T]) ListModelProviders() []string {
	r.mu.RLock()
	defer r.mu.RUnlock()
	
	names := make([]string, 0, len(r.modelProviders))
	for name := range r.modelProviders {
		names = append(names, name)
	}
	return names
}

// Sequence provider registration methods

// RegisterSequenceProvider registers a sequence provider factory.
func (r *PluginRegistry[T]) RegisterSequenceProvider(name string, factory SequenceProviderFactory[T]) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	
	if _, exists := r.sequenceProviders[name]; exists {
		return fmt.Errorf("sequence provider '%s' is already registered", name)
	}
	
	r.sequenceProviders[name] = factory
	return nil
}

// GetSequenceProvider retrieves a registered sequence provider factory and creates an instance.
func (r *PluginRegistry[T]) GetSequenceProvider(ctx context.Context, name string, config map[string]interface{}) (SequenceProvider[T], error) {
	r.mu.RLock()
	factory, exists := r.sequenceProviders[name]
	r.mu.RUnlock()
	
	if !exists {
		return nil, fmt.Errorf("sequence provider '%s' not registered", name)
	}
	
	return factory(ctx, config)
}

// ListSequenceProviders returns all registered sequence provider names.
func (r *PluginRegistry[T]) ListSequenceProviders() []string {
	r.mu.RLock()
	defer r.mu.RUnlock()
	
	names := make([]string, 0, len(r.sequenceProviders))
	for name := range r.sequenceProviders {
		names = append(names, name)
	}
	return names
}

// Metric computer registration methods

// RegisterMetricComputer registers a metric computer factory.
func (r *PluginRegistry[T]) RegisterMetricComputer(name string, factory MetricComputerFactory[T]) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	
	if _, exists := r.metricComputers[name]; exists {
		return fmt.Errorf("metric computer '%s' is already registered", name)
	}
	
	r.metricComputers[name] = factory
	return nil
}

// GetMetricComputer retrieves a registered metric computer factory and creates an instance.
func (r *PluginRegistry[T]) GetMetricComputer(ctx context.Context, name string, config map[string]interface{}) (MetricComputer[T], error) {
	r.mu.RLock()
	factory, exists := r.metricComputers[name]
	r.mu.RUnlock()
	
	if !exists {
		return nil, fmt.Errorf("metric computer '%s' not registered", name)
	}
	
	return factory(ctx, config)
}

// ListMetricComputers returns all registered metric computer names.
func (r *PluginRegistry[T]) ListMetricComputers() []string {
	r.mu.RLock()
	defer r.mu.RUnlock()
	
	names := make([]string, 0, len(r.metricComputers))
	for name := range r.metricComputers {
		names = append(names, name)
	}
	return names
}

// Cross validator registration methods

// RegisterCrossValidator registers a cross validator factory.
func (r *PluginRegistry[T]) RegisterCrossValidator(name string, factory CrossValidatorFactory[T]) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	
	if _, exists := r.crossValidators[name]; exists {
		return fmt.Errorf("cross validator '%s' is already registered", name)
	}
	
	r.crossValidators[name] = factory
	return nil
}

// GetCrossValidator retrieves a registered cross validator factory and creates an instance.
func (r *PluginRegistry[T]) GetCrossValidator(ctx context.Context, name string, config map[string]interface{}) (CrossValidator[T], error) {
	r.mu.RLock()
	factory, exists := r.crossValidators[name]
	r.mu.RUnlock()
	
	if !exists {
		return nil, fmt.Errorf("cross validator '%s' not registered", name)
	}
	
	return factory(ctx, config)
}

// ListCrossValidators returns all registered cross validator names.
func (r *PluginRegistry[T]) ListCrossValidators() []string {
	r.mu.RLock()
	defer r.mu.RUnlock()
	
	names := make([]string, 0, len(r.crossValidators))
	for name := range r.crossValidators {
		names = append(names, name)
	}
	return names
}

// Utility methods

// UnregisterWorkflow removes a workflow registration.
func (r *PluginRegistry[T]) UnregisterWorkflow(name string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	delete(r.workflows, name)
}

// UnregisterDataProvider removes a data provider registration.
func (r *PluginRegistry[T]) UnregisterDataProvider(name string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	delete(r.dataProviders, name)
}

// UnregisterModelProvider removes a model provider registration.
func (r *PluginRegistry[T]) UnregisterModelProvider(name string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	delete(r.modelProviders, name)
}

// UnregisterSequenceProvider removes a sequence provider registration.
func (r *PluginRegistry[T]) UnregisterSequenceProvider(name string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	delete(r.sequenceProviders, name)
}

// UnregisterMetricComputer removes a metric computer registration.
func (r *PluginRegistry[T]) UnregisterMetricComputer(name string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	delete(r.metricComputers, name)
}

// UnregisterCrossValidator removes a cross validator registration.
func (r *PluginRegistry[T]) UnregisterCrossValidator(name string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	delete(r.crossValidators, name)
}

// Clear removes all registrations.
func (r *PluginRegistry[T]) Clear() {
	r.mu.Lock()
	defer r.mu.Unlock()
	
	r.workflows = make(map[string]WorkflowFactory[T])
	r.dataProviders = make(map[string]DataProviderFactory[T])
	r.modelProviders = make(map[string]ModelProviderFactory[T])
	r.sequenceProviders = make(map[string]SequenceProviderFactory[T])
	r.metricComputers = make(map[string]MetricComputerFactory[T])
	r.crossValidators = make(map[string]CrossValidatorFactory[T])
}

// Summary returns a summary of all registered components.
func (r *PluginRegistry[T]) Summary() map[string]int {
	r.mu.RLock()
	defer r.mu.RUnlock()
	
	return map[string]int{
		"workflows":          len(r.workflows),
		"dataProviders":      len(r.dataProviders),
		"modelProviders":     len(r.modelProviders),
		"sequenceProviders":  len(r.sequenceProviders),
		"metricComputers":    len(r.metricComputers),
		"crossValidators":    len(r.crossValidators),
	}
}