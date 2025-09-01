// Package model provides a comprehensive model registry for managing pluggable model components.
package model

import (
	"context"
	"fmt"
	"sync"

	"github.com/zerfoo/zerfoo/tensor"
)

// ModelRegistry manages registered model components and provides factory functions.
// This registry enables runtime component selection and supports multiple implementations
// of each model interface.
type ModelRegistry[T tensor.Numeric] struct {
	mu           sync.RWMutex
	providers    map[string]ModelProviderFactory[T]
	serializers  map[string]ModelSerializerFactory[T]
	loaders      map[string]ModelLoaderFactory[T]
	exporters    map[string]ModelExporterFactory[T]
	validators   map[string]ModelValidatorFactory[T]
	optimizers   map[string]ModelOptimizerFactory[T]
}

// Factory function types for creating model component instances

// ModelProviderFactory creates ModelProvider instances
type ModelProviderFactory[T tensor.Numeric] func(ctx context.Context, config map[string]interface{}) (ModelProvider[T], error)

// ModelSerializerFactory creates ModelSerializer instances
type ModelSerializerFactory[T tensor.Numeric] func(ctx context.Context, config map[string]interface{}) (ModelSerializer[T], error)

// ModelLoaderFactory creates ModelLoader instances
type ModelLoaderFactory[T tensor.Numeric] func(ctx context.Context, config map[string]interface{}) (ModelLoader[T], error)

// ModelExporterFactory creates ModelExporter instances
type ModelExporterFactory[T tensor.Numeric] func(ctx context.Context, config map[string]interface{}) (ModelExporter[T], error)

// ModelValidatorFactory creates ModelValidator instances
type ModelValidatorFactory[T tensor.Numeric] func(ctx context.Context, config map[string]interface{}) (ModelValidator[T], error)

// ModelOptimizerFactory creates ModelOptimizer instances
type ModelOptimizerFactory[T tensor.Numeric] func(ctx context.Context, config map[string]interface{}) (ModelOptimizer[T], error)

// Global registry instances for common numeric types
var (
	Float32ModelRegistry = NewModelRegistry[float32]()
	Float64ModelRegistry = NewModelRegistry[float64]()
)

// NewModelRegistry creates a new model registry.
func NewModelRegistry[T tensor.Numeric]() *ModelRegistry[T] {
	return &ModelRegistry[T]{
		providers:   make(map[string]ModelProviderFactory[T]),
		serializers: make(map[string]ModelSerializerFactory[T]),
		loaders:     make(map[string]ModelLoaderFactory[T]),
		exporters:   make(map[string]ModelExporterFactory[T]),
		validators:  make(map[string]ModelValidatorFactory[T]),
		optimizers:  make(map[string]ModelOptimizerFactory[T]),
	}
}

// Model provider registration methods

// RegisterModelProvider registers a model provider factory.
func (r *ModelRegistry[T]) RegisterModelProvider(name string, factory ModelProviderFactory[T]) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	
	if _, exists := r.providers[name]; exists {
		return fmt.Errorf("model provider '%s' is already registered", name)
	}
	
	r.providers[name] = factory
	return nil
}

// GetModelProvider retrieves a registered model provider factory and creates an instance.
func (r *ModelRegistry[T]) GetModelProvider(ctx context.Context, name string, config map[string]interface{}) (ModelProvider[T], error) {
	r.mu.RLock()
	factory, exists := r.providers[name]
	r.mu.RUnlock()
	
	if !exists {
		return nil, fmt.Errorf("model provider '%s' not registered", name)
	}
	
	return factory(ctx, config)
}

// ListModelProviders returns all registered model provider names.
func (r *ModelRegistry[T]) ListModelProviders() []string {
	r.mu.RLock()
	defer r.mu.RUnlock()
	
	names := make([]string, 0, len(r.providers))
	for name := range r.providers {
		names = append(names, name)
	}
	return names
}

// Model serializer registration methods

// RegisterModelSerializer registers a model serializer factory.
func (r *ModelRegistry[T]) RegisterModelSerializer(name string, factory ModelSerializerFactory[T]) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	
	if _, exists := r.serializers[name]; exists {
		return fmt.Errorf("model serializer '%s' is already registered", name)
	}
	
	r.serializers[name] = factory
	return nil
}

// GetModelSerializer retrieves a registered model serializer factory and creates an instance.
func (r *ModelRegistry[T]) GetModelSerializer(ctx context.Context, name string, config map[string]interface{}) (ModelSerializer[T], error) {
	r.mu.RLock()
	factory, exists := r.serializers[name]
	r.mu.RUnlock()
	
	if !exists {
		return nil, fmt.Errorf("model serializer '%s' not registered", name)
	}
	
	return factory(ctx, config)
}

// ListModelSerializers returns all registered model serializer names.
func (r *ModelRegistry[T]) ListModelSerializers() []string {
	r.mu.RLock()
	defer r.mu.RUnlock()
	
	names := make([]string, 0, len(r.serializers))
	for name := range r.serializers {
		names = append(names, name)
	}
	return names
}

// Model loader registration methods

// RegisterModelLoader registers a model loader factory.
func (r *ModelRegistry[T]) RegisterModelLoader(name string, factory ModelLoaderFactory[T]) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	
	if _, exists := r.loaders[name]; exists {
		return fmt.Errorf("model loader '%s' is already registered", name)
	}
	
	r.loaders[name] = factory
	return nil
}

// GetModelLoader retrieves a registered model loader factory and creates an instance.
func (r *ModelRegistry[T]) GetModelLoader(ctx context.Context, name string, config map[string]interface{}) (ModelLoader[T], error) {
	r.mu.RLock()
	factory, exists := r.loaders[name]
	r.mu.RUnlock()
	
	if !exists {
		return nil, fmt.Errorf("model loader '%s' not registered", name)
	}
	
	return factory(ctx, config)
}

// ListModelLoaders returns all registered model loader names.
func (r *ModelRegistry[T]) ListModelLoaders() []string {
	r.mu.RLock()
	defer r.mu.RUnlock()
	
	names := make([]string, 0, len(r.loaders))
	for name := range r.loaders {
		names = append(names, name)
	}
	return names
}

// Model exporter registration methods

// RegisterModelExporter registers a model exporter factory.
func (r *ModelRegistry[T]) RegisterModelExporter(name string, factory ModelExporterFactory[T]) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	
	if _, exists := r.exporters[name]; exists {
		return fmt.Errorf("model exporter '%s' is already registered", name)
	}
	
	r.exporters[name] = factory
	return nil
}

// GetModelExporter retrieves a registered model exporter factory and creates an instance.
func (r *ModelRegistry[T]) GetModelExporter(ctx context.Context, name string, config map[string]interface{}) (ModelExporter[T], error) {
	r.mu.RLock()
	factory, exists := r.exporters[name]
	r.mu.RUnlock()
	
	if !exists {
		return nil, fmt.Errorf("model exporter '%s' not registered", name)
	}
	
	return factory(ctx, config)
}

// ListModelExporters returns all registered model exporter names.
func (r *ModelRegistry[T]) ListModelExporters() []string {
	r.mu.RLock()
	defer r.mu.RUnlock()
	
	names := make([]string, 0, len(r.exporters))
	for name := range r.exporters {
		names = append(names, name)
	}
	return names
}

// Model validator registration methods

// RegisterModelValidator registers a model validator factory.
func (r *ModelRegistry[T]) RegisterModelValidator(name string, factory ModelValidatorFactory[T]) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	
	if _, exists := r.validators[name]; exists {
		return fmt.Errorf("model validator '%s' is already registered", name)
	}
	
	r.validators[name] = factory
	return nil
}

// GetModelValidator retrieves a registered model validator factory and creates an instance.
func (r *ModelRegistry[T]) GetModelValidator(ctx context.Context, name string, config map[string]interface{}) (ModelValidator[T], error) {
	r.mu.RLock()
	factory, exists := r.validators[name]
	r.mu.RUnlock()
	
	if !exists {
		return nil, fmt.Errorf("model validator '%s' not registered", name)
	}
	
	return factory(ctx, config)
}

// ListModelValidators returns all registered model validator names.
func (r *ModelRegistry[T]) ListModelValidators() []string {
	r.mu.RLock()
	defer r.mu.RUnlock()
	
	names := make([]string, 0, len(r.validators))
	for name := range r.validators {
		names = append(names, name)
	}
	return names
}

// Model optimizer registration methods

// RegisterModelOptimizer registers a model optimizer factory.
func (r *ModelRegistry[T]) RegisterModelOptimizer(name string, factory ModelOptimizerFactory[T]) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	
	if _, exists := r.optimizers[name]; exists {
		return fmt.Errorf("model optimizer '%s' is already registered", name)
	}
	
	r.optimizers[name] = factory
	return nil
}

// GetModelOptimizer retrieves a registered model optimizer factory and creates an instance.
func (r *ModelRegistry[T]) GetModelOptimizer(ctx context.Context, name string, config map[string]interface{}) (ModelOptimizer[T], error) {
	r.mu.RLock()
	factory, exists := r.optimizers[name]
	r.mu.RUnlock()
	
	if !exists {
		return nil, fmt.Errorf("model optimizer '%s' not registered", name)
	}
	
	return factory(ctx, config)
}

// ListModelOptimizers returns all registered model optimizer names.
func (r *ModelRegistry[T]) ListModelOptimizers() []string {
	r.mu.RLock()
	defer r.mu.RUnlock()
	
	names := make([]string, 0, len(r.optimizers))
	for name := range r.optimizers {
		names = append(names, name)
	}
	return names
}

// Utility methods

// UnregisterModelProvider removes a model provider registration.
func (r *ModelRegistry[T]) UnregisterModelProvider(name string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	delete(r.providers, name)
}

// UnregisterModelSerializer removes a model serializer registration.
func (r *ModelRegistry[T]) UnregisterModelSerializer(name string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	delete(r.serializers, name)
}

// UnregisterModelLoader removes a model loader registration.
func (r *ModelRegistry[T]) UnregisterModelLoader(name string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	delete(r.loaders, name)
}

// UnregisterModelExporter removes a model exporter registration.
func (r *ModelRegistry[T]) UnregisterModelExporter(name string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	delete(r.exporters, name)
}

// UnregisterModelValidator removes a model validator registration.
func (r *ModelRegistry[T]) UnregisterModelValidator(name string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	delete(r.validators, name)
}

// UnregisterModelOptimizer removes a model optimizer registration.
func (r *ModelRegistry[T]) UnregisterModelOptimizer(name string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	delete(r.optimizers, name)
}

// Clear removes all registrations.
func (r *ModelRegistry[T]) Clear() {
	r.mu.Lock()
	defer r.mu.Unlock()
	
	r.providers = make(map[string]ModelProviderFactory[T])
	r.serializers = make(map[string]ModelSerializerFactory[T])
	r.loaders = make(map[string]ModelLoaderFactory[T])
	r.exporters = make(map[string]ModelExporterFactory[T])
	r.validators = make(map[string]ModelValidatorFactory[T])
	r.optimizers = make(map[string]ModelOptimizerFactory[T])
}

// Summary returns a summary of all registered components.
func (r *ModelRegistry[T]) Summary() map[string]int {
	r.mu.RLock()
	defer r.mu.RUnlock()
	
	return map[string]int{
		"providers":   len(r.providers),
		"serializers": len(r.serializers),
		"loaders":     len(r.loaders),
		"exporters":   len(r.exporters),
		"validators":  len(r.validators),
		"optimizers":  len(r.optimizers),
	}
}

// GetAllRegistrations returns all registered component names by type.
func (r *ModelRegistry[T]) GetAllRegistrations() map[string][]string {
	r.mu.RLock()
	defer r.mu.RUnlock()
	
	// Extract names for each component type
	providerNames := make([]string, 0, len(r.providers))
	for name := range r.providers {
		providerNames = append(providerNames, name)
	}
	
	serializerNames := make([]string, 0, len(r.serializers))
	for name := range r.serializers {
		serializerNames = append(serializerNames, name)
	}
	
	loaderNames := make([]string, 0, len(r.loaders))
	for name := range r.loaders {
		loaderNames = append(loaderNames, name)
	}
	
	exporterNames := make([]string, 0, len(r.exporters))
	for name := range r.exporters {
		exporterNames = append(exporterNames, name)
	}
	
	validatorNames := make([]string, 0, len(r.validators))
	for name := range r.validators {
		validatorNames = append(validatorNames, name)
	}
	
	optimizerNames := make([]string, 0, len(r.optimizers))
	for name := range r.optimizers {
		optimizerNames = append(optimizerNames, name)
	}
	
	return map[string][]string{
		"providers":   providerNames,
		"serializers": serializerNames,
		"loaders":     loaderNames,
		"exporters":   exporterNames,
		"validators":  validatorNames,
		"optimizers":  optimizerNames,
	}
}

// FindProviderByCapability finds providers that support specific capabilities.
func (r *ModelRegistry[T]) FindProviderByCapability(ctx context.Context, requirement string) ([]string, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	
	var matching []string
	
	for name, factory := range r.providers {
		provider, err := factory(ctx, nil)
		if err != nil {
			continue // Skip providers that can't be instantiated
		}
		
		capabilities := provider.GetCapabilities()
		
		// Check if requirement is in supported types or other capability fields
		for _, supportedType := range capabilities.SupportedTypes {
			if supportedType == requirement {
				matching = append(matching, name)
				break
			}
		}
	}
	
	return matching, nil
}