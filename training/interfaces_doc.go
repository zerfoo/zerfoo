// Package training provides comprehensive documentation for the generic training interfaces.
//
// # Overview
//
// The training package implements a plugin-based architecture that allows domain-specific
// applications to customize training behavior while maintaining a generic, reusable core.
// This design follows the hexagonal architecture pattern, where the core framework defines
// ports (interfaces) and domain-specific applications provide adapters (implementations).
//
// # Core Design Principles
//
// 1. **Separation of Concerns**: Generic ML training logic is separated from domain-specific
//    business logic (e.g., Numerai tournament requirements).
//
// 2. **Dependency Inversion**: High-level training workflows depend on abstractions, not
//    concrete implementations.
//
// 3. **Plugin Architecture**: Components can be registered and swapped at runtime using
//    the plugin registry system.
//
// 4. **Backward Compatibility**: Adapter patterns allow legacy code to work with new interfaces.
//
// 5. **Extensibility**: Configuration structures include extension points for domain-specific
//    customization.
//
// # Interface Hierarchy
//
// ## Primary Interfaces
//
// ### TrainingWorkflow[T]
//
// The main orchestrator for training processes. Implementations define the complete training
// pipeline including initialization, training loops, validation, and cleanup.
//
// Usage Pattern:
//   workflow := registry.GetWorkflow(ctx, "standard", config)
//   workflow.Initialize(ctx, workflowConfig)
//   result := workflow.Train(ctx, dataProvider, modelProvider)
//
// ### DataProvider[T]
//
// Abstracts data access patterns for training and validation. This replaces domain-specific
// data loading with a generic interface that can handle any data source.
//
// Implementations should provide:
// - Efficient batch iteration
// - Train/validation data splitting
// - Metadata for training customization
// - Resource management (cleanup)
//
// Usage Pattern:
//   dataProvider := registry.GetDataProvider(ctx, "csv", config)
//   trainingData := dataProvider.GetTrainingData(ctx, batchConfig)
//   for trainingData.Next(ctx) {
//       batch := trainingData.Batch()
//       // Process batch
//   }
//
// ### ModelProvider[T]
//
// Abstracts model creation and management. This allows different model architectures
// to be used with the same training workflow.
//
// Implementations should handle:
// - Model instantiation from configuration
// - Model serialization/deserialization
// - Model metadata and introspection
//
// Usage Pattern:
//   modelProvider := registry.GetModelProvider(ctx, "mlp", config)
//   model := modelProvider.CreateModel(ctx, modelConfig)
//   modelProvider.SaveModel(ctx, model, "/path/to/model")
//
// ## Supporting Interfaces
//
// ### SequenceProvider[T]
//
// Generalizes sequence generation for curriculum learning. This replaces the domain-specific
// EraSequencer with a generic interface that can handle any sequencing strategy.
//
// Common implementations:
// - Consecutive sequence generation
// - Random sequence sampling
// - Curriculum learning strategies
// - Time-series aware sequencing
//
// ### MetricComputer[T]
//
// Provides extensible metric computation. Applications can register custom metrics
// while using standard training workflows.
//
// Usage Pattern:
//   computer := NewMetricComputer()
//   computer.RegisterMetric("mse", mseFuncition)
//   computer.RegisterMetric("custom_score", customFunction)
//   metrics := computer.ComputeMetrics(ctx, predictions, targets, metadata)
//
// ### CrossValidator[T]
//
// Implements various cross-validation strategies in a generic way. This allows
// time-series CV, k-fold CV, group-based CV, etc. to be used interchangeably.
//
// # Configuration System
//
// All interfaces use structured configuration with extension points:
//
// ## Extension Pattern
//
// All configuration structures include an `Extensions` field that allows
// domain-specific applications to pass additional configuration without
// modifying the core framework:
//
//   config := WorkflowConfig{
//       NumEpochs: 100,
//       Extensions: map[string]interface{}{
//           "numerai": map[string]interface{}{
//               "tournament_id": "numerai_main",
//               "era_limit": 120,
//           },
//       },
//   }
//
// ## Configuration Validation
//
// Implementations should validate their configuration and return descriptive
// errors for invalid or missing parameters.
//
// # Plugin Registry System
//
// The plugin registry enables runtime component selection and supports
// multiple implementations of each interface.
//
// ## Registration Pattern
//
//   // Register a component
//   err := Float32Registry.RegisterWorkflow("standard", func(ctx context.Context, config map[string]interface{}) (TrainingWorkflow[float32], error) {
//       return NewStandardWorkflow(config), nil
//   })
//
//   // Use registered component
//   workflow, err := Float32Registry.GetWorkflow(ctx, "standard", config)
//
// ## Factory Functions
//
// All registry factories receive context and configuration, enabling:
// - Initialization validation
// - Resource allocation
// - Graceful error handling
// - Context-aware cleanup
//
// # Adapter Pattern Implementation
//
// The adapter pattern allows smooth migration from legacy interfaces to new generic interfaces:
//
// ## TrainerWorkflowAdapter
//
// Adapts existing Trainer implementations to work with the TrainingWorkflow interface:
//
//   legacy := NewDefaultTrainer(graph, loss, optimizer, strategy)
//   adapter := NewTrainerWorkflowAdapter(legacy, optimizer)
//   // Now 'adapter' can be used with new workflow system
//
// ## Migration Strategy
//
// 1. Implement adapters for existing components
// 2. Register adapted components in plugin registry
// 3. Gradually replace legacy direct usage with registry-based usage
// 4. Implement new components directly against generic interfaces
//
// # Error Handling Patterns
//
// ## Context Cancellation
//
// All interface methods accept context.Context and should respect cancellation:
//
//   func (w *MyWorkflow) Train(ctx context.Context, ...) (*TrainingResult[T], error) {
//       for epoch := 0; epoch < maxEpochs; epoch++ {
//           select {
//           case <-ctx.Done():
//               return nil, ctx.Err()
//           default:
//               // Continue training
//           }
//       }
//   }
//
// ## Resource Cleanup
//
// Implementations should provide proper resource cleanup:
//
//   func (p *MyDataProvider) Close() error {
//       // Close files, database connections, etc.
//       return nil
//   }
//
// Use defer for automatic cleanup:
//
//   dataProvider := registry.GetDataProvider(ctx, "csv", config)
//   defer dataProvider.Close()
//
// # Performance Considerations
//
// ## Memory Management
//
// - Implement proper resource cleanup in Close() methods
// - Use streaming where possible for large datasets
// - Consider memory-mapped files for large data
//
// ## Concurrency
//
// - DataIterator implementations should be thread-safe
// - Plugin registry uses read-write mutexes for thread safety
// - Consider goroutine pools for parallel processing
//
// # Testing Patterns
//
// ## Mock Implementations
//
// Create mock implementations for testing:
//
//   type MockDataProvider[T] struct {
//       batches []*Batch[T]
//   }
//
//   func (m *MockDataProvider[T]) GetTrainingData(ctx context.Context, config BatchConfig) (DataIterator[T], error) {
//       return NewDataIteratorAdapter(m.batches), nil
//   }
//
// ## Integration Testing
//
// Test complete workflows with real implementations:
//
//   func TestWorkflowIntegration(t *testing.T) {
//       registry := NewPluginRegistry[float32]()
//       registry.RegisterWorkflow("test", testWorkflowFactory)
//       registry.RegisterDataProvider("mock", mockDataProviderFactory)
//       
//       workflow, _ := registry.GetWorkflow(ctx, "test", config)
//       dataProvider, _ := registry.GetDataProvider(ctx, "mock", dataConfig)
//       
//       result, err := workflow.Train(ctx, dataProvider, modelProvider)
//       assert.NoError(t, err)
//       assert.NotNil(t, result)
//   }
//
// # Migration from Domain-Specific Code
//
// ## Era-Specific Code Migration
//
// The EraSequencer has been migrated from zerfoo to audacity and replaced with
// the generic SequenceProvider interface:
//
// Before (domain-specific):
//   sequencer := NewEraSequencer(maxLen)
//   sequences := sequencer.GenerateSequences(dataset, numSeq)
//
// After (generic):
//   sequenceProvider, _ := registry.GetSequenceProvider(ctx, "consecutive", config)
//   sequences, _ := sequenceProvider.GenerateSequences(ctx, dataProvider, seqConfig)
//
// ## Configuration Migration
//
// Domain-specific configuration moves to extensions:
//
// Before:
//   config := NumeraiTrainingConfig{
//       Epochs: 100,
//       TournamentID: "main",
//       EraLimit: 120,
//   }
//
// After:
//   config := WorkflowConfig{
//       NumEpochs: 100,
//       Extensions: map[string]interface{}{
//           "numerai": map[string]interface{}{
//               "tournament_id": "main",
//               "era_limit": 120,
//           },
//       },
//   }
//
// # Best Practices
//
// ## Implementation Guidelines
//
// 1. **Validate Early**: Check configuration in Initialize() methods
// 2. **Fail Fast**: Return errors immediately for invalid states
// 3. **Resource Cleanup**: Always implement proper cleanup in Close() methods
// 4. **Context Awareness**: Respect context cancellation in long-running operations
// 5. **Thread Safety**: Ensure implementations are thread-safe where documented
//
// ## Extension Guidelines
//
// 1. **Namespace Extensions**: Use descriptive keys in Extensions maps
// 2. **Validate Extensions**: Check extension configuration early
// 3. **Provide Defaults**: Handle missing extension configuration gracefully
// 4. **Document Extensions**: Clearly document expected extension parameters
//
// ## Testing Guidelines
//
// 1. **Mock Dependencies**: Use mock implementations for unit testing
// 2. **Test Error Cases**: Ensure proper error handling and cleanup
// 3. **Integration Tests**: Test complete workflows with real data
// 4. **Performance Tests**: Benchmark critical paths with realistic data
//
// # Examples
//
// See the following files for complete implementation examples:
// - adapter.go: Adapter pattern implementations
// - registry.go: Plugin registry system
// - interfaces.go: Core interface definitions
//
// For domain-specific implementations, see the audacity project:
// - audacity/internal/training/: Era-specific implementations
// - audacity/internal/numerai/: Numerai tournament implementations
//
package training