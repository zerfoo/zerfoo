// Package training provides adapter implementations for bridging existing and new interfaces.
package training

import (
	"context"
	"fmt"

	"github.com/zerfoo/zerfoo/graph"
	"github.com/zerfoo/zerfoo/tensor"
	"github.com/zerfoo/zerfoo/training/optimizer"
)

// TrainerWorkflowAdapter adapts the existing Trainer interface to the new TrainingWorkflow interface.
// This allows legacy trainer implementations to work with the new generic workflow system.
type TrainerWorkflowAdapter[T tensor.Numeric] struct {
	trainer    Trainer[T]
	optimizer  optimizer.Optimizer[T]
	config     WorkflowConfig
	metrics    map[string]interface{}
}

// NewTrainerWorkflowAdapter creates a new adapter for legacy trainers.
func NewTrainerWorkflowAdapter[T tensor.Numeric](trainer Trainer[T], opt optimizer.Optimizer[T]) *TrainerWorkflowAdapter[T] {
	return &TrainerWorkflowAdapter[T]{
		trainer:   trainer,
		optimizer: opt,
		metrics:   make(map[string]interface{}),
	}
}

// Initialize implements TrainingWorkflow.Initialize
func (a *TrainerWorkflowAdapter[T]) Initialize(ctx context.Context, config WorkflowConfig) error {
	a.config = config
	return nil
}

// Train implements TrainingWorkflow.Train by adapting to the legacy Trainer interface
func (a *TrainerWorkflowAdapter[T]) Train(ctx context.Context, dataset DataProvider[T], modelProvider ModelProvider[T]) (*TrainingResult[T], error) {
	// Create model
	model, err := modelProvider.CreateModel(ctx, a.config.ModelConfig)
	if err != nil {
		return nil, fmt.Errorf("failed to create model: %w", err)
	}

	// Get training data
	dataIter, err := dataset.GetTrainingData(ctx, a.config.BatchConfig)
	if err != nil {
		return nil, fmt.Errorf("failed to get training data: %w", err)
	}
	defer dataIter.Close()

	var totalLoss T
	var bestLoss T
	bestEpoch := 0
	epoch := 0
	
	// Training loop
	for epoch < a.config.NumEpochs {
		epochLoss := T(0)
		batchCount := 0
		
		// Reset iterator for new epoch
		if err := dataIter.Reset(); err != nil {
			return nil, fmt.Errorf("failed to reset data iterator: %w", err)
		}
		
		// Process all batches in epoch
		for dataIter.Next(ctx) {
			batch := dataIter.Batch()
			if batch == nil {
				break
			}
			
			// Convert batch targets to the required format for legacy trainer
			targets := batch.Targets
			
			// Perform training step using legacy trainer
			stepLoss, err := a.trainer.TrainStep(ctx, model, a.optimizer, batch.Inputs, targets)
			if err != nil {
				return nil, fmt.Errorf("training step failed at epoch %d: %w", epoch, err)
			}
			
			epochLoss += stepLoss
			batchCount++
		}
		
		if err := dataIter.Error(); err != nil {
			return nil, fmt.Errorf("data iteration failed at epoch %d: %w", epoch, err)
		}
		
		// Calculate average loss for epoch
		if batchCount > 0 {
			epochLoss /= T(batchCount)
		}
		
		totalLoss = epochLoss
		
		// Track best loss
		if epoch == 0 || epochLoss < bestLoss {
			bestLoss = epochLoss
			bestEpoch = epoch
		}
		
		// Store metrics
		a.metrics[fmt.Sprintf("epoch_%d_loss", epoch)] = float64(epochLoss)
		
		epoch++
	}

	// Return training result
	result := &TrainingResult[T]{
		FinalLoss:   totalLoss,
		BestLoss:    bestLoss,
		BestEpoch:   bestEpoch,
		TotalEpochs: epoch,
		Metrics:     make(map[string]float64),
		Extensions:  make(map[string]interface{}),
	}

	// Convert metrics to float64 for result
	for key, value := range a.metrics {
		if floatVal, ok := value.(float64); ok {
			result.Metrics[key] = floatVal
		}
	}

	return result, nil
}

// Validate implements TrainingWorkflow.Validate
func (a *TrainerWorkflowAdapter[T]) Validate(ctx context.Context, dataset DataProvider[T], modelProvider ModelProvider[T]) (*ValidationResult[T], error) {
	// For the adapter, validation is simplified - we just run through validation data
	// without updating model parameters
	dataIter, err := dataset.GetValidationData(ctx, a.config.BatchConfig)
	if err != nil {
		return nil, fmt.Errorf("failed to get validation data: %w", err)
	}
	defer dataIter.Close()

	var totalLoss T
	sampleCount := 0

	for dataIter.Next(ctx) {
		batch := dataIter.Batch()
		if batch == nil {
			break
		}
		
		sampleCount++
		
		// Note: For validation, we would typically run forward pass only,
		// but legacy trainer interface doesn't provide this.
		// This is a limitation of the adapter approach.
		totalLoss += T(0) // Placeholder - actual validation would need model forward pass
	}

	if err := dataIter.Error(); err != nil {
		return nil, fmt.Errorf("validation data iteration failed: %w", err)
	}

	avgLoss := T(0)
	if sampleCount > 0 {
		avgLoss = totalLoss / T(sampleCount)
	}

	result := &ValidationResult[T]{
		Loss:        avgLoss,
		Metrics:     make(map[string]float64),
		SampleCount: sampleCount,
		Extensions:  make(map[string]interface{}),
	}

	return result, nil
}

// GetMetrics implements TrainingWorkflow.GetMetrics
func (a *TrainerWorkflowAdapter[T]) GetMetrics() map[string]interface{} {
	return a.metrics
}

// Shutdown implements TrainingWorkflow.Shutdown
func (a *TrainerWorkflowAdapter[T]) Shutdown(ctx context.Context) error {
	// Clear metrics
	a.metrics = make(map[string]interface{})
	return nil
}

// GradientStrategyAdapter adapts GradientStrategy to work with the new interface system.
type GradientStrategyAdapter[T tensor.Numeric] struct {
	strategy GradientStrategy[T]
	graph    *graph.Graph[T]
	lossNode graph.Node[T]
}

// NewGradientStrategyAdapter creates a new gradient strategy adapter.
func NewGradientStrategyAdapter[T tensor.Numeric](strategy GradientStrategy[T], g *graph.Graph[T], lossNode graph.Node[T]) *GradientStrategyAdapter[T] {
	return &GradientStrategyAdapter[T]{
		strategy: strategy,
		graph:    g,
		lossNode: lossNode,
	}
}

// ComputeGradientsFromBatch adapts batch processing to the legacy GradientStrategy interface.
func (a *GradientStrategyAdapter[T]) ComputeGradientsFromBatch(ctx context.Context, batch *Batch[T]) (T, error) {
	return a.strategy.ComputeGradients(ctx, a.graph, a.lossNode, *batch)
}

// DataIteratorAdapter provides a simple iterator implementation over static data.
type DataIteratorAdapter[T tensor.Numeric] struct {
	batches     []*Batch[T]
	currentIdx  int
	err         error
}

// NewDataIteratorAdapter creates a new data iterator from a slice of batches.
func NewDataIteratorAdapter[T tensor.Numeric](batches []*Batch[T]) *DataIteratorAdapter[T] {
	return &DataIteratorAdapter[T]{
		batches:    batches,
		currentIdx: -1,
	}
}

// Next implements DataIterator.Next
func (d *DataIteratorAdapter[T]) Next(ctx context.Context) bool {
	d.currentIdx++
	return d.currentIdx < len(d.batches)
}

// Batch implements DataIterator.Batch
func (d *DataIteratorAdapter[T]) Batch() *Batch[T] {
	if d.currentIdx < 0 || d.currentIdx >= len(d.batches) {
		return nil
	}
	return d.batches[d.currentIdx]
}

// Error implements DataIterator.Error
func (d *DataIteratorAdapter[T]) Error() error {
	return d.err
}

// Close implements DataIterator.Close
func (d *DataIteratorAdapter[T]) Close() error {
	return nil
}

// Reset implements DataIterator.Reset
func (d *DataIteratorAdapter[T]) Reset() error {
	d.currentIdx = -1
	d.err = nil
	return nil
}

// SimpleModelProvider provides a basic model provider implementation.
type SimpleModelProvider[T tensor.Numeric] struct {
	modelFactory func(ctx context.Context, config ModelConfig) (*graph.Graph[T], error)
	modelInfo    ModelInfo
}

// NewSimpleModelProvider creates a new simple model provider.
func NewSimpleModelProvider[T tensor.Numeric](
	factory func(ctx context.Context, config ModelConfig) (*graph.Graph[T], error),
	info ModelInfo,
) *SimpleModelProvider[T] {
	return &SimpleModelProvider[T]{
		modelFactory: factory,
		modelInfo:    info,
	}
}

// CreateModel implements ModelProvider.CreateModel
func (s *SimpleModelProvider[T]) CreateModel(ctx context.Context, config ModelConfig) (*graph.Graph[T], error) {
	if s.modelFactory == nil {
		return nil, fmt.Errorf("no model factory configured")
	}
	return s.modelFactory(ctx, config)
}

// LoadModel implements ModelProvider.LoadModel
func (s *SimpleModelProvider[T]) LoadModel(ctx context.Context, path string) (*graph.Graph[T], error) {
	return nil, fmt.Errorf("model loading not implemented in SimpleModelProvider")
}

// SaveModel implements ModelProvider.SaveModel
func (s *SimpleModelProvider[T]) SaveModel(ctx context.Context, model *graph.Graph[T], path string) error {
	return fmt.Errorf("model saving not implemented in SimpleModelProvider")
}

// GetModelInfo implements ModelProvider.GetModelInfo
func (s *SimpleModelProvider[T]) GetModelInfo() ModelInfo {
	return s.modelInfo
}

// Ensure adapters implement their respective interfaces
var _ TrainingWorkflow[float32] = (*TrainerWorkflowAdapter[float32])(nil)
var _ DataIterator[float32] = (*DataIteratorAdapter[float32])(nil)
var _ ModelProvider[float32] = (*SimpleModelProvider[float32])(nil)