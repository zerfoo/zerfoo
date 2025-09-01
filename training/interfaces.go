// Package training provides generic interfaces for ML training workflows.
package training

import (
	"context"

	"github.com/zerfoo/zerfoo/graph"
	"github.com/zerfoo/zerfoo/tensor"
)

// TrainingWorkflow orchestrates the complete training process with pluggable components.
// This interface allows domain-specific applications to customize training behavior
// while maintaining framework-agnostic core logic.
type TrainingWorkflow[T tensor.Numeric] interface {
	// Initialize prepares the workflow with configuration and dependencies
	Initialize(ctx context.Context, config WorkflowConfig) error

	// Train executes the complete training workflow
	Train(ctx context.Context, dataset DataProvider[T], model ModelProvider[T]) (*TrainingResult[T], error)

	// Validate performs validation on the trained model
	Validate(ctx context.Context, dataset DataProvider[T], model ModelProvider[T]) (*ValidationResult[T], error)

	// GetMetrics returns current training metrics
	GetMetrics() map[string]interface{}

	// Shutdown cleans up resources
	Shutdown(ctx context.Context) error
}

// DataProvider abstracts data access patterns for training and validation.
// This replaces domain-specific data loading with a generic interface.
type DataProvider[T tensor.Numeric] interface {
	// GetTrainingData returns training data in batches
	GetTrainingData(ctx context.Context, config BatchConfig) (DataIterator[T], error)

	// GetValidationData returns validation data in batches
	GetValidationData(ctx context.Context, config BatchConfig) (DataIterator[T], error)

	// GetMetadata returns dataset metadata for training customization
	GetMetadata() map[string]interface{}

	// Close releases any resources held by the provider
	Close() error
}

// DataIterator provides sequential access to training/validation batches.
type DataIterator[T tensor.Numeric] interface {
	// Next advances to the next batch, returns false when exhausted
	Next(ctx context.Context) bool

	// Batch returns the current batch data
	Batch() *Batch[T]

	// Error returns any error that occurred during iteration
	Error() error

	// Close releases iterator resources
	Close() error

	// Reset rewinds the iterator to the beginning
	Reset() error
}

// ModelProvider abstracts model creation and management.
// This allows different model architectures to be used with the same training workflow.
type ModelProvider[T tensor.Numeric] interface {
	// CreateModel creates a new model instance
	CreateModel(ctx context.Context, config ModelConfig) (*graph.Graph[T], error)

	// LoadModel loads a pre-trained model
	LoadModel(ctx context.Context, path string) (*graph.Graph[T], error)

	// SaveModel saves the current model state
	SaveModel(ctx context.Context, model *graph.Graph[T], path string) error

	// GetModelInfo returns model metadata
	GetModelInfo() ModelInfo
}

// SequenceProvider abstracts sequence generation for curriculum learning.
// This replaces the domain-specific EraSequencer with a generic interface.
type SequenceProvider[T tensor.Numeric] interface {
	// GenerateSequences creates training sequences from the dataset
	GenerateSequences(ctx context.Context, dataset DataProvider[T], config SequenceConfig) ([]DataProvider[T], error)

	// GenerateTrainValidationSplit creates train/validation splits
	GenerateTrainValidationSplit(ctx context.Context, dataset DataProvider[T], config SplitConfig) (DataProvider[T], DataProvider[T], error)

	// SetRandomSeed sets the random seed for reproducible sequence generation
	SetRandomSeed(seed uint64)
}

// MetricComputer provides extensible metric computation.
type MetricComputer[T tensor.Numeric] interface {
	// ComputeMetrics calculates metrics from predictions and targets
	ComputeMetrics(ctx context.Context, predictions, targets *tensor.TensorNumeric[T], metadata map[string]interface{}) (map[string]float64, error)

	// RegisterMetric adds a new metric computation
	RegisterMetric(name string, metric MetricFunction[T])

	// UnregisterMetric removes a metric computation
	UnregisterMetric(name string)

	// AvailableMetrics returns all registered metric names
	AvailableMetrics() []string
}

// MetricFunction defines a single metric computation.
type MetricFunction[T tensor.Numeric] func(ctx context.Context, predictions, targets *tensor.TensorNumeric[T], metadata map[string]interface{}) (float64, error)

// CrossValidator provides generic cross-validation strategies.
type CrossValidator[T tensor.Numeric] interface {
	// CreateFolds generates cross-validation folds from the dataset
	CreateFolds(ctx context.Context, dataset DataProvider[T], config CVConfig) ([]Fold[T], error)

	// ValidateModel performs cross-validation on a model
	ValidateModel(ctx context.Context, dataset DataProvider[T], modelProvider ModelProvider[T], config CVConfig) (*CVResult[T], error)
}

// Fold represents a single cross-validation fold.
type Fold[T tensor.Numeric] interface {
	// TrainData returns the training data for this fold
	TrainData() DataProvider[T]

	// ValidData returns the validation data for this fold
	ValidData() DataProvider[T]

	// FoldIndex returns the fold index
	FoldIndex() int

	// Metadata returns fold-specific metadata
	Metadata() map[string]interface{}
}

// Configuration structures for the generic interfaces

// WorkflowConfig configures the training workflow.
type WorkflowConfig struct {
	// Training configuration
	NumEpochs      int                    `json:"num_epochs"`
	LearningRate   float64                `json:"learning_rate"`
	EarlyStopTol   float64                `json:"early_stop_tolerance"`
	MaxNoImprove   int                    `json:"max_no_improve"`
	RandomSeed     uint64                 `json:"random_seed"`

	// Component configurations
	BatchConfig    BatchConfig            `json:"batch_config"`
	ModelConfig    ModelConfig            `json:"model_config"`
	MetricConfigs  map[string]interface{} `json:"metric_configs"`

	// Extension point for domain-specific configuration
	Extensions     map[string]interface{} `json:"extensions"`
}

// BatchConfig configures batch processing.
type BatchConfig struct {
	BatchSize      int                    `json:"batch_size"`
	Shuffle        bool                   `json:"shuffle"`
	DropLast       bool                   `json:"drop_last"`
	NumWorkers     int                    `json:"num_workers"`
	Extensions     map[string]interface{} `json:"extensions"`
}

// ModelConfig configures model creation.
type ModelConfig struct {
	Type           string                 `json:"type"`
	Architecture   map[string]interface{} `json:"architecture"`
	Hyperparams    map[string]interface{} `json:"hyperparams"`
	Extensions     map[string]interface{} `json:"extensions"`
}

// SequenceConfig configures sequence generation.
type SequenceConfig struct {
	MaxSeqLen      int                    `json:"max_seq_len"`
	NumSequences   int                    `json:"num_sequences"`
	Strategy       string                 `json:"strategy"`       // "consecutive", "random", "curriculum"
	Extensions     map[string]interface{} `json:"extensions"`
}

// SplitConfig configures train/validation splitting.
type SplitConfig struct {
	ValidationRatio float64                `json:"validation_ratio"`
	Strategy        string                 `json:"strategy"`         // "random", "chronological", "stratified"
	RandomSeed      uint64                 `json:"random_seed"`
	Extensions      map[string]interface{} `json:"extensions"`
}

// CVConfig configures cross-validation.
type CVConfig struct {
	Strategy       string                 `json:"strategy"`        // "k_fold", "time_series", "group"
	NumFolds       int                    `json:"num_folds"`
	GroupBy        string                 `json:"group_by"`        // For group-based CV
	PurgeGap       int                    `json:"purge_gap"`       // For time-series CV
	TestSize       float64                `json:"test_size"`
	RandomSeed     uint64                 `json:"random_seed"`
	Extensions     map[string]interface{} `json:"extensions"`
}

// Result structures

// TrainingResult contains training outcome information.
type TrainingResult[T tensor.Numeric] struct {
	FinalLoss      T                      `json:"final_loss"`
	BestLoss       T                      `json:"best_loss"`
	BestEpoch      int                    `json:"best_epoch"`
	TotalEpochs    int                    `json:"total_epochs"`
	TrainingTime   float64                `json:"training_time_seconds"`
	Metrics        map[string]float64     `json:"metrics"`
	ModelPath      string                 `json:"model_path,omitempty"`
	Extensions     map[string]interface{} `json:"extensions"`
}

// ValidationResult contains validation outcome information.
type ValidationResult[T tensor.Numeric] struct {
	Loss           T                      `json:"loss"`
	Metrics        map[string]float64     `json:"metrics"`
	SampleCount    int                    `json:"sample_count"`
	ValidationTime float64                `json:"validation_time_seconds"`
	Extensions     map[string]interface{} `json:"extensions"`
}

// CVResult contains cross-validation results.
type CVResult[T tensor.Numeric] struct {
	MeanLoss       T                      `json:"mean_loss"`
	StdLoss        T                      `json:"std_loss"`
	MeanMetrics    map[string]float64     `json:"mean_metrics"`
	StdMetrics     map[string]float64     `json:"std_metrics"`
	FoldResults    []ValidationResult[T]  `json:"fold_results"`
	TotalTime      float64                `json:"total_time_seconds"`
	Extensions     map[string]interface{} `json:"extensions"`
}

// ModelInfo contains model metadata.
type ModelInfo struct {
	Name           string                 `json:"name"`
	Version        string                 `json:"version"`
	Architecture   string                 `json:"architecture"`
	Parameters     int64                  `json:"parameter_count"`
	InputShape     []int                  `json:"input_shape"`
	OutputShape    []int                  `json:"output_shape"`
	Extensions     map[string]interface{} `json:"extensions"`
}