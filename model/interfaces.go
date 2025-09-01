// Package model provides generic interfaces for model management and abstraction.
package model

import (
	"context"
	"io"

	"github.com/zerfoo/zerfoo/graph"
	"github.com/zerfoo/zerfoo/tensor"
)

// ModelProvider creates and manages model instances with pluggable architectures.
// This interface allows different model implementations to be used interchangeably
// while maintaining consistent creation and management patterns.
type ModelProvider[T tensor.Numeric] interface {
	// CreateModel creates a new model instance from configuration
	CreateModel(ctx context.Context, config ModelConfig) (ModelInstance[T], error)

	// CreateFromGraph creates a model instance from an existing graph
	CreateFromGraph(ctx context.Context, g *graph.Graph[T], config ModelConfig) (ModelInstance[T], error)

	// GetCapabilities returns the capabilities supported by this provider
	GetCapabilities() ModelCapabilities

	// GetProviderInfo returns metadata about this provider
	GetProviderInfo() ProviderInfo
}

// ModelInstance represents a specific model instance with inference and training capabilities.
type ModelInstance[T tensor.Numeric] interface {
	// Forward performs model inference
	Forward(ctx context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error)

	// Backward performs backpropagation (for training)
	Backward(ctx context.Context, inputs ...*tensor.TensorNumeric[T]) error

	// GetGraph returns the underlying computation graph
	GetGraph() *graph.Graph[T]

	// GetMetadata returns model metadata
	GetMetadata() ModelMetadata

	// Parameters returns model parameters for optimization
	Parameters() []*graph.Parameter[T]

	// SetTrainingMode sets the model to training or inference mode
	SetTrainingMode(training bool)

	// IsTraining returns whether the model is in training mode
	IsTraining() bool
}

// ModelSerializer handles model persistence in various formats.
type ModelSerializer[T tensor.Numeric] interface {
	// Save serializes a model to the specified path or writer
	Save(ctx context.Context, model ModelInstance[T], destination interface{}) error

	// Load deserializes a model from the specified path or reader
	Load(ctx context.Context, source interface{}) (ModelInstance[T], error)

	// GetSupportedFormats returns the file formats supported by this serializer
	GetSupportedFormats() []string

	// GetSerializerInfo returns metadata about this serializer
	GetSerializerInfo() SerializerInfo
}

// ModelLoader provides a generic interface for loading models from various sources.
type ModelLoader[T tensor.Numeric] interface {
	// LoadFromPath loads a model from a file path
	LoadFromPath(ctx context.Context, path string) (ModelInstance[T], error)

	// LoadFromReader loads a model from an io.Reader
	LoadFromReader(ctx context.Context, reader io.Reader) (ModelInstance[T], error)

	// LoadFromBytes loads a model from byte data
	LoadFromBytes(ctx context.Context, data []byte) (ModelInstance[T], error)

	// SupportsFormat returns whether the loader supports the given format
	SupportsFormat(format string) bool

	// GetLoaderInfo returns metadata about this loader
	GetLoaderInfo() LoaderInfo
}

// ModelExporter provides a generic interface for exporting models to various formats.
type ModelExporter[T tensor.Numeric] interface {
	// ExportToPath exports a model to a file path
	ExportToPath(ctx context.Context, model ModelInstance[T], path string) error

	// ExportToWriter exports a model to an io.Writer
	ExportToWriter(ctx context.Context, model ModelInstance[T], writer io.Writer) error

	// ExportToBytes exports a model to byte data
	ExportToBytes(ctx context.Context, model ModelInstance[T]) ([]byte, error)

	// SupportsFormat returns whether the exporter supports the given format
	SupportsFormat(format string) bool

	// GetExporterInfo returns metadata about this exporter
	GetExporterInfo() ExporterInfo
}

// ModelValidator validates model correctness and compatibility.
type ModelValidator[T tensor.Numeric] interface {
	// ValidateModel performs comprehensive model validation
	ValidateModel(ctx context.Context, model ModelInstance[T]) (*ValidationResult, error)

	// ValidateInputs checks if inputs are compatible with the model
	ValidateInputs(ctx context.Context, model ModelInstance[T], inputs ...*tensor.TensorNumeric[T]) error

	// ValidateArchitecture checks model architecture consistency
	ValidateArchitecture(ctx context.Context, model ModelInstance[T]) error

	// GetValidatorInfo returns metadata about this validator
	GetValidatorInfo() ValidatorInfo
}

// ModelOptimizer provides model optimization capabilities.
type ModelOptimizer[T tensor.Numeric] interface {
	// OptimizeModel applies optimizations to improve performance
	OptimizeModel(ctx context.Context, model ModelInstance[T], config OptimizationConfig) (ModelInstance[T], error)

	// GetOptimizations returns available optimization strategies
	GetOptimizations() []OptimizationStrategy

	// GetOptimizerInfo returns metadata about this optimizer
	GetOptimizerInfo() OptimizerInfo
}

// Configuration structures

// ModelConfig configures model creation and behavior.
type ModelConfig struct {
	// Core configuration
	Type          string                 `json:"type"`            // "standard", "hrm", "ensemble", etc.
	Architecture  map[string]interface{} `json:"architecture"`    // Architecture-specific parameters
	Parameters    map[string]interface{} `json:"parameters"`      // Model parameters
	
	// Behavior configuration
	TrainingMode  bool                   `json:"training_mode"`   // Whether to initialize in training mode
	BatchSize     int                    `json:"batch_size"`      // Default batch size for inference
	
	// Format and compatibility
	InputFormat   string                 `json:"input_format"`    // Expected input format
	OutputFormat  string                 `json:"output_format"`   // Expected output format
	Version       string                 `json:"version"`         // Model format version
	
	// Extension point for domain-specific configuration
	Extensions    map[string]interface{} `json:"extensions"`      // Domain-specific extensions
}

// OptimizationConfig configures model optimization.
type OptimizationConfig struct {
	Strategies    []string               `json:"strategies"`      // Optimization strategies to apply
	TargetDevice  string                 `json:"target_device"`   // Target device for optimization
	Precision     string                 `json:"precision"`       // Target precision (fp32, fp16, int8)
	MaxMemory     int64                  `json:"max_memory"`      // Memory constraints
	Extensions    map[string]interface{} `json:"extensions"`      // Strategy-specific options
}

// Metadata and information structures

// ModelMetadata contains information about a model instance.
type ModelMetadata struct {
	Name          string                 `json:"name"`
	Version       string                 `json:"version"`
	Architecture  string                 `json:"architecture"`
	Framework     string                 `json:"framework"`
	CreatedAt     string                 `json:"created_at"`
	ModifiedAt    string                 `json:"modified_at"`
	Parameters    int64                  `json:"parameter_count"`
	InputShape    [][]int                `json:"input_shapes"`
	OutputShape   []int                  `json:"output_shape"`
	Tags          []string               `json:"tags"`
	Extensions    map[string]interface{} `json:"extensions"`
}

// ModelCapabilities describes what a model provider can do.
type ModelCapabilities struct {
	SupportedTypes      []string `json:"supported_types"`
	SupportedPrecisions []string `json:"supported_precisions"`
	SupportsTraining    bool     `json:"supports_training"`
	SupportsInference   bool     `json:"supports_inference"`
	SupportsBatching    bool     `json:"supports_batching"`
	SupportsStreaming   bool     `json:"supports_streaming"`
	MaxBatchSize        int      `json:"max_batch_size"`
	MaxSequenceLength   int      `json:"max_sequence_length"`
}

// ProviderInfo contains metadata about a model provider.
type ProviderInfo struct {
	Name         string   `json:"name"`
	Version      string   `json:"version"`
	Description  string   `json:"description"`
	SupportedOps []string `json:"supported_operations"`
	Website      string   `json:"website"`
	License      string   `json:"license"`
}

// SerializerInfo contains metadata about a model serializer.
type SerializerInfo struct {
	Name            string   `json:"name"`
	Version         string   `json:"version"`
	Description     string   `json:"description"`
	SupportedFormats []string `json:"supported_formats"`
	Compression     bool     `json:"supports_compression"`
	Encryption      bool     `json:"supports_encryption"`
}

// LoaderInfo contains metadata about a model loader.
type LoaderInfo struct {
	Name            string   `json:"name"`
	Version         string   `json:"version"`
	Description     string   `json:"description"`
	SupportedFormats []string `json:"supported_formats"`
	StreamingLoad   bool     `json:"supports_streaming_load"`
	LazyLoad        bool     `json:"supports_lazy_load"`
}

// ExporterInfo contains metadata about a model exporter.
type ExporterInfo struct {
	Name            string   `json:"name"`
	Version         string   `json:"version"`
	Description     string   `json:"description"`
	SupportedFormats []string `json:"supported_formats"`
	Optimization    bool     `json:"supports_optimization"`
	Quantization    bool     `json:"supports_quantization"`
}

// ValidatorInfo contains metadata about a model validator.
type ValidatorInfo struct {
	Name         string   `json:"name"`
	Version      string   `json:"version"`
	Description  string   `json:"description"`
	CheckTypes   []string `json:"check_types"`
	Strictness   string   `json:"strictness"`
}

// OptimizerInfo contains metadata about a model optimizer.
type OptimizerInfo struct {
	Name         string   `json:"name"`
	Version      string   `json:"version"`
	Description  string   `json:"description"`
	Strategies   []string `json:"available_strategies"`
	TargetDevices []string `json:"target_devices"`
}

// Result structures

// ValidationResult contains model validation results.
type ValidationResult struct {
	IsValid     bool                   `json:"is_valid"`
	Errors      []ValidationError      `json:"errors"`
	Warnings    []ValidationWarning    `json:"warnings"`
	Metrics     map[string]float64     `json:"metrics"`
	Summary     string                 `json:"summary"`
	Extensions  map[string]interface{} `json:"extensions"`
}

// ValidationError represents a validation error.
type ValidationError struct {
	Type        string `json:"type"`
	Message     string `json:"message"`
	Component   string `json:"component"`
	Severity    string `json:"severity"`
	Suggestion  string `json:"suggestion"`
}

// ValidationWarning represents a validation warning.
type ValidationWarning struct {
	Type        string `json:"type"`
	Message     string `json:"message"`
	Component   string `json:"component"`
	Suggestion  string `json:"suggestion"`
}

// OptimizationStrategy describes an optimization approach.
type OptimizationStrategy struct {
	Name         string                 `json:"name"`
	Description  string                 `json:"description"`
	Category     string                 `json:"category"`     // "performance", "memory", "accuracy"
	Impact       string                 `json:"impact"`       // "low", "medium", "high"
	Requirements []string               `json:"requirements"` // Prerequisites for this optimization
	Options      map[string]interface{} `json:"options"`      // Strategy-specific options
}