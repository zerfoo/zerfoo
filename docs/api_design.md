# Generic API Design for Zerfoo Framework

**Date:** 2025-08-31  
**Task:** T8.1 - Architectural Analysis and Planning  
**Subtask:** S8.1.4 - Design clean public APIs to support domain-agnostic ML workflows

## Design Principles

1. **Accept interfaces, return concrete types** - APIs should accept the most general interface possible but return concrete implementations
2. **Domain independence** - No API should reference domain-specific concepts like "era", "tournament", "numerai"
3. **Extensibility** - APIs should support plugin architectures for domain-specific extensions
4. **Backward compatibility** - New APIs should not break existing functionality during transition
5. **Performance aware** - Generic interfaces should not introduce significant overhead

## Core Interface Hierarchy

### 1. Generic Data Loading Interfaces

```go
// File: zerfoo/data/interfaces.go

// DataPoint represents a single training example
type DataPoint interface {
    ID() string
    Features() []float64
    Target() float64
    Metadata() map[string]interface{}
}

// DataGroup represents a collection of related data points
// (replaces era-specific grouping with generic concept)
type DataGroup interface {
    GroupID() interface{}        // Can be int, string, time, etc.
    DataPoints() []DataPoint
    GroupMetadata() map[string]interface{}
    Size() int
}

// Dataset represents a complete dataset with multiple groups
type Dataset interface {
    Groups() []DataGroup
    Size() int
    NumFeatures() int
    NormalizeFeatures() error
    Metadata() map[string]interface{}
}

// DataLoader handles loading datasets from various sources
type DataLoader interface {
    Load(source string, config map[string]interface{}) (Dataset, error)
    SupportedFormats() []string
}

// DataProcessor handles data transformations
type DataProcessor interface {
    Process(dataset Dataset, config map[string]interface{}) (Dataset, error)
    ProcessInPlace(dataset Dataset, config map[string]interface{}) error
}
```

### 2. Generic Training Interfaces

```go
// File: zerfoo/training/interfaces.go

// Sequencer generates training sequences from datasets
// (replaces EraSequencer with generic concept)
type Sequencer interface {
    GenerateSequences(dataset Dataset, numSequences int, config map[string]interface{}) ([]Dataset, error)
    SetRandomSeed(seed int64)
}

// TrainingStrategy defines how training should be conducted
type TrainingStrategy interface {
    Train(dataset Dataset, model Model, config TrainingConfig) (*TrainingResult, error)
    Validate(dataset Dataset, model Model, config ValidationConfig) (*ValidationResult, error)
}

// CrossValidator handles cross-validation strategies
type CrossValidator interface {
    Split(dataset Dataset, config CVConfig) ([]TrainTestSplit, error)
    Validate(dataset Dataset, model Model, config CVConfig) (*CVResult, error)
}

// TrainTestSplit represents a single train/test split
type TrainTestSplit interface {
    TrainData() Dataset
    TestData() Dataset
    SplitMetadata() map[string]interface{}
}
```

### 3. Generic Evaluation Interfaces

```go
// File: zerfoo/evaluation/interfaces.go

// Metric defines a performance metric
type Metric interface {
    Name() string
    Compute(predictions, targets []float64, metadata map[string]interface{}) (float64, error)
    Direction() OptimizationDirection // Higher or Lower is better
}

// Evaluator computes multiple metrics on predictions
type Evaluator interface {
    Evaluate(predictions, targets []float64, metadata map[string]interface{}) (map[string]float64, error)
    AddMetric(metric Metric)
    RemoveMetric(name string)
}

// OptimizationDirection indicates whether higher or lower values are better
type OptimizationDirection int

const (
    Higher OptimizationDirection = iota
    Lower
)
```

### 4. Generic Model Interfaces

```go
// File: zerfoo/model/interfaces.go

// Model represents a trainable ML model
type Model interface {
    Train(dataset Dataset, config TrainingConfig) error
    Predict(dataset Dataset) ([]float64, error)
    Save(path string) error
    Load(path string) error
    Metadata() map[string]interface{}
}

// ModelBuilder creates models from configuration
type ModelBuilder interface {
    Build(config ModelConfig) (Model, error)
    SupportedTypes() []string
}

// Ensemble represents a collection of models
type Ensemble interface {
    Model // Implements the same interface as individual models
    AddModel(model Model, weight float64)
    RemoveModel(id string)
    Models() map[string]Model
}
```

## Configuration Structures

### 1. Training Configuration
```go
// File: zerfoo/config/training.go

type TrainingConfig struct {
    NumEpochs      int                    `json:"num_epochs"`
    LearningRate   float64                `json:"learning_rate"`
    BatchSize      int                    `json:"batch_size"`
    EarlyStop      bool                   `json:"early_stop"`
    RandomSeed     int64                  `json:"random_seed"`
    Extensions     map[string]interface{} `json:"extensions"` // Domain-specific config
}

type ValidationConfig struct {
    MetricNames    []string               `json:"metric_names"`
    TestSize       float64                `json:"test_size"`
    RandomSeed     int64                  `json:"random_seed"`
    Extensions     map[string]interface{} `json:"extensions"`
}

type CVConfig struct {
    Strategy       string                 `json:"strategy"`        // "k_fold", "time_series", "group"
    NumFolds       int                    `json:"num_folds"`
    GroupBy        string                 `json:"group_by"`        // For group-based CV
    PurgeGap       int                    `json:"purge_gap"`       // For time-series CV
    TestSize       float64                `json:"test_size"`
    RandomSeed     int64                  `json:"random_seed"`
    Extensions     map[string]interface{} `json:"extensions"`
}
```

### 2. Model Configuration
```go
// File: zerfoo/config/model.go

type ModelConfig struct {
    Type           string                 `json:"type"`            // "linear", "mlp", "ensemble"
    Architecture   map[string]interface{} `json:"architecture"`
    Hyperparams    map[string]interface{} `json:"hyperparams"`
    Extensions     map[string]interface{} `json:"extensions"`
}
```

## Plugin Architecture

### 1. Registry Pattern
```go
// File: zerfoo/registry/registry.go

var (
    DataLoaders    = make(map[string]func() DataLoader)
    Sequencers     = make(map[string]func(config map[string]interface{}) Sequencer)
    CrossValidators = make(map[string]func() CrossValidator)
    Models         = make(map[string]func(config ModelConfig) Model)
    Metrics        = make(map[string]func() Metric)
)

// RegisterDataLoader registers a new data loader
func RegisterDataLoader(name string, factory func() DataLoader) {
    DataLoaders[name] = factory
}

// GetDataLoader retrieves a registered data loader
func GetDataLoader(name string) (DataLoader, error) {
    factory, exists := DataLoaders[name]
    if !exists {
        return nil, fmt.Errorf("data loader %s not registered", name)
    }
    return factory(), nil
}

// Similar patterns for other components...
```

### 2. Extension Points
```go
// File: zerfoo/extensions/interfaces.go

// Extension allows domain-specific functionality to be plugged in
type Extension interface {
    Name() string
    Version() string
    Initialize(config map[string]interface{}) error
    Shutdown() error
}

// DataExtension allows custom data processing
type DataExtension interface {
    Extension
    ProcessDataPoint(dp DataPoint) (DataPoint, error)
    ProcessGroup(group DataGroup) (DataGroup, error)
}

// TrainingExtension allows custom training behavior
type TrainingExtension interface {
    Extension
    PreTrain(dataset Dataset, model Model) error
    PostTrain(dataset Dataset, model Model, result *TrainingResult) error
}
```

## Domain-Specific Implementation Example (Audacity)

### Numerai Data Implementation
```go
// File: audacity/internal/numerai/data.go

// NumeraiDataPoint implements the generic DataPoint interface
type NumeraiDataPoint struct {
    id       string
    features []float64
    target   float64
    era      int
}

func (ndp *NumeraiDataPoint) ID() string { return ndp.id }
func (ndp *NumeraiDataPoint) Features() []float64 { return ndp.features }
func (ndp *NumeraiDataPoint) Target() float64 { return ndp.target }
func (ndp *NumeraiDataPoint) Metadata() map[string]interface{} {
    return map[string]interface{}{"era": ndp.era}
}

// NumeraiDataGroup implements the generic DataGroup interface
type NumeraiDataGroup struct {
    era    int
    stocks []zerfoo.DataPoint
}

func (ndg *NumeraiDataGroup) GroupID() interface{} { return ndg.era }
func (ndg *NumeraiDataGroup) DataPoints() []zerfoo.DataPoint { return ndg.stocks }
func (ndg *NumeraiDataGroup) GroupMetadata() map[string]interface{} {
    return map[string]interface{}{"era": ndg.era, "type": "numerai_era"}
}
func (ndg *NumeraiDataGroup) Size() int { return len(ndg.stocks) }
```

### Numerai Sequencer Implementation
```go
// File: audacity/internal/numerai/era_sequencer.go

type NumeraiEraSequencer struct {
    maxSeqLen int
    rand      *rand.Rand
}

func (nes *NumeraiEraSequencer) GenerateSequences(dataset zerfoo.Dataset, numSequences int, config map[string]interface{}) ([]zerfoo.Dataset, error) {
    // Implementation that works with generic interfaces but understands era ordering
    groups := dataset.Groups()
    
    // Sort by era (extracted from metadata)
    sort.Slice(groups, func(i, j int) bool {
        eraI := groups[i].GroupMetadata()["era"].(int)
        eraJ := groups[j].GroupMetadata()["era"].(int)
        return eraI < eraJ
    })
    
    // Generate sequences using generic interfaces
    // ... implementation details
}

func (nes *NumeraiEraSequencer) SetRandomSeed(seed int64) {
    nes.rand = rand.New(rand.NewPCG(uint64(seed), 0))
}

// Register the implementation
func init() {
    zerfoo.RegisterSequencer("numerai_era", func(config map[string]interface{}) zerfoo.Sequencer {
        maxSeqLen := config["max_seq_len"].(int)
        return &NumeraiEraSequencer{maxSeqLen: maxSeqLen}
    })
}
```

## Migration Strategy

### Phase 1: Interface Definition
1. Create interface files in zerfoo
2. Add registry system for plugins
3. Define configuration structures

### Phase 2: Implementation
1. Implement concrete types in zerfoo that satisfy interfaces
2. Create numerai-specific implementations in audacity
3. Update existing code to use interfaces gradually

### Phase 3: Integration
1. Update CLI tools to use generic interfaces
2. Move domain-specific implementations to audacity
3. Remove domain-specific code from zerfoo

## Benefits of This Design

### 1. Domain Independence
- Zerfoo contains no era, tournament, or numerai concepts
- New domains can be added without changing zerfoo core
- Clear separation of concerns

### 2. Extensibility
- Plugin architecture allows custom implementations
- Registry system enables runtime component selection
- Extension points support domain-specific behavior

### 3. Testability
- Interfaces enable easy mocking for tests
- Domain-specific tests isolated in audacity
- Generic functionality tested independently

### 4. Performance
- Interfaces designed to minimize overhead
- Concrete implementations can be optimized per domain
- No unnecessary abstractions in hot paths

## API Compatibility

### Backward Compatibility
- Existing concrete types implement new interfaces
- Gradual migration path with deprecated warnings
- No breaking changes until major version

### Forward Compatibility
- Extension mechanism supports future requirements
- Configuration structure allows new fields
- Plugin system enables third-party additions

---
**API Design Status:** Complete  
**Interface Count:** 15+ core interfaces  
**Plugin Points:** 5 major extension points  
**Next Action:** Proceed to S8.1.5 - Document Target Architecture