# Target Architecture - Zerfoo Generic Framework & Audacity Application

**Date:** 2025-08-31  
**Task:** T8.1 - Architectural Analysis and Planning  
**Subtask:** S8.1.5 - Document the target architecture showing clear separation of concerns

## Architectural Vision

The target architecture transforms the current monolithic structure into a clean layered architecture following the **Hexagonal Architecture** pattern, where:

- **Zerfoo** becomes a truly generic, reusable ML framework (the "hexagon core")
- **Audacity** becomes a domain-specific application that adapts Zerfoo for Numerai (the "adapter")
- Clear dependency direction: Audacity → Zerfoo (never the reverse)

## High-Level Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        Audacity Application                     │
│                     (Numerai Domain Logic)                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │  CLI Commands   │  │  Internal APIs  │  │  Integration    │ │
│  │                 │  │                 │  │  Tests          │ │
│  │ • numerai-train │  │ • REST API      │  │                 │ │
│  │ • numerai-pred  │  │ • Config Mgmt   │  │ • End-to-End    │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │                Audacity Internal Packages                   │ │
│  │                                                             │ │
│  │ ┌───────────────┐ ┌───────────────┐ ┌───────────────────┐  │ │
│  │ │ /numerai/     │ │ /training/    │ │ /config/          │  │ │
│  │ │               │ │               │ │                   │  │ │
│  │ │ • Baseline    │ │ • EraSeq      │ │ • Domain Config   │  │ │
│  │ │ • CrossVal    │ │ • Time Series │ │ • Validation      │  │ │
│  │ │ • Risk Mgmt   │ │ • Curriculum  │ │ • Lock Mechanism  │  │ │
│  │ │ • DataContr   │ │               │ │                   │  │ │
│  │ │ • PredShape   │ │               │ │                   │  │ │
│  │ │ • VarControl  │ │               │ │                   │  │ │
│  │ └───────────────┘ └───────────────┘ └───────────────────┘  │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │                   Adapter Layer                             │ │
│  │            (Implements Zerfoo Interfaces)                   │ │
│  │                                                             │ │
│  │ ┌───────────────┐ ┌───────────────┐ ┌───────────────────┐  │ │
│  │ │NumeraiDataset │ │ EraSequencer  │ │ NumeraiMetrics    │  │ │
│  │ │   Adapter     │ │   Adapter     │ │    Adapter        │  │ │
│  │ │               │ │               │ │                   │  │ │
│  │ │ implements    │ │ implements    │ │ implements        │  │ │
│  │ │ Dataset       │ │ Sequencer     │ │ Metric            │  │ │
│  │ └───────────────┘ └───────────────┘ └───────────────────┘  │ │
│  └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                                │
                                │ depends on
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                        Zerfoo Framework                         │
│                   (Generic ML Framework)                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │                  Public API Layer                           │ │
│  │                 (Generic Interfaces)                        │ │
│  │                                                             │ │
│  │ ┌───────────────┐ ┌───────────────┐ ┌───────────────────┐  │ │
│  │ │   Dataset     │ │   Sequencer   │ │     Metric        │  │ │
│  │ │   Interface   │ │   Interface   │ │    Interface      │  │ │
│  │ │               │ │               │ │                   │  │ │
│  │ │ • DataGroup   │ │ • Generate    │ │ • Compute         │  │ │
│  │ │ • DataPoint   │ │ • Configure   │ │ • Direction       │  │ │
│  │ │ • Normalize   │ │ • SetSeed     │ │ • Metadata        │  │ │
│  │ └───────────────┘ └───────────────┘ └───────────────────┘  │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │                Framework Implementation                      │ │
│  │                                                             │ │
│  │ ┌───────────────┐ ┌───────────────┐ ┌───────────────────┐  │ │
│  │ │   /data/      │ │  /training/   │ │   /model/         │  │ │
│  │ │               │ │               │ │                   │  │ │
│  │ │ • Interfaces  │ │ • Interfaces  │ │ • Builder         │  │ │
│  │ │ • Generic     │ │ • Generic     │ │ • Ensemble        │  │ │
│  │ │   Loaders     │ │   Trainers    │ │ • Serialization   │  │ │
│  │ │ • Processors  │ │ • Validators  │ │                   │  │ │
│  │ └───────────────┘ └───────────────┘ └───────────────────┘  │ │
│  │                                                             │ │
│  │ ┌───────────────┐ ┌───────────────┐ ┌───────────────────┐  │ │
│  │ │  /compute/    │ │   /tensor/    │ │   /layers/        │  │ │
│  │ │               │ │               │ │                   │  │ │
│  │ │ • CPU Engine  │ │ • Operations  │ │ • Neural Layers   │  │ │
│  │ │ • GPU Engine  │ │ • UINT8 Supp  │ │ • Activations     │  │ │
│  │ │ • Quantization│ │ • Broadcasting│ │ • Normalization   │  │ │
│  │ └───────────────┘ └───────────────┘ └───────────────────┘  │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │                   Plugin Registry                           │ │
│  │                                                             │ │
│  │ • DataLoader Registry    • Model Registry                   │ │
│  │ • Sequencer Registry     • Metric Registry                  │ │
│  │ • Evaluator Registry     • Extension Registry               │ │
│  └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## Package Organization

### Zerfoo Framework Structure
```
zerfoo/
├── README.md
├── go.mod
├── docs/
│   ├── api_reference.md
│   ├── user_guide.md
│   └── plugin_development.md
├── data/
│   ├── interfaces.go          # Generic data interfaces
│   ├── loaders.go            # Generic data loaders
│   └── processors.go         # Generic data processors
├── training/
│   ├── interfaces.go          # Training interfaces
│   ├── trainer.go            # Generic trainer implementation
│   ├── cross_validator.go    # Generic CV implementation
│   └── strategies.go         # Training strategies
├── evaluation/
│   ├── interfaces.go          # Evaluation interfaces
│   ├── metrics.go            # Standard metrics
│   └── evaluator.go          # Generic evaluator
├── model/
│   ├── interfaces.go          # Model interfaces
│   ├── builder.go            # Model builder
│   ├── ensemble.go           # Ensemble implementation
│   └── serialization.go      # Save/load functionality
├── config/
│   ├── training.go           # Training config structures
│   ├── model.go             # Model config structures
│   └── validation.go        # Validation helpers
├── registry/
│   ├── registry.go          # Plugin registry
│   └── extensions.go        # Extension interfaces
├── compute/                  # Unchanged - already generic
├── tensor/                   # Unchanged - already generic
├── layers/                   # Unchanged - already generic
├── numeric/                  # Unchanged - already generic
└── internal/                 # Internal utilities
```

### Audacity Application Structure
```
audacity/
├── README.md
├── go.mod
├── cmd/
│   ├── numerai-train/        # Moved from zerfoo
│   │   └── main.go
│   ├── numerai-predict/      # Moved from zerfoo
│   │   └── main.go
│   └── numerai-serve/        # New API server
│       └── main.go
├── internal/
│   ├── numerai/              # Moved from zerfoo/numerai/
│   │   ├── baseline_model.go
│   │   ├── config_lock.go
│   │   ├── cross_validation.go
│   │   ├── data_contracts.go
│   │   ├── prediction_shaping.go
│   │   ├── risk_module.go
│   │   └── variance_control.go
│   ├── training/             # Era-specific training
│   │   ├── era_sequencer.go  # Moved from zerfoo
│   │   ├── time_series_cv.go # Numerai-specific CV
│   │   └── curriculum.go     # Era curriculum learning
│   ├── data/
│   │   ├── numerai_dataset.go  # Implements zerfoo.Dataset
│   │   ├── era_loader.go      # Era-aware data loading
│   │   └── transformers.go    # Numerai feature transforms
│   ├── config/
│   │   ├── numerai_config.go  # Domain config
│   │   ├── validation.go     # Config validation
│   │   └── lock.go           # Config locking
│   ├── adapters/             # Zerfoo interface implementations
│   │   ├── dataset_adapter.go
│   │   ├── sequencer_adapter.go
│   │   ├── metric_adapter.go
│   │   └── evaluator_adapter.go
│   └── api/                  # REST API
│       ├── handlers.go
│       ├── middleware.go
│       └── routes.go
├── integration/              # Moved from zerfoo
│   └── numerai_integration_test.go
└── docs/
    ├── numerai_guide.md
    └── deployment.md
```

## Data Flow Architecture

### Training Flow
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Numerai Data   │───▶│ Audacity Loader │───▶│   Zerfoo        │
│   (Parquet)     │    │  (Era-aware)    │    │  Dataset        │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                      │
                                                      ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Trained Model   │◀───│ Zerfoo Trainer  │◀───│ Era Sequencer   │
│   (Persisted)   │    │   (Generic)     │    │  (Audacity)     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Prediction Flow
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│Tournament Data  │───▶│ Audacity Loader │───▶│ Trained Model   │
│   (Parquet)     │    │  (Era-aware)    │    │  (Zerfoo)       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                      │
                                                      ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Submission    │◀───│ Audacity Post   │◀───│  Predictions    │
│     File        │    │   Processor     │    │   (Raw)         │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Interface Implementation Strategy

### 1. Generic Data Interface
```go
// Zerfoo defines the contract
type Dataset interface {
    Groups() []DataGroup
    Size() int
    NormalizeFeatures() error
}

// Audacity implements for Numerai
type NumeraiDataset struct {
    eras []EraData
}

func (nd *NumeraiDataset) Groups() []zerfoo.DataGroup {
    // Convert eras to generic groups
}
```

### 2. Plugin Registration
```go
// Audacity registers its implementations
func init() {
    zerfoo.RegisterDataLoader("numerai", func() zerfoo.DataLoader {
        return &NumeraiParquetLoader{}
    })
    
    zerfoo.RegisterSequencer("era_sequence", func(config map[string]interface{}) zerfoo.Sequencer {
        return &EraSequencer{}
    })
    
    zerfoo.RegisterMetric("numerai_corr", func() zerfoo.Metric {
        return &NumeraiCorrelationMetric{}
    })
}
```

## Configuration Management

### Zerfoo Generic Configuration
```yaml
# training_config.yaml (Generic)
training:
  epochs: 100
  learning_rate: 0.001
  batch_size: 32
  early_stopping: true

cross_validation:
  strategy: "group"  # Generic strategy
  num_folds: 5
  group_by: "group_id"  # Generic group field

data:
  loader: "numerai"  # Plugin selection
  normalizer: "z_score"
```

### Audacity Domain Configuration
```yaml
# numerai_config.yaml (Domain-specific)
numerai:
  tournament: "numerai_tournament_api"
  target_column: "target_nomi_v4_20"
  
  era_sequencing:
    max_sequence_length: 4
    curriculum_strategy: "progressive"
    
  risk_management:
    max_exposure: 0.1
    neutralization:
      - "country"
      - "sector"
      
  prediction_shaping:
    gaussian_ranking: true
    clip_outliers: true
    outlier_threshold: 3.0

extensions:
  # Domain-specific extensions
  variance_control:
    enabled: true
    target_variance: 0.015
```

## Migration Phases

### Phase 1: Interface Definition (Week 1)
1. Create generic interfaces in zerfoo
2. Implement plugin registry system
3. Define configuration structures
4. Create adapter templates

### Phase 2: Implementation Migration (Week 2)
1. Move numerai package to audacity/internal/
2. Move era sequencer to audacity/internal/training/
3. Create adapter implementations
4. Update import paths

### Phase 3: CLI and Integration (Week 3)
1. Move CLI tools to audacity/cmd/
2. Create generic CLI framework in zerfoo
3. Update integration tests
4. Verify end-to-end functionality

### Phase 4: Cleanup and Validation (Week 4)
1. Remove domain references from zerfoo
2. Validate architectural purity
3. Performance testing and optimization
4. Documentation updates

## Quality Assurance

### Architectural Validation
```bash
# No domain references in zerfoo
grep -r "numerai\|era\|tournament" zerfoo/ --include="*.go" | grep -v test
# Should return empty

# No reverse dependencies
grep -r "github.com/feza-ai/audacity" zerfoo/ --include="*.go" 
# Should return empty

# Both projects build independently
cd zerfoo/ && go build ./...
cd audacity/ && go build ./...
```

### Interface Compliance Testing
```go
// Test that implementations satisfy interfaces
func TestInterfaceCompliance(t *testing.T) {
    var _ zerfoo.Dataset = (*NumeraiDataset)(nil)
    var _ zerfoo.Sequencer = (*EraSequencer)(nil)
    var _ zerfoo.Metric = (*NumeraiCorrelationMetric)(nil)
}
```

## Benefits of Target Architecture

### 1. True Framework Genericity
- Zerfoo can be used for any ML domain (not just finance)
- No domain knowledge embedded in framework
- Clean, documented APIs for extension

### 2. Domain Expertise Encapsulation
- All Numerai knowledge contained in Audacity
- Domain-specific optimizations possible
- Business logic separated from framework logic

### 3. Independent Development
- Teams can work on zerfoo and audacity separately
- Different release cycles possible
- Clear ownership boundaries

### 4. Extensibility
- New domains can create their own applications using zerfoo
- Plugin architecture supports third-party extensions
- Framework evolution doesn't break domain applications

### 5. Testing and Quality
- Framework tested independently of domain logic
- Domain logic tested with mock framework components
- Clear integration testing boundaries

---
**Architecture Status:** Fully Defined  
**Separation Level:** Complete (0 domain coupling in framework)  
**Migration Complexity:** High but systematic  
**Next Action:** Begin T8.2 - Numerai Package Migration