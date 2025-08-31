# What We're Not Doing

This document tracks decisions about what features, optimizations, or approaches we're deliberately NOT implementing in Zerfoo, along with the reasoning behind these decisions.

## Numerai-Specific Features

### 🚫 Target-Specific Metrics
- **Decision**: Removed Numerai-specific metrics (Sharpe, Calmar, drawdown analysis)
- **Rationale**: Maintain framework generality; target-specific metrics belong in application layer
- **Alternative**: Generic ML metrics with extensible interface for domain-specific additions
- **Commit**: c69620c, 1c3110e

### 🚫 Era-Specific Data Loaders  
- **Decision**: Keep generic time-series data loading capabilities
- **Rationale**: Era concepts are Numerai-specific; framework should handle generic temporal data
- **Alternative**: Configurable data loading with user-defined temporal grouping

## Performance Optimizations

### 🚫 Custom BLAS Implementations
- **Decision**: Use standard Go numeric libraries and existing BLAS bindings
- **Rationale**: Premature optimization; focus on correctness first
- **Timeline**: Revisit in Epic E4 (Performance) if benchmarks show clear need
- **Alternative**: Profile-guided optimization of critical paths

### 🚫 GPU Kernel Optimization
- **Decision**: No hand-tuned CUDA kernels in initial implementation
- **Rationale**: Maintenance burden outweighs benefits for prototype phase
- **Timeline**: Consider in Epic E6 (Production) for specific bottlenecks
- **Alternative**: Use mature GPU libraries (cuBLAS, cuDNN) through bindings

## Framework Architecture

### 🚫 Dynamic Graph Construction
- **Decision**: Static graph compilation approach only
- **Rationale**: Simplifies optimization and maintains performance predictability
- **Alternative**: User-defined static graph builders with compile-time validation

### 🚫 Automatic Differentiation DSL
- **Decision**: No domain-specific language for gradient computation
- **Rationale**: Go's type system provides sufficient expression without complexity
- **Alternative**: Interface-based gradient computation with manual implementation

### 🚫 Distributed Training (Initially)
- **Decision**: Single-node training focus for Epic E1-E3
- **Rationale**: Multi-node complexity deferred until single-node excellence achieved
- **Timeline**: Epic E5 (Scale) for distributed implementations
- **Alternative**: Design interfaces for future distributed extension

## Testing Strategy

### 🚫 Generative Property-Based Testing
- **Decision**: Focus on deterministic, reproducible test cases
- **Rationale**: ML model testing requires controlled inputs for meaningful validation
- **Alternative**: Comprehensive edge case enumeration with known expected outputs

### 🚫 Mock Implementations for Parity Tests
- **Decision**: Tests skip when implementations not wired rather than using mocks
- **Rationale**: Parity tests must validate real implementations, not mock behavior
- **Alternative**: Interface-based design allowing easy implementation swapping

## Security and Deployment

### 🚫 Model Encryption/Obfuscation
- **Decision**: Plain model storage and loading
- **Rationale**: Academic/research focus; deployment security is application concern
- **Timeline**: Consider in Epic E6 (Production) for commercial deployments

### 🚫 Remote Model Loading
- **Decision**: Local file system model loading only
- **Rationale**: Security and reliability; avoid network dependencies in core framework
- **Alternative**: User-provided model download/caching layer above Zerfoo

## Data Processing

### 🚫 Built-in Data Augmentation
- **Decision**: No framework-level data augmentation primitives
- **Rationale**: Domain-specific; belongs in preprocessing pipeline, not core framework
- **Alternative**: Clean interfaces for user-provided data transformation

### 🚫 Automatic Feature Engineering
- **Decision**: Manual feature definition and engineering
- **Rationale**: Explicit control over feature construction for reproducibility
- **Alternative**: Feature engineering utilities as separate package

## Model Architecture

### 🚫 Architecture Search (NAS)
- **Decision**: Manual architecture definition only
- **Rationale**: Scope limitation; NAS is research project unto itself
- **Timeline**: Potential Epic E8+ for research extensions

### 🚫 Dynamic Model Sizing
- **Decision**: Fixed model architectures defined at compile time
- **Rationale**: Simplifies memory management and performance optimization
- **Alternative**: Model size as explicit constructor parameter

## Monitoring and Observability

### 🚫 Built-in Experiment Tracking
- **Decision**: No integration with MLflow, Weights & Biases, etc.
- **Rationale**: Framework independence; users can integrate with preferred tools
- **Alternative**: Event/metric interfaces for external tool integration

### 🚫 Real-time Training Dashboards
- **Decision**: Artifact-based reporting (CSV/JSON) only
- **Rationale**: Reduces dependencies; separation of concerns
- **Alternative**: Static dashboard generation from artifacts

## Development Tools

### 🚫 Visual Model Graph Editor
- **Decision**: Code-based model definition only
- **Rationale**: Version control, reproducibility, and simplicity
- **Alternative**: Code generation tools for complex architectures

### 🚫 Interactive Jupyter Integration
- **Decision**: CLI-first approach
- **Rationale**: Reproducible scripts over interactive exploration
- **Timeline**: Consider notebook support in Epic E7 (Usability)

---

## Decision Review Process

These decisions are reviewed at each epic completion:
- **Epic E1**: Focus on testing and parity framework ✅
- **Epic E2**: Evaluate data pipeline needs
- **Epic E3**: Reassess model architecture constraints  
- **Epic E4**: Performance optimization decision points
- **Epic E5**: Distributed computing evaluation
- **Epic E6**: Production deployment requirements
- **Epic E7**: Usability and developer experience

## Reversal Criteria

Decisions may be reconsidered if:
1. Performance benchmarks show critical bottlenecks
2. User feedback indicates significant usability impact
3. Technical dependencies change substantially
4. Production requirements demand specific features

---

*Last updated: Epic E1 completion*  
*Next review: Epic E2 start*