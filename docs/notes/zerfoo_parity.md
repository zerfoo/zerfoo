# Zerfoo Parity Testing Framework

## Overview

This document describes the comprehensive parity testing framework implemented for Zerfoo, ensuring numerical consistency, deterministic behavior, and cross-device reproducibility.

## Test Categories

### 1. Tokenizer Parity (`tests/parity/tokenizer_test.go`)
- **Purpose**: Validate tokenization consistency across reference and implementation
- **Coverage**: Token sequence comparison, round-trip validation, edge case handling
- **Status**: ✅ Implemented with TDD approach

### 2. Logit Parity (`tests/parity/logit_parity_test.go`)
- **Purpose**: Ensure model outputs match reference implementation within tolerance
- **Coverage**: Logits comparison, top-k agreement metrics, CSV/JSON artifact generation
- **Tolerance**: Configurable (default: 5e-4 for float32)
- **Status**: ✅ Implemented with comprehensive reporting

### 3. Decoding Parity (`tests/parity/decoding_test.go`) 
- **Purpose**: Validate text generation consistency and deterministic sampling
- **Coverage**: Greedy decoding, top-p sampling, deterministic flag enforcement
- **Key Requirements**: 
  - S1.4.1: Deterministic decoding across runs
  - S1.4.2: Forced deterministic sampling with seed control
- **Status**: ✅ Implemented with edge case validation

### 4. Reproducibility (`tests/parity/repro_test.go`)
- **Purpose**: Ensure cross-device and cross-session reproducibility
- **Coverage**: CPU vs GPU consistency, session reproducibility, performance validation
- **Status**: ✅ Implemented with performance monitoring

## Numerics Red Team (`tests/numerics/`)

### 1. Finite Difference Checks (`finite_diff_test.go`)
- **Purpose**: Validate gradient computation accuracy using numerical differentiation
- **Method**: Central difference approximation with configurable epsilon
- **Coverage**: Various input ranges, edge cases, relative error validation
- **Tolerance**: Configurable per test case (typically 1e-3 to 1e-6)

### 2. NaN/Inf Detection (`nan_inf_hooks_test.go`)
- **Purpose**: Detect and handle invalid floating-point values in training
- **Coverage**: NaN/Inf detection, tensor dumping for debugging, training halt simulation
- **Features**: Comprehensive tensor analysis, invalid value propagation tracking

### 3. Mixed Precision (`mixed_precision_test.go`)
- **Purpose**: Validate numerical stability across different precision levels
- **Coverage**: Float64/Float32/Float16 ranges, precision degradation, consistency checks
- **Key Tests**: Dynamic range validation, subnormal detection, cross-session consistency

## Implementation Architecture

### Test Interface Design
All parity tests use the `helpers.ZerfooAPI` interface pattern:
```go
type ZerfooAPI interface {
    SetSeed(seed int)
    Tokenize(text string) ([]int, error)
    DecodeGreedy(ids []int) (string, error)
    DecodeTopP(ids []int, p float64) (string, error)
    Logits(prompt string, maxNewTokens int) ([]float32, error)
    DeviceName() string
}
```

### Graceful Degradation
- All tests check `helpers.ImplZerfoo == nil` and skip gracefully
- Ready for immediate activation when implementations are wired
- Non-blocking CI execution until interfaces are satisfied

## CI Integration

### Workflow Structure (`.github/workflows/ci.yml`)
1. **Unit Tests**: Core functionality excluding parity tests
2. **Linting**: Full golangci-lint validation  
3. **Parity Tests**: Non-blocking execution of cross-device consistency checks
4. **Numerics Red Team**: Non-blocking numerical validation suite
5. **Nightly Training**: Toy model training with artifact generation

### Quality Gates
- All tests must pass compilation
- Linting must be clean (gosec, gocritic, staticcheck)
- Performance benchmarks within reasonable bounds
- Deterministic behavior verified across multiple runs

## Key Achievements

### ✅ Completed Tasks (Epic E1)
- **T1.4**: Decode Parity - Comprehensive text generation validation
- **T1.5**: Numerics Red Team - Gradient validation and stability testing  
- **T1.6**: Determinism & CI - Cross-device reproducibility framework

### 🎯 Testing Philosophy
- **Test-Driven Development**: Tests written before implementation
- **Fail-Safe Design**: Tests skip gracefully when dependencies unavailable
- **Comprehensive Coverage**: Edge cases, error conditions, performance characteristics
- **Deterministic Validation**: Seed-based reproducibility enforcement

## Metrics and Reporting

### Artifact Generation
- **CSV Reports**: Detailed parity metrics with timestamps
- **JSON Logs**: Machine-readable test results for dashboards
- **Performance Data**: Latency measurements across batch sizes
- **Tensor Dumps**: Debug information for invalid value detection

### Success Criteria
- Tokenizer parity: 100% token sequence match
- Logit parity: <5e-4 relative error, >99% top-k agreement
- Decode parity: Identical outputs with same seed
- Numerics: <1e-3 gradient error, no NaN/Inf propagation

## Next Steps

### Epic E2 Prerequisites
Before advancing to Numerai pipeline development, ensure:
1. All parity interfaces are wired with real implementations
2. Cross-device testing validates on actual GPU hardware
3. Performance benchmarks meet production requirements
4. Training stability verified through numerics red team

### Implementation Wiring
To activate tests, implement and wire in `tests/helpers/wire.go`:
- `ImplZerfoo`: Connect to actual tokenizer/model implementation
- `ImplNumerics`: Connect to gradient computation backend
- `ImplPerf`: Connect to performance measurement system

## Technical Debt and Considerations

### Current Limitations
- Tests currently skip due to unwired implementations
- GPU testing requires actual hardware for validation
- Performance baselines need establishment with real implementations

### Security Considerations
- File creation uses secure permissions (0600)
- No sensitive data logged in artifacts
- Gosec warnings addressed with appropriate suppressions

This parity framework provides a robust foundation for ensuring Zerfoo's numerical accuracy and reproducibility across deployment environments.