# Zerfoo Architectural Analysis Report

**Date:** 2025-08-31  
**Task:** T8.1 - Architectural Analysis and Planning  
**Subtask:** S8.1.1 - Scan zerfoo codebase for domain-specific references

## Executive Summary

This report analyzes the current architectural violations in the Zerfoo codebase where domain-specific Numerai tournament concepts have leaked into what should be a generic ML framework. The analysis identifies specific files, code patterns, and dependencies that violate the architectural boundaries.

## Domain-Specific Violations Found

### 1. Entire `numerai/` Package (14 Go files)
**Location:** `/zerfoo/numerai/`  
**Violation:** Complete Numerai-specific package within generic framework  
**Files:**
- `baseline_model.go` & `baseline_model_test.go`
- `config_lock.go` & `config_lock_test.go`
- `cross_validation.go` & `cross_validation_test.go`
- `data_contracts.go` & `data_contracts_test.go`
- `prediction_shaping.go` & `prediction_shaping_test.go`
- `risk_module.go` & `risk_module_test.go`
- `variance_control.go` & `variance_control_test.go`

**Impact:** Entire package is domain-specific and should be moved to `audacity`

### 2. Era-Specific Training Code
**Location:** `/zerfoo/training/`  
**Files:**
- `era_sequencer.go` - Contains EraSequencer struct and era-specific logic
- `era_sequencer_test.go` - Tests for era sequencing

**Violation:** Generic training package contains domain-specific era concepts

### 3. Command Line Interface Coupling
**Location:** `/zerfoo/cmd/`  
**Files:**
- `zerfoo-train/main.go` - Line 13 imports "github.com/zerfoo/zerfoo/numerai"
- `zerfoo-predict/main.go` - Likely similar coupling

**Violation:** CLI tools directly import and use domain-specific packages

### 4. Data Package Domain Coupling
**Location:** `/zerfoo/data/dataset.go`  
**Violations:**
- Lines 10, 17, 21: Era-specific structs and concepts
- Lines 29-31: `NumeraiRow` struct with era field
- Lines 39-72: Era-based data processing logic

**Violation:** Core data package contains Numerai-specific data structures

### 5. Integration Test Coupling
**Location:** `/zerfoo/integration/`
**Files:**
- `config_lock_integration_test.go` - Tests numerai config lock functionality

**Violation:** Framework integration tests depend on domain-specific concepts

## Architectural Boundary Analysis

### Current State Problems
1. **Circular Conceptual Dependencies:** Generic framework knows about domain concepts
2. **Tight Coupling:** Multiple core packages directly reference Numerai concepts
3. **Leaky Abstractions:** Era and tournament concepts spread throughout framework
4. **API Pollution:** Generic interfaces contaminated with domain-specific requirements

### Expected Clean Architecture
```
┌─────────────────────────────────────┐
│            Audacity                 │
│         (Domain Application)        │
│                                     │
│  ┌─────────────────────────────────┐│
│  │         Numerai Package         ││
│  │    - Data Contracts             ││
│  │    - Cross Validation           ││
│  │    - Risk Management            ││
│  │    - Prediction Shaping         ││
│  └─────────────────────────────────┘│
└─────────────────────────────────────┘
                  │
                  │ (depends on)
                  ▼
┌─────────────────────────────────────┐
│            Zerfoo                   │
│        (Generic Framework)          │
│                                     │
│  ┌─────────────────────────────────┐│
│  │       Generic Interfaces        ││
│  │    - Data Loading               ││
│  │    - Model Training             ││
│  │    - Evaluation                 ││
│  │    - Cross Validation           ││
│  └─────────────────────────────────┘│
└─────────────────────────────────────┘
```

## Migration Complexity Assessment

### High Complexity (2-3 days)
- **Data Package Refactoring:** Extract generic interfaces from era-specific implementations
- **Numerai Package Migration:** Move entire package with dependency resolution
- **API Design:** Create clean interfaces that audacity can implement

### Medium Complexity (1-2 days)
- **Training Package Cleanup:** Remove era sequencer, design generic sequencing interfaces
- **CLI Refactoring:** Decouple command-line tools from domain logic
- **Integration Test Migration:** Move domain-specific tests to audacity

### Low Complexity (< 1 day)
- **Import Path Updates:** Update all import statements after migration
- **Documentation Updates:** Reflect new architecture in docs
- **Build Configuration:** Ensure both projects build independently

## Proposed Migration Strategy

### Phase 1: Analysis and Interface Design
1. Design generic data loading interfaces
2. Design generic training and evaluation interfaces
3. Create plugin architecture for domain extensions

### Phase 2: Code Migration
1. Create audacity/internal/numerai structure
2. Move numerai package files
3. Move era sequencer to audacity
4. Extract and move domain-specific data structures

### Phase 3: Interface Implementation
1. Implement generic interfaces in zerfoo
2. Create audacity adapters for zerfoo interfaces
3. Update CLIs to use audacity as integration layer

### Phase 4: Cleanup and Validation
1. Remove domain references from zerfoo
2. Update all import paths
3. Validate both projects build and test independently
4. Document new architecture

## Risk Assessment

### High Risk
- **Breaking Changes:** Existing code using zerfoo directly may break
- **Complex Dependencies:** Deep coupling may require significant refactoring

### Medium Risk
- **Interface Design:** Generic interfaces must be flexible enough for future domains
- **Performance Impact:** New abstraction layers may impact performance

### Low Risk
- **Testing:** Existing tests provide safety net for refactoring
- **Documentation:** Clear plan reduces risk of architectural drift

## Recommendations

1. **Start with T8.1:** Complete detailed analysis before beginning migration
2. **TDD Approach:** Write tests for new interfaces before implementing
3. **Small Commits:** Break migration into small, atomic changes
4. **Parallel Development:** Keep both projects building during migration
5. **Documentation First:** Update architecture docs before code changes

## Next Steps

1. Complete S8.1.2: Identify cross-package dependencies
2. Complete S8.1.3: Create detailed migration plan
3. Complete S8.1.4: Design clean public APIs
4. Complete S8.1.5: Document target architecture
5. Begin T8.2: Execute numerai package migration

---
**Report Status:** Complete  
**Next Action:** Proceed to S8.1.2 - Cross-package dependency analysis