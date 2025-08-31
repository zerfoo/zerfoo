# Cross-Package Dependency Analysis

**Date:** 2025-08-31  
**Task:** T8.1 - Architectural Analysis and Planning  
**Subtask:** S8.1.2 - Identify cross-package dependencies that violate architectural boundaries

## Summary

This analysis identifies specific import relationships that violate the intended architecture where `zerfoo` should be a generic framework without domain-specific knowledge.

## Problematic Dependencies

### 1. Direct Numerai Package Imports

**Violation Pattern:** Non-domain packages importing domain-specific code

#### CMD Package → Numerai Package
- **File:** `cmd/zerfoo-train/main.go:13`
- **Import:** `"github.com/zerfoo/zerfoo/numerai"`
- **Impact:** CLI tools directly coupled to domain logic
- **Migration:** Move CLI to audacity or create generic CLI interface

#### Integration Tests → Numerai Package  
- **File:** `integration/config_lock_integration_test.go`
- **Import:** `"github.com/zerfoo/zerfoo/numerai"`
- **Impact:** Framework integration tests depend on domain code
- **Migration:** Move integration tests to audacity

### 2. Numerai Package Dependencies (17 Internal Imports)

**Violation Pattern:** Domain package depending on framework internals (acceptable direction)

#### Heavy Framework Dependencies:
- `tensor` package (6 imports across files)
- `compute` package (baseline_model.go)
- `graph` package (baseline_model.go)  
- `layers/core` package (baseline_model.go)
- `numeric` package (baseline_model.go)
- `training` package (baseline_model.go)

**Analysis:** These dependencies are architecturally acceptable (domain → framework) but indicate tight coupling that may require interface design during migration.

### 3. Era-Specific Data Dependencies

**Violation Pattern:** Generic packages depending on era-contaminated data structures

#### Training Package → Data Package
- **Files:** 
  - `training/era_sequencer.go` → `"github.com/zerfoo/zerfoo/data"`
  - `training/era_sequencer_test.go` → `"github.com/zerfoo/zerfoo/data"`
- **Impact:** Generic training code depends on domain-contaminated data structures
- **Migration:** Extract generic interfaces from data package

#### Features Package → Data Package
- **Files:**
  - `features/transformers.go` → `"github.com/zerfoo/zerfoo/data"`
  - `features/transformers_test.go` → `"github.com/zerfoo/zerfoo/data"`
- **Impact:** Feature extraction depends on era-specific data structures
- **Migration:** Create generic data interfaces

## Dependency Graph Analysis

### Current Problematic Flow:
```
┌─────────────────┐    ┌─────────────────┐
│   CMD Package   │───▶│ Numerai Package │
│  (Generic CLI)  │    │ (Domain Logic)  │
└─────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │  Zerfoo Core    │
                       │  (Framework)    │
                       └─────────────────┘

┌─────────────────┐    ┌─────────────────┐
│Training Package │───▶│  Data Package   │
│   (Generic)     │    │(Era-Contaminated)│
└─────────────────┘    └─────────────────┘
```

### Target Clean Flow:
```
┌─────────────────┐    ┌─────────────────┐
│   Audacity      │───▶│  Zerfoo Core    │
│  (Domain App)   │    │  (Framework)    │
└─────────────────┘    └─────────────────┘
        │
        ▼
┌─────────────────┐
│ Numerai Package │
│ (Domain Logic)  │
└─────────────────┘
```

## Migration Impact Assessment

### High Impact Changes Required

#### 1. CLI Refactoring
- **Current:** `cmd/zerfoo-train/main.go` directly imports numerai
- **Required:** Extract generic training interface or move CLI to audacity
- **Effort:** 1-2 days
- **Risk:** High - changes user-facing interface

#### 2. Data Package Interface Extraction  
- **Current:** Era-specific structs mixed with generic functionality
- **Required:** Extract generic interfaces, move era logic to audacity
- **Effort:** 2-3 days
- **Risk:** High - affects multiple consumers

#### 3. Training Package Refactoring
- **Current:** EraSequencer directly uses era-contaminated data
- **Required:** Generic sequencing interface with era-specific implementation in audacity
- **Effort:** 1-2 days  
- **Risk:** Medium - well-contained functionality

### Medium Impact Changes Required

#### 1. Integration Test Migration
- **Current:** Framework tests depend on domain logic
- **Required:** Move to audacity or create mock implementations
- **Effort:** 1 day
- **Risk:** Low - test-only changes

#### 2. Features Package Cleanup
- **Current:** Transformers depend on era-specific data
- **Required:** Generic transformer interfaces
- **Effort:** 1-2 days
- **Risk:** Medium - may affect existing users

## Recommended Migration Order

### Phase 1: Interface Design (S8.1.4)
1. Design generic data loading interfaces
2. Design generic training sequence interfaces  
3. Design generic CLI interfaces or decide on audacity migration

### Phase 2: Implementation Preparation (T8.2)
1. Create audacity internal structure
2. Implement generic interfaces in zerfoo
3. Create audacity-specific implementations

### Phase 3: Migration Execution (T8.3)
1. Move numerai package to audacity
2. Move era sequencer to audacity
3. Refactor CLI tools
4. Update integration tests

### Phase 4: Interface Cleanup (T8.4)
1. Extract generic data interfaces
2. Clean up training package dependencies
3. Verify all violations resolved

## Quality Gates

### Verification Criteria:
- [ ] No packages in zerfoo import numerai package
- [ ] No era-specific types in zerfoo data package
- [ ] No domain references in zerfoo training package
- [ ] CLI tools work through generic interfaces
- [ ] All tests pass in both projects

### Validation Commands:
```bash
# Verify no domain imports in framework
grep -r "numerai\|era" zerfoo/ --include="*.go" | grep -v "test" | wc -l
# Should return 0

# Verify zerfoo builds without numerai package  
rm -rf numerai/
go build ./...
# Should succeed
```

## Next Steps

1. Complete S8.1.3: Create detailed migration plan
2. Complete S8.1.4: Design clean public APIs  
3. Complete S8.1.5: Document target architecture
4. Begin implementation with T8.2

---
**Analysis Status:** Complete  
**Critical Dependencies Identified:** 6 high-impact violations  
**Migration Complexity:** High (2-3 week effort)  
**Next Action:** Proceed to detailed migration planning