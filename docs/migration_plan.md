# Detailed Migration Plan - Zerfoo Architectural Refactoring

**Date:** 2025-08-31  
**Task:** T8.1 - Architectural Analysis and Planning  
**Subtask:** S8.1.3 - Create detailed migration plan for each identified component with impact analysis

## Executive Summary

This detailed migration plan provides step-by-step instructions for executing the architectural refactoring to separate domain-specific Numerai logic from the generic Zerfoo framework. The plan prioritizes maintaining system stability while achieving complete architectural separation.

## Migration Components Overview

| Component | Current Location | Target Location | Complexity | Effort | Dependencies |
|-----------|-----------------|-----------------|------------|--------|--------------|
| Numerai Package (14 files) | `zerfoo/numerai/` | `audacity/internal/numerai/` | High | 2d | None |
| Era Sequencer | `zerfoo/training/era_sequencer*` | `audacity/internal/training/` | Medium | 1d | T8.2 |
| CLI Tools | `zerfoo/cmd/zerfoo-*` | Audacity integration | High | 1.5d | T8.2, T8.5 |
| Era Data Structures | `zerfoo/data/dataset.go` | Generic interfaces | High | 2d | T8.2, T8.5 |
| Integration Tests | `zerfoo/integration/` | `audacity/integration/` | Low | 0.5d | T8.2 |

## Detailed Migration Steps

### Component 1: Numerai Package Migration (14 files)

#### Files to Migrate:
```
zerfoo/numerai/baseline_model.go           → audacity/internal/numerai/baseline_model.go
zerfoo/numerai/baseline_model_test.go      → audacity/internal/numerai/baseline_model_test.go  
zerfoo/numerai/config_lock.go              → audacity/internal/numerai/config_lock.go
zerfoo/numerai/config_lock_test.go         → audacity/internal/numerai/config_lock_test.go
zerfoo/numerai/cross_validation.go         → audacity/internal/numerai/cross_validation.go
zerfoo/numerai/cross_validation_test.go    → audacity/internal/numerai/cross_validation_test.go
zerfoo/numerai/data_contracts.go           → audacity/internal/numerai/data_contracts.go
zerfoo/numerai/data_contracts_test.go      → audacity/internal/numerai/data_contracts_test.go
zerfoo/numerai/prediction_shaping.go       → audacity/internal/numerai/prediction_shaping.go
zerfoo/numerai/prediction_shaping_test.go  → audacity/internal/numerai/prediction_shaping_test.go
zerfoo/numerai/risk_module.go              → audacity/internal/numerai/risk_module.go
zerfoo/numerai/risk_module_test.go         → audacity/internal/numerai/risk_module_test.go
zerfoo/numerai/variance_control.go         → audacity/internal/numerai/variance_control.go
zerfoo/numerai/variance_control_test.go    → audacity/internal/numerai/variance_control_test.go
```

#### Migration Steps:
1. **Create directory structure** in audacity
   ```bash
   mkdir -p /path/to/audacity/internal/numerai
   ```

2. **Copy files** with updated package declarations
   ```bash
   # For each file, update: package numerai → package numerai
   # (package name stays the same but internal to audacity)
   ```

3. **Update import paths** in migrated files
   ```go
   // Old: "github.com/zerfoo/zerfoo/tensor"
   // New: "github.com/zerfoo/zerfoo/tensor" (external dependency remains)
   ```

4. **Update audacity dependencies**
   ```bash
   cd audacity/
   go mod edit -require github.com/zerfoo/zerfoo@latest
   go mod tidy
   ```

5. **Run tests** to verify migration
   ```bash
   go test ./internal/numerai/...
   ```

#### Impact Analysis:
- **Risk:** Medium - Many internal dependencies to resolve
- **Breaking Changes:** None for external users (internal package)
- **Dependencies:** Must keep zerfoo imports working in audacity

### Component 2: Era Sequencer Migration

#### Files to Migrate:
```
zerfoo/training/era_sequencer.go      → audacity/internal/training/era_sequencer.go
zerfoo/training/era_sequencer_test.go → audacity/internal/training/era_sequencer_test.go
```

#### Migration Steps:
1. **Create training directory** in audacity
   ```bash
   mkdir -p /path/to/audacity/internal/training
   ```

2. **Move files** with package updates
   ```go
   // Old: package training
   // New: package training (internal to audacity)
   ```

3. **Update import paths**
   ```go
   // Old: "github.com/zerfoo/zerfoo/data"  
   // New: Use generic interfaces when available (S8.5.2)
   ```

4. **Remove from zerfoo** after verification
   ```bash
   git rm zerfoo/training/era_sequencer.go zerfoo/training/era_sequencer_test.go
   ```

#### Impact Analysis:
- **Risk:** Low - Self-contained functionality
- **Breaking Changes:** Only affects era-specific training workflows
- **Dependencies:** May need generic sequencing interface in zerfoo

### Component 3: CLI Tools Refactoring

#### Files to Modify:
```
zerfoo/cmd/zerfoo-train/main.go       → Refactor or move to audacity
zerfoo/cmd/zerfoo-predict/main.go     → Refactor or move to audacity
```

#### Option A: Move CLI to Audacity (Recommended)
1. **Create CLI directory** in audacity
   ```bash
   mkdir -p /path/to/audacity/cmd/
   ```

2. **Move and rename** tools
   ```bash
   mv zerfoo/cmd/zerfoo-train audacity/cmd/numerai-train
   mv zerfoo/cmd/zerfoo-predict audacity/cmd/numerai-predict
   ```

3. **Update import paths** to use internal packages
   ```go
   // Old: "github.com/zerfoo/zerfoo/numerai"
   // New: "github.com/feza-ai/audacity/internal/numerai"
   ```

#### Option B: Generic CLI Interface (Complex)
1. Create generic training interface in zerfoo
2. Implement interface in audacity  
3. CLI uses interface, audacity provides implementation

#### Impact Analysis:
- **Risk:** High - User-facing changes
- **Breaking Changes:** CLI command names change
- **Dependencies:** Requires completed numerai package migration

### Component 4: Era Data Structures Refactoring

#### Files to Modify:
```
zerfoo/data/dataset.go → Extract generic interfaces
```

#### Migration Steps:
1. **Design generic interfaces**
   ```go
   // New file: zerfoo/data/interfaces.go
   type DataLoader interface {
       LoadData(path string) (Dataset, error)
   }
   
   type Dataset interface {
       Groups() []DataGroup
       Size() int
   }
   
   type DataGroup interface {
       ID() interface{}
       Data() []DataPoint
   }
   ```

2. **Create era-specific implementation** in audacity
   ```go
   // New file: audacity/internal/data/era_dataset.go
   type EraDataset struct {
       Eras []EraData
   }
   
   func (d *EraDataset) Groups() []zerfoo.DataGroup {
       // Convert eras to generic groups
   }
   ```

3. **Update existing code** to use interfaces
   ```go
   // Old: func Process(dataset *data.Dataset)
   // New: func Process(dataset data.Dataset)
   ```

#### Impact Analysis:
- **Risk:** High - Core data structure changes
- **Breaking Changes:** API changes for data handling
- **Dependencies:** Requires careful interface design

### Component 5: Integration Tests Migration

#### Files to Migrate:
```
zerfoo/integration/config_lock_integration_test.go → audacity/integration/
```

#### Migration Steps:
1. **Create integration test directory** in audacity
2. **Move test files** with updated imports
3. **Remove from zerfoo** integration suite

#### Impact Analysis:
- **Risk:** Low - Test-only changes
- **Breaking Changes:** None
- **Dependencies:** Requires migrated numerai package

## Implementation Timeline

### Week 1: Foundation (T8.1 - T8.2)
- **Day 1:** Complete T8.1 (Analysis and API design)
- **Day 2-3:** Execute T8.2 (Numerai package migration)

### Week 2: Core Migration (T8.3 - T8.4)  
- **Day 1:** Execute T8.3 (Era sequencer migration)
- **Day 2:** Execute T8.4 (Configuration migration)
- **Day 3:** Begin T8.5 (Generic API design)

### Week 3: API Design and Integration (T8.5 - T8.6)
- **Day 1-2:** Complete T8.5 (Generic API implementation)
- **Day 3-4:** Execute T8.6 (Audacity integration layer)

### Week 4: Validation and Cleanup (T8.7 - T8.10)
- **Day 1:** T8.7 (Dependency cleanup and validation)  
- **Day 2:** T8.8 (Documentation and API stability)
- **Day 3:** T8.9 (CI/CD pipeline updates)
- **Day 4:** T8.10 (Final quality assurance)

## Risk Mitigation Strategies

### High Risk: Data Structure Changes
**Mitigation:**
- Design interfaces first, implement gradually
- Maintain backward compatibility during transition
- Extensive testing at each step

### High Risk: CLI Changes
**Mitigation:**
- Document migration path for users
- Consider maintaining deprecated commands temporarily
- Provide clear upgrade instructions

### Medium Risk: Import Path Changes
**Mitigation:**
- Use go.mod replace directives during transition
- Test both projects after each change
- Maintain comprehensive test suite

### Medium Risk: Performance Impact
**Mitigation:**
- Benchmark critical paths before and after
- Profile memory usage with new interfaces
- Optimize hot paths if regression occurs

## Quality Gates and Validation

### After Each Component Migration:
1. **Build Success:** Both projects build without errors
2. **Test Success:** All tests pass in both projects
3. **Import Validation:** No prohibited imports exist
4. **Performance Check:** No significant performance regression

### Final Validation Criteria:
1. **Architecture Purity:** 
   ```bash
   grep -r "numerai\|era\|tournament" zerfoo/ --include="*.go" | grep -v test | wc -l
   # Should return 0
   ```

2. **Dependency Direction:**
   ```bash
   # Audacity can import zerfoo, but not vice versa
   grep -r "github.com/feza-ai/audacity" zerfoo/ --include="*.go" | wc -l
   # Should return 0
   ```

3. **Independent Builds:**
   ```bash
   cd zerfoo/ && go build ./...    # Should succeed
   cd audacity/ && go build ./...  # Should succeed
   ```

4. **Test Coverage:** Maintain >90% coverage in both projects

## Rollback Plan

### If Migration Fails:
1. **Git Reset:** Each component has individual commits for easy rollback
2. **Branch Strategy:** Work on feature branch, merge only when complete
3. **Backup:** Keep original code in separate branch until validation complete

### Emergency Rollback Steps:
```bash
# Rollback specific component
git revert <commit-hash>

# Rollback entire migration
git checkout main
git reset --hard <pre-migration-commit>
```

## Success Criteria

### Technical Criteria:
- [ ] Zero domain-specific references in zerfoo codebase
- [ ] All tests passing in both projects
- [ ] Both projects build independently
- [ ] Performance within 5% of baseline
- [ ] API documentation complete

### Architectural Criteria:
- [ ] Clean dependency direction (audacity → zerfoo only)
- [ ] Generic interfaces support multiple domains
- [ ] Zerfoo can be used for non-Numerai ML projects
- [ ] Clear separation of concerns maintained

---
**Migration Plan Status:** Complete  
**Total Effort Estimate:** 16-20 days  
**Risk Level:** High (but mitigatable)  
**Next Action:** Proceed to S8.1.4 - API Design