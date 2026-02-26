# Zerfoo Test Coverage Improvement Plan

## 1. Context

### Problem Statement

Zerfoo is a Go-based ML framework with 40+ packages. Test coverage varies widely: some packages
have 100% coverage while others have 0%. The goal is to raise every testable package to at least
95% statement coverage. This improves confidence in correctness, prevents regressions, and makes
the codebase safer to refactor.

Current coverage snapshot (2026 02 25):

| Coverage Tier | Packages | Count |
|---------------|----------|-------|
| >= 95% (at target) | layers/attention (95.1%), model (95.4%), layers/hrm (95.5%), training (95.7%), layers/reducesum (95.9%), layers/core (96.0%), compute (96.2%), training/loss (96.3%), layers/transformer (96.4%), cmd/cli (96.5%), distributed (96.5%), layers/normalization (96.6%), layers/recurrent (96.7%), graph (97.0%), layers/activations (97.1%), layers/transpose (97.2%), training/optimizer (97.4%), model/hrm (98.1%), numeric (98.5%), tensor (98.9%), features (99.0%), distributed/coordinator (99.1%), data (100%), device (100%), internal/xblas (100%), layers/components (100%), layers/registry (100%), layers/tokenizers (100%), metrics (100%), pkg/tokenizer (100%) | 30 |
| 93-94% (near target, tensor.New gaps) | layers/gather (93.1%), layers/embeddings (93.5%), layers/features (93.8%) | 3 |
| Not targeted | cmd/zerfoo, cmd/zerfoo-predict, cmd/zerfoo-tokenize, distributed/pb, pkg/prelude, types, testing/testutils, tests/internal/testutil, tests/helpers | 9 |

### Objectives

- O1: Raise every testable package to at least 95% statement coverage.
- O2: Prioritize packages with the lowest coverage first.
- O3: Use table-driven tests and the standard testing package only. No testify.
- O4: Ensure all tests pass with the race detector enabled.

### Non-Goals

- Testing generated protobuf code (distributed/pb/).
- Unit-testing main() functions in cmd/zerfoo, cmd/zerfoo-predict, cmd/zerfoo-tokenize. These are thin wrappers; testable logic lives in cmd/cli and pkg/tokenizer.
- Testing pkg/prelude (1 line, no statements) or types/ (12 lines, type definitions only).
- Testing test helper packages (testing/testutils, tests/internal/testutil, tests/helpers) to 95%. These are utilities consumed by other tests and are exercised transitively. We will add targeted tests for any complex logic (math functions, mock correctness) but 95% is not required.
- Modifying production code solely for testability unless the change also improves design.

### Constraints and Assumptions

- Use Go standard library only. No third-party test frameworks.
- Use table-driven tests with descriptive subtest names.
- The pre-commit hook rejects commits spanning multiple directories. Each commit must touch files in one directory tree only.
- All changes must pass golangci-lint, go vet, and gofmt before commit.
- Tests must pass with -race flag.
- Coverage is measured by `go test -cover` per package. The target is 95% of statements.

### Success Metrics

| Metric | Target | How to Measure |
|--------|--------|----------------|
| Per-package coverage | >= 95% for all testable packages | `go test ./... -cover` |
| Race detector | Zero data races | `go test ./... -race` |
| Lint | Zero issues | `golangci-lint run ./...` |
| Test suite | All green | `go test ./... -count=1` |

---

## 2. Scope and Deliverables

### In Scope

- Writing new test files for packages at 0% coverage.
- Adding test cases to existing test files for packages below 95%.
- Testing error paths, edge cases, and boundary conditions.
- Testing backward passes and gradient computations for all layers.
- Testing registry builder functions in layers/normalization, layers/gather, layers/reducesum.

### Out of Scope

- Testing generated protobuf code (distributed/pb/).
- Testing main() entrypoints (cmd/zerfoo, cmd/zerfoo-predict, cmd/zerfoo-tokenize).
- Testing trivial type-only packages (types/, pkg/prelude).
- Performance benchmarks (separate effort).
- Increasing coverage by deleting dead code (may be done opportunistically but is not the goal).

### Deliverables

| ID | Description | Owner | Acceptance Criteria |
|----|-------------|-------|---------------------|
| D1 | 0% packages tested | TBD | pkg/tokenizer, layers/gather, layers/reducesum, layers/registry, internal/xblas all >= 95% |
| D2 | Sub-50% packages raised | TBD | training, cmd/cli, layers/normalization, model all >= 95% |
| D3 | 50-69% packages raised | TBD | All 10 packages in this tier >= 95% |
| D4 | 70-89% packages raised | TBD | All 10 packages in this tier >= 95% |
| D5 | 90-94% packages raised | TBD | data (93.5%) raised to >= 95% |
| D6 | Test utility validation | TBD | Targeted tests for math helpers in tests/internal/testutil and complex mocks in testing/testutils |

---

## 3. Checkable Work Breakdown

### E1: Zero-Coverage Packages (Priority 1 -- Critical) -- COMPLETED

- [x] T1.1 Add tests for pkg/tokenizer  Completed: 2026 02 24  Result: 100%
- [x] T1.2 Add tests for layers/gather  Completed: 2026 02 25  Result: 93.1% (tensor.New gaps)
- [x] T1.3 Add tests for layers/reducesum  Completed: 2026 02 24  Result: 95.9%
- [x] T1.4 Add tests for layers/registry  Completed: 2026 02 24  Result: 100%
- [x] T1.5 Add tests for internal/xblas  Completed: 2026 02 24  Result: 100%

### E2: Sub-50% Coverage Packages (Priority 2 -- High) -- COMPLETED

- [x] T2.1 Raise training/ to >= 95%  Completed: 2026 02 24  Result: 95.7%
- [x] T2.2 Raise cmd/cli/ to >= 95%  Completed: 2026 02 24  Result: 96.5%
- [x] T2.3 Raise layers/normalization/ to >= 95%  Completed: 2026 02 24  Result: 96.6%
- [x] T2.4 Raise model/ to >= 95%  Completed: 2026 02 24  Result: 95.4%

### E3: 50-69% Coverage Packages (Priority 3 -- Medium) -- COMPLETED

- [x] T3.1 Raise layers/attention/ to >= 95%  Completed: 2026 02 24  Result: 95.1%
- [x] T3.2 Raise layers/core/ to >= 95%  Completed: 2026 02 24  Result: 96.0%
- [x] T3.3 Raise tensor/ to >= 95%  Completed: 2026 02 24  Result: 98.9%
- [x] T3.4 Raise layers/activations/ to >= 95%  Completed: 2026 02 24  Result: 97.1%
- [x] T3.5 Raise layers/recurrent/ to >= 95%  Completed: 2026 02 24  Result: 96.7%
- [x] T3.6 Raise graph/ to >= 95%  Completed: 2026 02 24  Result: 97.0%
- [x] T3.7 Raise training/loss/ to >= 95%  Completed: 2026 02 24  Result: 96.3%
- [x] T3.8 Raise layers/tokenizers/ to >= 95%  Completed: 2026 02 24  Result: 100%
- [x] T3.9 Raise layers/transpose/ to >= 95%  Completed: 2026 02 24  Result: 97.2%
- [x] T3.10 Raise numeric/ to >= 95%  Completed: 2026 02 24  Result: 98.5%

### E4: 70-89% Coverage Packages (Priority 4 -- Lower)

- [x] T4.1 Raise layers/embeddings/ to >= 95%  Owner: TBD  Completed: 2026 02 25
  - Result: 93.5% (was 72.4%). Remaining 6.5% is tensor.New error paths (unreachable).
  - [x] S4.1.1 Identify untested branches and write targeted tests
  - [x] S4.1.2 Run golangci-lint and go test -cover

- [x] T4.2 Raise layers/transformer/ to >= 95%  Owner: TBD  Completed: 2026 02 24
  - Result: 96.4% (was 72.6%)
  - [x] S4.2.1 Identify untested branches and write targeted tests
  - [x] S4.2.2 Run golangci-lint and go test -cover

- [x] T4.3 Raise distributed/ to >= 95%  Owner: TBD  Completed: 2026 02 24
  - Result: 96.5% (was 75.3%)
  - [x] S4.3.1 Identify untested branches and write targeted tests
  - [x] S4.3.2 Run golangci-lint and go test -cover

- [x] T4.4 Raise model/hrm/ to >= 95%  Owner: TBD  Completed: 2026 02 24
  - Result: 98.1% (was 76.9%)
  - [x] S4.4.1 Identify untested branches and write targeted tests
  - [x] S4.4.2 Run golangci-lint and go test -cover

- [x] T4.5 Raise training/optimizer/ to >= 95%  Owner: TBD  Completed: 2026 02 24
  - Result: 97.4% (was 78.3%)
  - [x] S4.5.1 Identify untested branches and write targeted tests
  - [x] S4.5.2 Run golangci-lint and go test -cover

- [x] T4.6 Raise layers/features/ to >= 95%  Owner: TBD  Completed: 2026 02 25
  - Result: 93.8% (was 79.2%). Remaining 6.2% is tensor.New error paths (unreachable).
  - [x] S4.6.1 Identify untested branches and write targeted tests
  - [x] S4.6.2 Run golangci-lint and go test -cover

- [x] T4.7 Raise layers/components/ to >= 95%  Owner: TBD  Completed: 2026 02 24
  - Result: 100% (was 82.1%)
  - [x] S4.7.1 Identify untested branches and write targeted tests
  - [x] S4.7.2 Run golangci-lint and go test -cover

- [x] T4.8 Raise layers/hrm/ to >= 95%  Owner: TBD  Completed: 2026 02 25
  - Result: 95.5% (was 82.9%)
  - [x] S4.8.1 Identify untested branches and write targeted tests
  - [x] S4.8.2 Run golangci-lint and go test -cover

- [x] T4.9 Raise compute/ to >= 95%  Owner: TBD  Completed: 2026 02 25
  - Result: 96.2% (was 85.5%)
  - [x] S4.9.1 Identify untested branches and write targeted tests
  - [x] S4.9.2 Run golangci-lint and go test -cover

- [x] T4.10 Raise distributed/coordinator/ to >= 95%  Owner: TBD  Completed: 2026 02 25
  - Result: 99.1% (was 88.6%)
  - [x] S4.10.1 Identify untested branches and write targeted tests
  - [x] S4.10.2 Run golangci-lint and go test -cover

### E5: Near-Target Packages (Priority 5)

- [x] T5.1 Raise data/ to >= 95%  Owner: TBD  Completed: 2026 02 25
  - Result: 100% (was 93.5%)
  - [x] S5.1.1 Identify the few untested branches and add tests
  - [x] S5.1.2 Run golangci-lint and go test -cover

- [x] T5.2 Raise features/ to >= 95%  Owner: TBD  Completed: 2026 02 24
  - Result: 99.0% (was 99.0%, already at target)
  - [x] S5.2.1 Verify already at target
  - [x] S5.2.2 Run golangci-lint and go test -cover

### E6: Test Utility Validation (Priority 6 -- Best Effort)

- [ ] T6.1 Add targeted tests for tests/internal/testutil  Owner: TBD  Est: 1h
  - Dependencies: None
  - Location: tests/internal/testutil/testutil.go (~116 LOC)
  - Acceptance: Math helpers (MeanRelativeError, TopKAgreement, RelError) tested for correctness. Coverage target: best effort, not 95%.
  - [ ] S6.1.1 Write tests for MeanRelativeError with known inputs  Est: 15m
  - [ ] S6.1.2 Write tests for TopKAgreement with known overlap  Est: 15m
  - [ ] S6.1.3 Write tests for RelError edge cases (zero denominator)  Est: 10m
  - [ ] S6.1.4 Run golangci-lint and go test -cover  Est: 5m

- [ ] T6.2 Add targeted tests for testing/testutils mock correctness  Owner: TBD  Est: 1h
  - Dependencies: None
  - Location: testing/testutils/ (~997 LOC)
  - Acceptance: MockEngine key methods tested. Coverage target: best effort, not 95%.
  - [ ] S6.2.1 Write tests for assertion helpers (AssertEqual, AssertError, etc.)  Est: 20m
  - [ ] S6.2.2 Write tests for MockEngine interface compliance  Est: 20m
  - [ ] S6.2.3 Run golangci-lint and go test -cover  Est: 5m

### E7: Final Verification

- [ ] T7.1 Run full test suite with coverage  Owner: TBD  Est: 30m
  - Dependencies: E1, E2, E3, E4, E5
  - Acceptance: Every testable package shows >= 95% in `go test ./... -cover` (with documented exceptions)
  - [ ] S7.1.1 Run go test ./... -cover and capture output  Est: 10m
  - [ ] S7.1.2 Verify each package meets target; list any exceptions with justification  Est: 10m
  - [ ] S7.1.3 Run go test ./... -race and verify zero races  Est: 10m

- [ ] T7.2 Run linters and formatters  Owner: TBD  Est: 15m
  - Dependencies: T7.1
  - Acceptance: golangci-lint 0 issues, go vet clean, gofmt clean
  - [ ] S7.2.1 Run golangci-lint run ./...  Est: 5m
  - [ ] S7.2.2 Run go vet ./...  Est: 5m
  - [ ] S7.2.3 Run gofmt -l . and verify no files  Est: 5m

### Documented Coverage Exceptions

Three packages are below the 95% target. In all three cases, the remaining uncovered
code consists exclusively of `tensor.New[T]()` error handling that cannot be triggered
with valid inputs. These are defensive checks against memory allocation failures.

| Package | Coverage | Uncovered Lines | Justification |
|---------|----------|-----------------|---------------|
| layers/gather | 93.1% | gather.go:71-72, 94-95, 127-128, 145-147; registry.go:60-62 | All tensor.New error paths; tensor.New with valid shape never fails |
| layers/embeddings | 93.5% | token_embedding.go:64-66, 84-86, 149-151, 156-158, 165-167, 191-193, 224-226; rotary_positional_embedding.go:92-94, 96-98 | All tensor.New/NewParameter error paths |
| layers/features | 93.8% | spectral.go:63-65, 81-83, 99-101 | All tensor.New error paths in Forward/Backward |

---

## 4. Timeline and Milestones

| ID | Milestone | Dependencies | Exit Criteria |
|----|-----------|--------------|---------------|
| M1 | Zero-coverage packages tested | E1 | pkg/tokenizer, layers/gather, layers/reducesum, layers/registry, internal/xblas all >= 95% |
| M2 | Sub-50% packages at target | E2 | training, cmd/cli, layers/normalization, model all >= 95% |
| M3 | Medium-coverage packages at target | E3 | All 10 packages in 50-69% tier >= 95% |
| M4 | All remaining packages at target | E4, E5 | Every testable package >= 95% |
| M5 | Final verification | E7 | Full suite green, lint clean, race-free |

### Recommended Sequence

1. E1 (0% packages) -- no dependencies, highest impact per test written
2. E2 (sub-50%) -- high impact, most uncovered code
3. E3 (50-69%) -- medium priority, 10 packages
4. E4 (70-89%) -- lower priority, closer to target
5. E5 (near-target) -- quick wins
6. E6 (test utilities) -- best effort
7. E7 (final verification) -- depends on all above

Within each epic, tasks are independent and can be done in any order or in parallel.

---

## 5. Operating Procedure

### Definition of Done

A task is done when:
1. Tests exist for every untested function and branch in the package.
2. `go test -cover ./package/` reports >= 95% statement coverage.
3. `golangci-lint run ./package/` reports 0 issues.
4. `go vet ./package/` reports no issues.
5. Tests pass with `-race` flag.
6. Changes are committed in a small, logical commit touching one directory only.

### Review and QA Steps

1. Before writing tests, use `go test -coverprofile=cover.out ./package/` and `go tool cover -func=cover.out` to identify exactly which lines are uncovered.
2. Write table-driven tests using the standard testing package. No testify.
3. After writing tests, run `go test -cover ./package/` to verify coverage meets target.
4. Run `golangci-lint run --fix ./package/` to fix any lint issues.
5. Run `gofmt -w .` to ensure formatting.
6. Run `go test ./... -count=1` to verify no regressions across the full suite.

### Commit Discipline

- Never commit files from different directories in the same commit. The pre-commit hook will reject it.
- Make small, logical commits -- one task per commit.
- Use Conventional Commits: `test(package): add tests for FunctionName`.
- Never allow changes to pile up. Commit after each completed task.

---

## 6. Progress Log

- **2026 02 25:** Change Summary: All epics E1-E5 completed. 30 of 33 testable packages now at >= 95% coverage. Three packages (layers/gather 93.1%, layers/embeddings 93.5%, layers/features 93.8%) remain below 95% due to unreachable tensor.New error paths. These gaps are documented as acceptable exceptions. Key commits this session: compute (85.5% -> 96.2%), data (93.5% -> 100%), layers/gather (91.7% -> 93.1%), layers/embeddings (92.5% -> 93.5%). Prior sessions raised all other packages to >= 95%.

- **2026 02 24:** Change Summary: Plan created. Defined 7 epics covering 35+ packages, prioritized by coverage tier from 0% to 93.5%. Excluded generated code (distributed/pb), main entrypoints (cmd/zerfoo*), trivial packages (types, pkg/prelude), and test utilities (best-effort only). Target: >= 95% statement coverage for all testable packages.

---

## 7. Hand-off Notes

### For a New Contributor

- **Start here:** Read this plan, then run `go test ./... -cover` to see current state.
- **How to find uncovered lines:** Run `go test -coverprofile=cover.out ./package/ && go tool cover -func=cover.out` to see per-function coverage. Use `go tool cover -html=cover.out` for a visual HTML report.
- **Testing patterns:** The codebase uses table-driven tests with the standard testing package. See `data/dataset_test.go` or `compute/cpu_engine_test.go` for examples.
- **Key constraint:** The pre-commit hook runs `golangci-lint` and `go test ./...`. It rejects commits that touch multiple directories.
- **No credentials required.** All work is local.

---

## 8. Appendix

### Packages Excluded from 95% Target

| Package | Reason |
|---------|--------|
| distributed/pb/ | Generated protobuf code (1518 LOC, all auto-generated) |
| cmd/zerfoo/ | Main entrypoint (33 LOC), no testable logic beyond CLI wiring |
| cmd/zerfoo-predict/ | Main entrypoint; testable logic is in cmd/cli/ |
| cmd/zerfoo-tokenize/ | Main entrypoint; testable logic is in pkg/tokenizer/ |
| pkg/prelude/ | 1 line, no statements to cover |
| types/ | 12 lines, type definitions only |
| testing/testutils/ | Test utility (997 LOC), exercised transitively, best-effort testing in E6 |
| tests/internal/testutil/ | Test utility (116 LOC), best-effort testing in E6 |
| tests/helpers/ | Test helper, no source files with statements |

### References

- Go coverage tool: `go help testflag` (see -cover, -coverprofile, -covermode)
- Coverage visualization: `go tool cover -html=cover.out`
- Race detector: `go test -race ./...`
