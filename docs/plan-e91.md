# E91: Extract crossasset Package from Zerfoo to Wolf

## Context

### Problem Statement

The `crossasset/` package in zerfoo implements a domain-specific multi-source
attention model for financial market direction prediction. It is an
application-level model, not a generic ML building block. Zerfoo is a generic
ML framework; domain-specific models belong in the systems that use them.

Wolf (`feza-ai/wolf`) is the only consumer of `crossasset/`. It already imports
`github.com/zerfoo/zerfoo/crossasset` in two places. Moving crossasset to wolf
collapses that dependency and co-locates the model with its training and
inference pipelines.

### Objectives

1. Move `crossasset/` (2,672 lines, 10 files) from zerfoo to wolf.
2. Move `timeseries/crossasset_engine.go` (274 lines) adapter from zerfoo to wolf.
3. Update wolf imports to reference the new local package.
4. Remove all crossasset references from zerfoo.
5. Cut zerfoo v3.0.0 for the breaking change.

### Non-Goals

- Refactoring the crossasset model itself (move as-is).
- Adding new features to crossasset during extraction.
- Changing wolf's training or inference pipelines beyond import path updates.

### Constraints and Assumptions

- Wolf already depends on `github.com/zerfoo/zerfoo` and `github.com/zerfoo/ztensor`.
- Wolf's go.mod uses a pseudo-version pin for zerfoo (v1.46.1-0.20260412...).
- crossasset imports only `github.com/zerfoo/ztensor/compute`,
  `github.com/zerfoo/ztensor/tensor`, and `github.com/zerfoo/ztensor/numeric`.
  It does NOT import other zerfoo packages, so it can move cleanly.
- The `timeseries/crossasset_engine.go` adapter imports `crossasset` -- it must
  move to wolf alongside the package or be deleted.
- Zerfoo uses release-please for versioning. Removing crossasset is a breaking
  change that triggers a major version bump.

### Success Metrics

- `go build ./...` and `go test ./...` pass in both zerfoo and wolf after extraction.
- Zero references to `crossasset` remain in zerfoo source code.
- Wolf's `cmd/train-crossasset` and `internal/model/crossasset.go` import from
  `github.com/feza-ai/wolf/crossasset` (or a subpackage).
- Zerfoo v3.0.0 released.

### Decision Rationale

See docs/adr/084-extract-crossasset-to-wolf.md.

## Discovery Summary

### Inventory

**Files to move from zerfoo to wolf:**

| File | Lines | Purpose |
|------|-------|---------|
| `crossasset/crossasset.go` | 714 | Model struct, Forward, Train, Predict, Engine ops |
| `crossasset/backward.go` | 531 | Backward pass with SDPA, LayerNorm, FFN |
| `crossasset/gpu_train.go` | 182 | TrainGPU with weight upload |
| `crossasset/adamw.go` | 136 | AdamW optimizer for crossasset |
| `crossasset/serialize.go` | 218 | Model save/load (.zcam format) |
| `crossasset/doc.go` | 13 | Package documentation |
| `crossasset/crossasset_test.go` | 520 | Unit tests |
| `crossasset/gpu_train_test.go` | 120 | GPU training tests |
| `crossasset/gpu_parity_test.go` | 105 | GPU vs CPU parity test |
| `crossasset/serialize_test.go` | 133 | Serialization tests |
| `timeseries/crossasset_engine.go` | 274 | Timeseries training adapter |
| **Total** | **2,946** | |

**Wolf files that import crossasset (need import path update):**

| File | Current Import |
|------|---------------|
| `cmd/train-crossasset/main.go` | `github.com/zerfoo/zerfoo/crossasset` |
| `internal/model/crossasset.go` | `github.com/zerfoo/zerfoo/crossasset` |

**Zerfoo files with crossasset references (need cleanup after removal):**

| File | Reference Type |
|------|---------------|
| `docs/plan.md` | E60, E68, E90 epic descriptions |
| `docs/adr/084-extract-crossasset-to-wolf.md` | ADR for this extraction |

### Use Cases

3 use cases identified. Reference: .claude/scratch/usecases-manifest.json.

| ID | Priority | Name | Status |
|----|----------|------|--------|
| UC-EXT-01 | P0 | Import crossasset from wolf | PLANNED |
| UC-EXT-02 | P0 | Train crossasset via wolf CLI | PLANNED |
| UC-EXT-03 | P0 | Zerfoo builds without crossasset | PLANNED |

## Scope and Deliverables

### In Scope

- Copy crossasset/ files to wolf repo with updated package path.
- Copy timeseries/crossasset_engine.go to wolf.
- Update wolf imports.
- Remove crossasset/ and crossasset_engine.go from zerfoo.
- Remove crossasset references from zerfoo plan epics (E60, E68, E90).
- Run full test suites in both repos.
- Cut zerfoo v3.0.0.

### Out of Scope

- Refactoring crossasset internals.
- Adding GPU CI to wolf.
- Changing crossasset's public API.
- Moving any other zerfoo packages.

### Deliverables

| ID | Description | Acceptance Criteria |
|----|-------------|---------------------|
| D1 | crossasset package in wolf | `go test ./crossasset/...` passes in wolf |
| D2 | crossasset adapter in wolf | `crossasset_engine.go` compiles and is importable |
| D3 | crossasset removed from zerfoo | Zero crossasset imports in zerfoo; `go build ./...` passes |
| D4 | Wolf imports updated | `cmd/train-crossasset` and `internal/model` use wolf-local imports |
| D5 | Zerfoo v3.0.0 released | GitHub release created via release-please |

## Checkable Work Breakdown

### E91.1: Copy crossasset to wolf

- [ ] T91.1.1 Create `crossasset/` directory in wolf repo  Owner: TBD  Est: 15m  verifies: [UC-EXT-01]
  Copy all 10 files from zerfoo/crossasset/ to wolf/crossasset/.
  Update package imports: replace `github.com/zerfoo/zerfoo/crossasset` references
  within the package itself (if any internal cross-references exist).
  AC: `go build ./crossasset/...` passes in wolf.

- [ ] T91.1.2 Copy crossasset_engine.go adapter to wolf  Owner: TBD  Est: 15m  verifies: [UC-EXT-01]
  Copy `timeseries/crossasset_engine.go` to an appropriate location in wolf
  (suggested: `wolf/crossasset/engine.go` or `wolf/internal/crossasset/engine.go`).
  Update the import from `github.com/zerfoo/zerfoo/crossasset` to the wolf-local path.
  AC: File compiles in wolf.

- [ ] T91.1.3 Run crossasset tests in wolf  Owner: TBD  Est: 15m  verifies: [UC-EXT-01]
  Run `go test ./crossasset/... -count=1 -race -timeout 120s`.
  AC: All tests pass (GPU tests skip on CPU-only machines).

### E91.2: Update wolf imports

- [ ] T91.2.1 Update cmd/train-crossasset/main.go imports  Owner: TBD  Est: 15m  verifies: [UC-EXT-02]
  Change `github.com/zerfoo/zerfoo/crossasset` to `github.com/feza-ai/wolf/crossasset`.
  AC: `go build ./cmd/train-crossasset/...` passes.

- [ ] T91.2.2 Update internal/model/crossasset.go imports  Owner: TBD  Est: 15m  verifies: [UC-EXT-02]
  Change `github.com/zerfoo/zerfoo/crossasset` to `github.com/feza-ai/wolf/crossasset`.
  AC: `go build ./internal/model/...` passes.

- [ ] T91.2.3 Run full wolf test suite  Owner: TBD  Est: 30m  verifies: [UC-EXT-02]
  Run `go test ./... -count=1 -timeout 300s`.
  AC: All tests pass. No references to `github.com/zerfoo/zerfoo/crossasset` remain.

- [ ] T91.2.4 Run go vet and linter in wolf  Owner: TBD  Est: 15m  verifies: [infrastructure]
  AC: Zero errors, zero warnings.

### E91.3: Remove crossasset from zerfoo

- [ ] T91.3.1 Delete crossasset/ directory from zerfoo  Owner: TBD  Est: 15m  verifies: [UC-EXT-03]
  Remove the entire `crossasset/` directory.
  AC: Directory no longer exists.

- [ ] T91.3.2 Delete timeseries/crossasset_engine.go from zerfoo  Owner: TBD  Est: 15m  verifies: [UC-EXT-03]
  Remove the adapter file.
  AC: File no longer exists.

- [ ] T91.3.3 Remove crossasset references from zerfoo code  Owner: TBD  Est: 30m  verifies: [UC-EXT-03]
  Grep for any remaining `crossasset` references in Go source files.
  Fix any compilation errors from removed imports.
  AC: `grep -rn crossasset --include="*.go" .` returns zero results (excluding docs/adr/).

- [ ] T91.3.4 Run full zerfoo test suite  Owner: TBD  Est: 30m  verifies: [UC-EXT-03]
  Run `go build ./...` and `go test ./... -count=1 -timeout 120s`.
  AC: Build and all tests pass.

- [ ] T91.3.5 Run go vet and linter in zerfoo  Owner: TBD  Est: 15m  verifies: [infrastructure]
  AC: Zero errors.

### E91.4: Ship and release

- [ ] T91.4.1 Commit and push wolf changes  Owner: TBD  Est: 15m  verifies: [infrastructure]
  Create PR in wolf repo with the extracted crossasset package.
  AC: PR created, CI passes, merged.

- [ ] T91.4.2 Commit and push zerfoo changes  Owner: TBD  Est: 15m  verifies: [infrastructure]
  Create PR in zerfoo repo removing crossasset.
  Use conventional commit with breaking change: `feat(crossasset)!: extract to feza-ai/wolf`.
  AC: PR created, CI passes, merged.
  Deps: T91.4.1 (wolf PR must merge first so wolf has the package before zerfoo removes it).

- [ ] T91.4.3 Update wolf go.mod to use released zerfoo v3  Owner: TBD  Est: 15m  verifies: [infrastructure]
  After zerfoo v3.0.0 is released, update wolf's go.mod to import
  `github.com/zerfoo/zerfoo/v3` (Go module v3 path).
  Note: This changes ALL zerfoo imports in wolf to use the /v3 path.
  AC: `go build ./...` passes in wolf with v3 import paths.
  Deps: T91.4.2.

- [ ] T91.4.4 Cut zerfoo v3.0.0 release  Owner: TBD  Est: 15m  verifies: [infrastructure]
  Merge release-please PR that bumps to v3.0.0.
  Verify GitHub release is created.
  AC: `gh release view v3.0.0` succeeds.
  Deps: T91.4.2.

- [ ] T91.4.5 Update plan.md  Owner: TBD  Est: 15m  delivers: [updated plan reflecting extraction]
  Mark E91 complete. Update E60, E68, E90 descriptions to note crossasset
  was extracted to wolf. Update status summary.
  AC: Plan reflects current state.

## Parallel Work

### Parallel Tracks

| Track | Tasks | Description | Sync Point |
|-------|-------|-------------|------------|
| A: Wolf copy | T91.1.1, T91.1.2, T91.1.3 | Copy files to wolf, verify tests | Before Track C |
| B: Wolf imports | T91.2.1, T91.2.2, T91.2.3, T91.2.4 | Update wolf imports | After Track A |
| C: Zerfoo removal | T91.3.1-T91.3.5 | Remove from zerfoo | After Track A (independent of B) |
| D: Ship | T91.4.1-T91.4.5 | PRs, releases | After B and C |

### Waves

#### Wave 1: Copy to wolf (1 agent)
All copy tasks are sequential within one repo, best handled by one agent.

- [ ] Agent 1: T91.1.1 + T91.1.2 + T91.1.3 (copy crossasset to wolf, run tests)

#### Wave 2: Update imports + remove from zerfoo (2 agents)
These are in different repos, fully independent.

- [ ] Agent 1: T91.2.1 + T91.2.2 + T91.2.3 + T91.2.4 (update wolf imports, run tests)
- [ ] Agent 2: T91.3.1 + T91.3.2 + T91.3.3 + T91.3.4 + T91.3.5 (remove from zerfoo, run tests)

#### Wave 3: Ship (1 agent)
Sequential: wolf PR first, then zerfoo PR, then release.

- [ ] Agent 1: T91.4.1 + T91.4.2 + T91.4.3 + T91.4.4 + T91.4.5

## Timeline and Milestones

| ID | Milestone | Exit Criteria | Dependencies |
|----|-----------|---------------|--------------|
| M1 | crossasset compiles in wolf | `go build ./crossasset/...` passes in wolf | T91.1.3 |
| M2 | Wolf fully migrated | All wolf tests pass with local crossasset imports | T91.2.3 |
| M3 | Zerfoo clean | Zero crossasset references, all tests pass | T91.3.4 |
| M4 | v3.0.0 released | GitHub release exists | T91.4.4 |

## Risk Register

| ID | Risk | Impact | Likelihood | Mitigation |
|----|------|--------|------------|------------|
| R1 | crossasset has hidden dependencies on zerfoo internals | High | Low | Audit imports before copy; crossasset only imports ztensor, not other zerfoo packages |
| R2 | Wolf CI breaks due to missing GPU test infrastructure | Medium | Medium | GPU tests already skip on CPU-only; wolf CI uses ubuntu runners |
| R3 | Go module v3 path change breaks wolf imports | Medium | Medium | Update all imports in one commit; use `go mod edit` for module path |
| R4 | timeseries/crossasset_engine.go has callers beyond wolf | Low | Low | Grep zerfoo for CrossAssetConfig references; only used by timeseries package internally |

## Operating Procedure

- Definition of done: PR merged, CI green, release created where applicable.
- Both repos must pass `go build ./...` and `go test ./...` independently.
- Wolf PR merges before zerfoo PR (wolf needs the package before zerfoo removes it).
- Use conventional commits with `!` for the breaking change in zerfoo.
- Rebase and merge (not squash, not merge commits) per project convention.

## Progress Log

### 2026-04-12: Plan created
- Created E91 plan for crossasset extraction from zerfoo to wolf.
- Created ADR 084 documenting the decision rationale.
- Identified 2,946 lines across 11 files to move.
- Wolf already imports crossasset in 2 files -- straightforward import path update.
- 14 tasks across 4 sub-epics, 3 waves, estimated ~4 hours total.

## Hand-off Notes

- Wolf repo is at `../../feza-ai/wolf` relative to zerfoo.
- Wolf already depends on zerfoo v1.46.1 (pseudo-version) and ztensor v1.5.0.
- crossasset imports only ztensor packages (compute, tensor, numeric), not other zerfoo packages. This makes the extraction clean.
- The `timeseries/crossasset_engine.go` adapter bridges crossasset.Model into the timeseries.TrainWindowed interface. It must move with crossasset.
- After extraction, zerfoo v3.0.0 will use Go module major version path (`github.com/zerfoo/zerfoo/v3`). All downstream consumers (wolf) must update their import paths.
- GPU parity tests for crossasset will skip on CPU-only CI (same behavior as in zerfoo).

## Appendix

### Files inventory (zerfoo source)

```
crossasset/
  adamw.go           136 lines  -- AdamW optimizer
  backward.go        531 lines  -- backward pass with SDPA
  crossasset.go      714 lines  -- Model, Forward, Train, Predict
  crossasset_test.go 520 lines  -- unit tests
  doc.go              13 lines  -- package doc
  gpu_parity_test.go 105 lines  -- GPU vs CPU parity test
  gpu_train.go       182 lines  -- TrainGPU with weight upload
  gpu_train_test.go  120 lines  -- GPU training tests
  serialize.go       218 lines  -- save/load .zcam format
  serialize_test.go  133 lines  -- serialization tests

timeseries/
  crossasset_engine.go 274 lines -- timeseries training adapter
```

### Wolf consumer files

```
cmd/train-crossasset/main.go          -- CLI for training
internal/model/crossasset.go          -- runtime model wrapper
```
