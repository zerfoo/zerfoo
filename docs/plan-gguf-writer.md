# Shared GGUF Writer in ztensor

## Context

### Problem Statement

Zerfoo uses GGUF as its sole model format (ADR-037) but has no shared GGUF writer.
Five independent, hand-rolled GGUF write implementations exist across zerfoo and
zonnx, with duplicated logic for header encoding, metadata serialization, tensor
alignment, and string encoding. The generic `SaveModel` interface returns
"not implemented" because there is nothing to build on.

The GGUF writer in zonnx (`pkg/gguf/Writer`) is the best implementation but is
trapped in zonnx -- zerfoo cannot import it (wrong dependency direction). The
solution is to promote the writer to ztensor, the shared foundation library that
both zerfoo and zonnx already import.

Decision rationale: docs/adr/061-gguf-writer-in-ztensor.md

### Objectives

- Create a single, tested GGUF v3 writer in `github.com/zerfoo/ztensor/gguf`.
- Replace all five hand-rolled writers in zerfoo with the shared writer.
- Update zonnx to import the shared writer and delete its local copy.
- Implement `SaveModel` in `training/adapter.go` using the shared writer.

### Non-Goals

- GGUF reading/parsing (stays in `zerfoo/model/gguf`).
- Tensor name mapping or model config extraction (stays in consumers).
- GGUF v4 or new format features (future work).
- Changing the GGUF reader in zerfoo.

### Constraints and Assumptions

- ztensor, zerfoo, and zonnx are separate repos with independent go.mod files.
- ztensor must be released first; zerfoo and zonnx update their go.mod after.
- The writer must be format-only: no domain knowledge (no model names, no
  architecture-specific logic).
- Standard library only. No third-party dependencies.
- All existing tests must continue to pass after migration.

### Success Metrics

| Metric | Target |
|--------|--------|
| Hand-rolled GGUF writers in zerfoo | 0 (currently 4) |
| Hand-rolled GGUF writers in zonnx | 0 (currently 1) |
| SaveModel implementation | Functional (currently "not implemented") |
| Test coverage for ztensor/gguf | >= 90% line coverage |
| GGUF round-trip test | Write + read back + compare passes |

---

## Scope and Deliverables

### In Scope

- New `gguf/` package in ztensor with Writer, constants, and metadata types.
- Round-trip test: write GGUF with ztensor writer, read back with zerfoo parser.
- Migration of 4 hand-rolled writers in zerfoo (lora, fsdp, nas, ts_train).
- Migration of 1 writer in zonnx.
- Implementation of `SaveModel` in `training/adapter.go`.
- ztensor release with new package.

### Out of Scope

- GGUF reader changes in `zerfoo/model/gguf`.
- zonnx `pkg/gguf/metadata.go` and `pkg/gguf/tensornames.go` (ONNX-specific mapping).
- New GGUF metadata types or format extensions.
- zonnx's ZMF intermediate representation cleanup (separate effort).

### Deliverables Table

| ID | Description | Owner | Acceptance Criterion |
|----|-------------|-------|----------------------|
| D1 | ztensor/gguf package | Kernel Eng | Writer produces valid GGUF v3 files readable by zerfoo parser and llama.cpp |
| D2 | zerfoo writer consolidation | Lead Eng | All 4 hand-rolled writers replaced; all existing tests pass |
| D3 | SaveModel implementation | Lead Eng | `training/adapter.go:SaveModel` writes GGUF; round-trip test passes |
| D4 | zonnx writer migration | Lead Eng | zonnx imports ztensor/gguf; local writer.go deleted; all tests pass |

---

## Checkable Work Breakdown

### E1: Create ztensor/gguf Package [ztensor repo]

- [ ] T1.1 Create gguf/constants.go with GGUF v3 magic, version, alignment, type constants
  Owner: TBD  Est: 30m
  Deps: none
  Acceptance: Constants match GGUF v3 spec. Magic=0x46554747, Version=3, Alignment=64.
  All metadata value types (0-12) and common tensor dtypes (F32, F16, Q4_0, Q8_0,
  BF16, Q4_K, Q5_K, Q6_K, Q8_K) defined. TestConstants verifies values.

- [ ] T1.2 Create gguf/writer.go with Writer struct and metadata/tensor methods
  Owner: TBD  Est: 2h
  Deps: T1.1
  Acceptance: Writer struct with NewWriter(io.Writer), AddMetadataString,
  AddMetadataUint32, AddMetadataInt32, AddMetadataUint64, AddMetadataFloat32,
  AddMetadataBool, AddMetadataStringArray, AddMetadataUint32Array, AddTensor(name,
  dtype, shape, data), Flush(). Shape dimensions reversed on write per GGUF spec.
  Tensor data aligned to 64-byte boundaries. Based on zonnx/pkg/gguf/writer.go
  with these additions:
  - AddMetadataUint64 (needed by fsdp checkpoint for large tensor counts)
  - AddMetadataUint32Array (needed for tokenizer token type arrays)
  All methods documented with GoDoc comments.

- [ ] T1.3 Write unit tests for gguf/writer.go
  Owner: TBD  Est: 1h30m
  Deps: T1.2
  Acceptance: Tests cover: empty file (metadata only, no tensors), single tensor,
  multiple tensors with different dtypes, all metadata types, alignment padding
  correctness, dimension reversal, large tensor (>1MB), zero-length tensor name
  edge case. >= 90% line coverage. TestWriterFlush, TestWriterAlignment,
  TestWriterMetadataTypes, TestWriterDimensionReversal all pass.

- [ ] T1.4 Create gguf/reader.go with minimal Reader for round-trip testing
  Owner: TBD  Est: 1h30m
  Deps: T1.1
  Acceptance: Reader struct with Parse(io.ReadSeeker) that reads header, metadata
  KV pairs, and tensor info entries. Returns File struct with Metadata map and
  TensorInfos slice. Does NOT load tensor data (that is zerfoo's job). This reader
  exists for ztensor's own round-trip tests and for zonnx validation. TestParse
  passes on files written by Writer.

- [ ] T1.5 Write round-trip integration test: Writer -> Reader
  Owner: TBD  Est: 1h
  Deps: T1.2, T1.3, T1.4
  Acceptance: Test writes a GGUF file with 3 metadata KVs and 2 tensors using Writer,
  reads it back with Reader, verifies all metadata values and tensor info match.
  TestRoundTrip passes.

- [ ] T1.6 Run go vet and golangci-lint on ztensor/gguf
  Owner: TBD  Est: 15m
  Deps: T1.2, T1.3, T1.4, T1.5
  Acceptance: `go vet ./gguf/...` and `golangci-lint run ./gguf/...` produce zero
  warnings. `go test ./gguf/... -race` passes.

---

### E2: Migrate zerfoo Writers to ztensor/gguf [zerfoo repo]

- [ ] T2.1 Update zerfoo go.mod to import ztensor version with gguf package
  Owner: TBD  Est: 15m
  Deps: E1 (all tasks)
  Acceptance: `go mod tidy` succeeds. `go build ./...` compiles. No replace
  directives needed (or local replace for development).

- [ ] T2.2 Migrate training/lora/checkpoint.go to use ztensor/gguf.Writer
  Owner: TBD  Est: 1h
  Deps: T2.1
  Acceptance: SaveAdapter uses ztensor/gguf.Writer instead of hand-rolled binary
  writes. Delete local writeGGUFString function. All existing lora checkpoint tests
  pass unchanged. `go test ./training/lora/... -race` passes.

- [ ] T2.3 Migrate training/nas/export.go to use ztensor/gguf.Writer
  Owner: TBD  Est: 1h
  Deps: T2.1
  Acceptance: ExportGGUF uses ztensor/gguf.Writer. Delete local GGUF binary write
  helpers. ValidateExportRoundTrip and all TestExportGGUF* tests pass unchanged.
  `go test ./training/nas/... -race` passes.

- [ ] T2.4 Migrate distributed/fsdp/checkpoint.go to use ztensor/gguf.Writer
  Owner: TBD  Est: 1h
  Deps: T2.1
  Acceptance: SaveCheckpoint uses ztensor/gguf.Writer. Delete local writeGGUF and
  writeGGUFString functions. All fsdp checkpoint tests pass unchanged.
  `go test ./distributed/fsdp/... -race` passes.

- [ ] T2.5 Migrate cmd/ts_train/main.go to use ztensor/gguf.Writer
  Owner: TBD  Est: 45m
  Deps: T2.1
  Acceptance: saveModelGGUF uses ztensor/gguf.Writer. Delete local saveModelGGUF
  and writeGGUFString functions. `go build ./cmd/ts_train/` succeeds.
  `go test ./cmd/ts_train/... -race` passes (if tests exist).

- [ ] T2.6 Migrate inference test helpers to use ztensor/gguf.Writer
  Owner: TBD  Est: 45m
  Deps: T2.1
  Acceptance: ggufWriter struct in inference/load_gguf_test.go,
  inference/arch_phi_test.go, and inference/structured_output_test.go replaced with
  ztensor/gguf.Writer. All inference tests pass unchanged.
  `go test ./inference/... -race` passes.

- [ ] T2.7 Implement SaveModel in training/adapter.go using ztensor/gguf.Writer
  Owner: TBD  Est: 1h30m
  Deps: T2.1
  Acceptance: SimpleModelProvider.SaveModel extracts graph parameters, writes them
  as GGUF file with architecture metadata. Replaces "not implemented" error.
  TestSimpleModelProvider_SaveModel updated to verify file is written and readable.
  Round-trip test: save model -> load with inference.LoadGGUF -> verify tensor
  shapes match. `go test ./training/... -race` passes.

- [ ] T2.8 Run go vet and golangci-lint on all changed packages
  Owner: TBD  Est: 15m
  Deps: T2.2, T2.3, T2.4, T2.5, T2.6, T2.7
  Acceptance: `go vet ./...` and `golangci-lint run ./...` produce zero warnings
  in zerfoo repo. `go test ./... -race -timeout 300s` passes.

---

### E3: Migrate zonnx Writer to ztensor/gguf [zonnx repo]

- [ ] T3.1 Update zonnx go.mod to import ztensor version with gguf package
  Owner: TBD  Est: 15m
  Deps: E1 (all tasks)
  Acceptance: `go mod tidy` succeeds. Remove local replace directive for zmf if
  possible (separate concern but check). `go build ./...` compiles.

- [ ] T3.2 Migrate zonnx converter to use ztensor/gguf.Writer
  Owner: TBD  Est: 1h
  Deps: T3.1
  Acceptance: pkg/gguf/metadata.go and pkg/gguf/tensornames.go updated to import
  constants from ztensor/gguf instead of local definitions. Converter pipeline uses
  ztensor/gguf.Writer for all GGUF output. `go test ./... -race` passes.

- [ ] T3.3 Delete zonnx pkg/gguf/writer.go and deduplicated constants
  Owner: TBD  Est: 30m
  Deps: T3.2
  Acceptance: writer.go and writer_test.go deleted. Constants (Magic, Version3,
  Alignment, DType*, Type*) removed from pkg/gguf/ and imported from ztensor/gguf.
  All remaining tests pass. `go test ./... -race` passes.

- [ ] T3.4 Run go vet and golangci-lint on zonnx
  Owner: TBD  Est: 15m
  Deps: T3.2, T3.3
  Acceptance: `go vet ./...` and `golangci-lint run ./...` produce zero warnings.

---

## Parallel Work

### Parallel Tracks

| Track | Tasks | Description |
|-------|-------|-------------|
| A: ztensor writer | T1.1, T1.2, T1.3, T1.4, T1.5, T1.6 | New shared GGUF package |
| B: zerfoo migration | T2.1-T2.8 | Replace hand-rolled writers |
| C: zonnx migration | T3.1-T3.4 | Replace zonnx writer |

Sync points:
- Track A must complete before Tracks B and C can start (T2.1 and T3.1 depend on E1).
- Tracks B and C are independent and run in parallel after Track A.

### Maximum Parallelism

**Wave 1** (up to 2 parallel tasks -- ztensor only):
- T1.1 Create constants.go
- T1.4 Create reader.go (depends only on T1.1 constants, can start with stub)

**Wave 2** (up to 2 parallel tasks):
- T1.2 Create writer.go (deps: T1.1)
- T1.4 continued if not done

**Wave 3** (up to 2 parallel tasks):
- T1.3 Writer unit tests (deps: T1.2)
- T1.5 Round-trip integration test (deps: T1.2, T1.4)

**Wave 4** (1 task):
- T1.6 Lint and vet (deps: all T1.*)

**Wave 5** (up to 7 parallel tasks -- zerfoo + zonnx):
- T2.1 Update zerfoo go.mod
- T3.1 Update zonnx go.mod

**Wave 6** (up to 7 parallel tasks):
- T2.2 Migrate lora checkpoint
- T2.3 Migrate nas export
- T2.4 Migrate fsdp checkpoint
- T2.5 Migrate ts_train
- T2.6 Migrate inference test helpers
- T2.7 Implement SaveModel
- T3.2 Migrate zonnx converter

**Wave 7** (up to 3 parallel tasks):
- T2.8 Lint zerfoo
- T3.3 Delete zonnx writer
- T3.4 Lint zonnx

---

## Timeline and Milestones

| ID | Milestone | Deps | Exit Criteria |
|----|-----------|------|---------------|
| M1 | ztensor/gguf package released | E1 | Package published, round-trip test passes, ztensor tagged |
| M2 | zerfoo writers consolidated | E2 | Zero hand-rolled writers, SaveModel works, all tests pass |
| M3 | zonnx migrated | E3 | zonnx imports ztensor/gguf, local writer deleted, tests pass |

Estimated total effort: ~14 hours across 3 repos.
Estimated calendar time: 2-3 days (waves 1-7).

---

## Risk Register

| ID | Risk | Impact | Likelihood | Mitigation |
|----|------|--------|------------|------------|
| R1 | Binary compatibility: new writer produces slightly different bytes than old writers | High | Medium | Round-trip test reads back with existing zerfoo parser. Byte-exact comparison not required; semantic equivalence is sufficient. |
| R2 | Coordinated release across 3 repos fails | Medium | Low | Release ztensor first, use go.mod replace directives during development, remove replace before final commit. |
| R3 | Existing tests depend on specific binary layout of hand-rolled writers | Medium | Low | Review each test before migration. If a test checks raw bytes, update it to use the reader for validation instead. |
| R4 | zonnx has additional writer features not captured in ztensor version | Low | Low | zonnx writer was used as the reference; audit before deleting. |

---

## Operating Procedure

### Definition of Done

1. Code compiles: `go build ./...` in the target repo directory.
2. Tests pass: `go test ./... -race -timeout 300s` in the target repo.
3. No vet warnings: `go vet ./...` clean.
4. Acceptance criteria satisfied as written in the task.
5. Each task committed as its own commit. One logical change per commit.

### Quality Gates

- Every implementation task must have a paired test.
- Run `go vet ./...` after every code change before committing.
- Standard library only: no testify, no cobra.
- Never commit files from different repo directories in the same commit.
- Run `golangci-lint` on all changed packages before committing.

### Release Order

1. Commit and tag ztensor with new gguf/ package.
2. Update zerfoo go.mod to new ztensor version, migrate writers, commit and tag.
3. Update zonnx go.mod to new ztensor version, migrate writer, commit and tag.

---

## Progress Log

### 2026-03-18: Plan created

Created plan for consolidating GGUF writer into ztensor. 3 epics, 18 tasks,
7 waves. ADR-061 created (docs/adr/061-gguf-writer-in-ztensor.md).

---

## Hand-Off Notes

### What You Need to Know

- **Reference implementation:** zonnx/pkg/gguf/writer.go is the best existing
  writer. Copy its structure for ztensor/gguf/writer.go, then extend with
  AddMetadataUint64 and AddMetadataUint32Array.
- **Repos:** ztensor at /Users/dndungu/Code/zerfoo/ztensor, zerfoo at
  /Users/dndungu/Code/zerfoo/zerfoo, zonnx at /Users/dndungu/Code/zerfoo/zonnx.
  Each has its own go.mod. Never commit across repos.
- **Existing GGUF reader:** zerfoo/model/gguf/parser.go -- use this to validate
  round-trip correctness. Do NOT modify the reader.
- **SaveModel gap:** training/adapter.go:287 returns "not implemented". T2.7
  closes this gap.
- **Constants duplication:** zerfoo/model/gguf/parser.go and zonnx/pkg/gguf/writer.go
  both define Magic, type constants, dtype constants independently. The ztensor
  package becomes the single source of truth. zerfoo's parser.go constants can
  remain (reader does not need to import ztensor/gguf for reading), but new code
  should import from ztensor/gguf.
- **ADR:** docs/adr/061-gguf-writer-in-ztensor.md documents the decision.
