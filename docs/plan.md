# Zerfoo Phase 17: Repository Extraction + GGUF-Only Model Format

## 1. Context

### Problem Statement

Two structural problems limit the Zerfoo ecosystem:

**1. Monolithic zerfoo repository.** The zerfoo repo (~50,000 lines of Go)
contains everything from tensor storage and GPU kernels to HTTP servers and
gRPC distributed training. Importing GPU tensors or a BPE tokenizer pulls in
the entire framework. See docs/adr/036-ztensor-ztoken-repo-extraction.md.

**2. Two inference paths with 50x performance gap.** The GGUF path achieves
232 tok/s with 99.5% CUDA graph capture. The ZMF/ONNX path achieves 4-16 tok/s
with 1-4% capture. The ZMF path uses protobuf for tensor storage (no mmap, 2x
memory), stores decomposed operation graphs that cannot be efficiently fused,
and has a blocking bug (PR #70 RMSNorm fusion). Every model people want to run
has GGUF variants on HuggingFace. See docs/adr/037-gguf-only-drop-zmf-model-format.md.

### Objectives

- O1: Extract tensor/compute/graph into `github.com/zerfoo/ztensor`.
- O2: Extract tokenizer into `github.com/zerfoo/ztoken`.
- O3: Drop ZMF as a model format. Make GGUF the sole model loading format.
- O4: Remove ~7,500 lines of ZMF-dependent code from zerfoo. Close PR #70.
- O5: Pivot zonnx from ONNX-to-ZMF to ONNX-to-GGUF converter.
- O6: Archive the zmf repository.
- O7: All tests pass in all repositories after changes.

### Non-Goals

- New model architectures or performance optimizations.
- Training checkpoint implementation (design only; implementation deferred).
- Python bindings or documentation website.
- Breaking the public API of zerfoo for existing GGUF users.

### Constraints and Assumptions

- Pre-commit hook rejects multi-directory commits.
- Each new repo needs: go.mod, LICENSE (Apache 2.0), README.md, Makefile,
  .github/workflows/ci.yml, .gitignore, .goreleaser.yml, release-please config.
- DGX Spark: ssh ndungu@192.168.86.250, project at ~/zerfoo.
- Go 1.25 with purego GPU bindings (no CGo for CUDA).
- Rebase and merge workflow (not squash, not merge commits).
- Training checkpoint saving is currently unimplemented (SaveModel stub).

### Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| ztensor tests | 100% pass | `cd ztensor && go test ./...` |
| ztoken tests | 100% pass | `cd ztoken && go test ./...` |
| zerfoo tests | 100% pass after all changes | `cd zerfoo && go test ./...` |
| zonnx tests | 100% pass after GGUF pivot | `cd zonnx && go test ./...` |
| ztensor deps | Only float16, float8, stdlib | `go list -m all` |
| ztoken deps | Only stdlib | `go list -m all` |
| zerfoo zmf dep | Removed from go.mod | grep zmf go.mod returns nothing |
| Code removed | ~7,500 lines of ZMF code deleted | git diff --stat |

### Current State

- v1.2.0 released. All Phase 13-16 work complete.
- 5 models run on DGX Spark. Gemma 3 GGUF at 232 tok/s.
- ONNX models at 4-16 tok/s. RMSNorm fusion blocked (PR #70).
- Training SaveModel is a stub returning "not implemented".

### Checkpoint and Architecture Storage Design

**How will training checkpoints be stored without ZMF?**

GGUF is the answer. GGUF supports:
- Arbitrary key-value metadata (string, int, float, bool, arrays).
- Named tensors with shapes, data types, and raw aligned data.
- Mmap-friendly layout for fast loading.

A training checkpoint GGUF file contains:
- Model weights as named tensors (same names as inference GGUF).
- Optimizer state as additional tensors with `optimizer.` prefix
  (e.g., `optimizer.m.layers.0.weight` for Adam first moment).
- Training metadata as key-value pairs: `training.epoch`, `training.step`,
  `training.learning_rate`, `training.optimizer`, `training.loss`.
- Architecture metadata: same keys GGUF already uses
  (`llama.embedding_length`, `llama.block_count`, etc.).

This means checkpoints are loadable by any GGUF reader (llama.cpp, Ollama)
after stripping the optimizer tensors.

**How will model architecture be described?**

Architecture is NOT stored in a file. Instead, architecture-specific builders
in zerfoo construct the computation graph from GGUF metadata at load time.
Each architecture has a builder file (~200 lines):
- `inference/arch_llama.go` (Llama, Mistral, Qwen, Phi, DeepSeek)
- `inference/arch_gemma.go` (Gemma 2, 3)

Adding a new architecture means writing a new builder that reads GGUF metadata
and calls the layer constructors. This is the same approach llama.cpp uses.
The graph is ephemeral (exists only in memory during inference), not serialized.

---

## 2. Scope and Deliverables

### In Scope

- Create github.com/zerfoo/ztensor with tensor/compute/graph code.
- Create github.com/zerfoo/ztoken with tokenizer code.
- Remove ZMF model loading, ZMF export, generic graph builder, tensor
  codec, graph fusion pass, and all associated tests from zerfoo.
- Remove `github.com/zerfoo/zmf` dependency from zerfoo go.mod.
- Close PR #70 (RMSNorm fusion) as won't-fix.
- Archive github.com/zerfoo/zmf repository.
- Add GGUF writer to zonnx (pivot from ZMF output to GGUF output).
- Update all documentation to reflect changes.

### Out of Scope

- Implementing training checkpoint save/load (design only in this phase).
- New model architectures.
- Performance optimizations.
- Multi-GPU or distributed changes.

### Deliverables

| ID | Description | Acceptance Criteria |
|----|-------------|-------------------|
| D1 | ztensor repo | `go test ./...` passes, CI green, v0.1.0 tagged |
| D2 | ztoken repo | `go test ./...` passes, CI green, v0.1.0 tagged |
| D3 | zerfoo without ZMF | `go test ./...` passes, zmf not in go.mod, ~7,500 lines removed |
| D4 | zerfoo imports ztensor+ztoken | `go test ./...` passes with new imports |
| D5 | zonnx GGUF output | `zonnx convert model.onnx -o model.gguf` works, tests pass |
| D6 | zmf archived | Repository marked archived on GitHub |
| D7 | Updated documentation | All CLAUDE.md, VISION.md, design.md reflect new structure |

---

## 3. Checkable Work Breakdown

### E1: Create ztensor Repository

- [x] T1.1 Create github.com/zerfoo/ztensor repo on GitHub  Owner: TBD  Est: 15m
  - Acceptance: Empty repo exists with Apache 2.0 LICENSE.
- [x] T1.2 Initialize go.mod and repo scaffolding  Owner: TBD  Est: 30m
  - Create go.mod (module github.com/zerfoo/ztensor, go 1.25).
  - Create .gitignore, Makefile with test/lint/bench targets.
  - Deps: T1.1.
  - Acceptance: `go build ./...` succeeds on empty module.
- [x] T1.3 Copy leaf packages: types/, log/, metrics/runtime/  Owner: TBD  Est: 30m
  - Zero internal imports. Copy verbatim, update import paths.
  - Deps: T1.2.
  - Acceptance: `go test ./types/... ./log/... ./metrics/...` passes.
- [x] T1.4 Copy numeric/ package  Owner: TBD  Est: 30m
  - Add float16 and float8 to go.mod. Update imports.
  - Deps: T1.2.
  - Acceptance: `go test ./numeric/...` passes.
- [x] T1.5 Copy internal/ GPU and utility packages  Owner: TBD  Est: 2h
  - Copy: cuda/, cublas/, cudnn/, gpuapi/, hip/, miopen/, nccl/, opencl/,
    clblast/, rocblas/, tensorrt/, xblas/, workerpool/, codegen/.
  - Update all import paths.
  - Deps: T1.2.
  - Acceptance: `go build ./internal/...` compiles. `go vet` passes.
  - Risk: Assembly files (.s) reference Go symbols. Verify with go vet.
- [x] T1.6 Copy device/ package  Owner: TBD  Est: 30m
  - Depends on internal/cuda, internal/hip, internal/opencl.
  - Deps: T1.5.
  - Acceptance: `go test ./device/...` passes.
- [x] T1.7 Copy tensor/ package  Owner: TBD  Est: 1h
  - Depends on device/, internal/gpuapi/, internal/cuda/, float16, float8.
  - Copy testing/testutils/ too.
  - Deps: T1.5, T1.6.
  - Acceptance: `go test ./tensor/...` passes.
- [x] T1.8 Copy compute/ package  Owner: TBD  Est: 1h
  - Depends on tensor/, numeric/, log/, metrics/, internal/*.
  - Deps: T1.3, T1.4, T1.5, T1.7.
  - Acceptance: `go test ./compute/...` passes.
- [x] T1.9 Copy graph/ package  Owner: TBD  Est: 1h
  - Depends on tensor/, compute/, types/, internal/cuda/.
  - Include fusion.go (it moves to ztensor even though zerfoo no longer
    needs it -- ztensor users might).
  - Deps: T1.3, T1.7, T1.8.
  - Acceptance: `go test ./graph/...` passes.
- [x] T1.10 Run full ztensor test suite and linter  Owner: TBD  Est: 30m
  - `go test ./...`, `go vet ./...`.
  - Deps: T1.9.
  - Acceptance: Zero failures, zero vet errors.
- [x] T1.11 Add CI workflow, GoReleaser, release-please  Owner: TBD  Est: 30m
  - Deps: T1.10.
  - Acceptance: CI passes on push.
- [x] T1.12 Write README.md  Owner: TBD  Est: 30m
  - Deps: T1.10.
  - Acceptance: README has module name, purpose, install, usage example.
- [x] T1.13 Tag v0.1.0  Owner: TBD  Est: 15m
  - Deps: T1.10, T1.11, T1.12.
  - Acceptance: `go get github.com/zerfoo/ztensor@v0.1.0` works.

### E2: Create ztoken Repository

- [x] T2.1 Create github.com/zerfoo/ztoken repo on GitHub  Owner: TBD  Est: 15m
  - Acceptance: Empty repo with Apache 2.0 LICENSE.
- [x] T2.2 Initialize go.mod and scaffolding  Owner: TBD  Est: 15m
  - Deps: T2.1.
  - Acceptance: `go build ./...` succeeds.
- [x] T2.3 Copy pkg/tokenizer/ as root package  Owner: TBD  Est: 30m
  - Rename package to `ztoken`. Include testdata/.
  - Zero internal imports to update.
  - Deps: T2.2.
  - Acceptance: `go test ./...` passes.
- [x] T2.4 Extract GGUF tokenizer loader  Owner: TBD  Est: 1h
  - Copy model/gguf/tokenizer.go to ztoken/gguf/.
  - Define minimal metadata interface to avoid pulling full GGUF parser.
  - Deps: T2.3.
  - Acceptance: `go test ./gguf/...` passes.
  - Risk: May need small amount of GGUF type duplication. Keep under 100 lines.
- [x] T2.5 Run test suite and linter  Owner: TBD  Est: 15m
  - Deps: T2.4.
  - Acceptance: Zero failures.
- [x] T2.6 Add CI, GoReleaser, release-please, README  Owner: TBD  Est: 30m
  - Deps: T2.5.
  - Acceptance: CI passes, README has usage example.
- [x] T2.7 Tag v0.1.0  Owner: TBD  Est: 15m
  - Deps: T2.5, T2.6.
  - Acceptance: `go get github.com/zerfoo/ztoken@v0.1.0` works.

### E3: Remove ZMF from zerfoo

- [x] T3.1 Delete ZMF model loading code  Owner: TBD  Est: 1h
  - Delete: model/builder.go, model/zmf_loader.go, model/zmf_exporter.go,
    model/zmf_mmap.go, model/tensor_encoder.go, model/tensor_decoder.go.
  - Delete associated tests: model/builder_test.go, builder_graph_test.go,
    builder_coverage_test.go, builder_helpers_test.go, zmf_loader_test.go,
    zmf_exporter_test.go, zmf_mmap_test.go, tensor_codec_test.go.
  - Deps: none (can start immediately).
  - Acceptance: Deleted files do not exist. `go build ./...` may fail (fixed
    in T3.2).
- [x] T3.2 Remove ZMF references from model/ and zerfoo.go  Owner: TBD  Est: 1h
  - Update model/adapters.go: remove ZMFModelLoader, ZMFModelExporter.
  - Update zerfoo.go: remove BuildFromZMF and any ZMF-related public API.
  - Update model/model.go: remove ZMFVersion field if present.
  - Remove `github.com/zerfoo/zmf` from go.mod. Run `go mod tidy`.
  - Deps: T3.1.
  - Acceptance: `go build ./...` compiles. `grep zmf go.mod` returns nothing.
- [x] T3.3 Delete graph fusion pass  Owner: TBD  Est: 30m
  - Delete graph/fusion.go and graph/fusion_test.go.
  - Remove FuseRMSNorm call from graph/compile.go.
  - Deps: none (can start immediately).
  - Acceptance: `go build ./graph/...` compiles without fusion.
- [x] T3.4 Update ConstantOfShape to not import zmf  Owner: TBD  Est: 30m
  - layers/core/constantofshape.go imports zmf for dtype constants.
  - Replace zmf dtype references with local constants or ztensor types.
  - Deps: T3.2.
  - Acceptance: `go build ./layers/...` compiles.
- [x] T3.5 Close PR #70 (RMSNorm fusion)  Owner: TBD  Est: 5m
  - Close with comment: "Resolved by ADR-037. ZMF/ONNX path removed. Fusion
    pass moved to ztensor for standalone use."
  - Deps: T3.3.
  - Acceptance: PR #70 closed on GitHub.
- [x] T3.6 Run full zerfoo test suite  Owner: TBD  Est: 1h
  - `go test ./...` with race detector.
  - Deps: T3.2, T3.3, T3.4.
  - Acceptance: Zero test failures.
- [x] T3.7 Run linter and go vet  Owner: TBD  Est: 15m
  - Deps: T3.6.
  - Acceptance: Zero issues.

### E4: Migrate zerfoo to Import ztensor and ztoken

- [x] T4.1 Update go.mod to require ztensor and ztoken  Owner: TBD  Est: 15m
  - Add require directives. Run go mod tidy.
  - Deps: T1.13, T2.7, T3.6.
  - Acceptance: `go mod tidy` succeeds.
- [x] T4.2 Create type aliases for backward compatibility  Owner: TBD  Est: 2h
  - In each migrated package directory, replace source with alias files
    re-exporting from ztensor/ztoken.
  - Deps: T4.1.
  - Acceptance: `go build ./...` compiles.
  - Risk: Go generic type alias syntax (Go 1.24+). Test first.
- [x] T4.3 Update internal imports to ztensor  Owner: TBD  Est: 3h
  - In layers/, inference/, generate/, training/, distributed/, serve/, cmd/,
    model/, features/, data/, config/, health/, shutdown/, registry/:
    replace zerfoo/tensor with ztensor/tensor, etc.
  - Deps: T4.1.
  - Acceptance: `go build ./...` compiles.
- [x] T4.4 Update tokenizer imports to ztoken  Owner: TBD  Est: 30m
  - Replace zerfoo/pkg/tokenizer with ztoken in all importers.
  - Deps: T4.1.
  - Acceptance: `go build ./...` compiles.
- [x] T4.5 Remove migrated source from zerfoo  Owner: TBD  Est: 1h
  - Delete original tensor/, compute/, graph/, numeric/, device/, types/,
    log/, metrics/runtime/, all internal/ GPU packages, testing/testutils/,
    pkg/tokenizer/. Keep only alias files.
  - Deps: T4.2, T4.3, T4.4.
  - Acceptance: No duplicate source. `go build ./...` compiles.
- [x] T4.6 Run full zerfoo test suite  Owner: TBD  Est: 1h
  - Deps: T4.5.
  - Acceptance: Zero test failures.
- [x] T4.7 Run linter and go vet  Owner: TBD  Est: 15m
  - Deps: T4.6.
  - Acceptance: Zero issues.

### E5: Pivot zonnx to GGUF Output

- [x] T5.1 Implement GGUF writer in zonnx  Owner: TBD  Est: 3h
  - Write pkg/gguf/writer.go implementing the GGUF v3 binary format:
    magic, version, tensor count, metadata KV pairs, tensor info array,
    alignment padding, raw tensor data.
  - Support data types: F32, F16, BF16, Q4_0, Q8_0.
  - Deps: none (can start immediately).
  - Acceptance: Writer produces valid GGUF files readable by llama.cpp
    gguf-dump tool.
  - S5.1.1 Unit tests for GGUF writer  Owner: TBD  Est: 1h
    - Test: round-trip write-then-read for each data type.
    - Test: alignment, special values (NaN, Inf), empty tensors.
- [x] T5.2 Map ONNX metadata to GGUF metadata  Owner: TBD  Est: 1h
  - Convert ONNX model config (hidden_size, num_layers, vocab_size, etc.)
    to GGUF key-value metadata using llama.cpp naming conventions
    (e.g., `llama.embedding_length`, `llama.block_count`).
  - Deps: T5.1.
  - Acceptance: Metadata keys match llama.cpp expectations.
  - S5.2.1 Tests for metadata mapping  Owner: TBD  Est: 30m
- [x] T5.3 Map ONNX tensor names to GGUF tensor names  Owner: TBD  Est: 1h
  - Convert ONNX/HuggingFace tensor names to GGUF conventions
    (e.g., `model.layers.0.self_attn.q_proj.weight` to `blk.0.attn_q.weight`).
  - Deps: T5.1.
  - Acceptance: Tensor names match llama.cpp expectations.
  - S5.3.1 Tests for tensor name mapping  Owner: TBD  Est: 30m
- [x] T5.4 Update zonnx CLI to output GGUF  Owner: TBD  Est: 1h
  - Change default output from ZMF to GGUF.
  - `zonnx convert model.onnx -o model.gguf [--quantize q4_0]`
  - Remove ZMF output code path.
  - Remove `github.com/zerfoo/zmf` from zonnx go.mod.
  - Deps: T5.1, T5.2, T5.3.
  - Acceptance: `zonnx convert` produces GGUF. zmf not in go.mod.
  - S5.4.1 Integration test: convert MNIST ONNX to GGUF  Owner: TBD  Est: 30m
- [x] T5.5 Run full zonnx test suite and linter  Owner: TBD  Est: 30m
  - Deps: T5.4.
  - Acceptance: Zero failures.

### E6: Archive zmf and Update Documentation

- [x] T6.1 Archive github.com/zerfoo/zmf on GitHub  Owner: TBD  Est: 5m
  - Mark repository as archived. Add notice to README.
  - Deps: T3.6, T5.5 (both consumers removed).
  - Acceptance: Repo shows "archived" badge. README says "Archived. Use GGUF."
- [x] T6.2 Update root CLAUDE.md  Owner: TBD  Est: 30m
  - Remove zmf from project map. Add ztensor and ztoken.
  - Update dependency graph. Update from 5 repos to 6 active repos.
  - Note zonnx now outputs GGUF.
  - Deps: T4.6, T5.5.
  - Acceptance: CLAUDE.md reflects current structure.
- [x] T6.3 Update zerfoo CLAUDE.md  Owner: TBD  Est: 15m
  - Note tensor/compute/graph are now in ztensor.
  - Note GGUF is the sole model format. Remove ZMF references.
  - Deps: T4.6.
- [x] T6.4 Update docs/VISION.md  Owner: TBD  Est: 15m
  - Reflect GGUF-only strategy and 6-repo ecosystem.
  - Deps: T4.6.
- [x] T6.5 Update docs/design.md  Owner: TBD  Est: 30m
  - Remove Section 1.4 "Two Execution Paths" (now only one).
  - Update package layout to show ztensor dependency.
  - Remove ZMF/ONNX decomposed path references.
  - Deps: T4.6.
- [x] T6.6 Update docs/adr/README.md  Owner: TBD  Est: 5m
  - Add ADR-037 entry.
  - Deps: none (immediate).

---

## 4. Parallel Work

### Tracks

| Track | Tasks | Description |
|-------|-------|-------------|
| A: ztensor | T1.1-T1.13 | Create and populate ztensor repo |
| B: ztoken | T2.1-T2.7 | Create and populate ztoken repo |
| C: ZMF removal | T3.1-T3.7 | Delete ZMF code from zerfoo |
| D: Migration | T4.1-T4.7 | Update zerfoo imports to ztensor/ztoken |
| E: zonnx pivot | T5.1-T5.5 | GGUF writer and zonnx CLI update |
| F: Docs + archive | T6.1-T6.6 | Archive zmf, update all docs |

### Sync Points

- Tracks A, B, C, E all start in Wave 1 (independent).
- Track D waits for A, B, C to complete (Sync 1).
- Track F waits for C, D, E to complete (Sync 2).

### Maximum Parallelism

**Wave 1** (10 agents, no dependencies):
1. T1.1 -- Create ztensor repo
2. T2.1 -- Create ztoken repo
3. T3.1 -- Delete ZMF model loading code
4. T3.3 -- Delete graph fusion pass
5. T5.1 -- Implement GGUF writer in zonnx
6. T6.6 -- Update ADR index

**Wave 2** (10 agents):
1. T1.2 -- Initialize ztensor scaffolding
2. T1.5 -- Copy internal/ GPU packages
3. T1.3 -- Copy leaf packages
4. T1.4 -- Copy numeric/
5. T2.2 -- Initialize ztoken scaffolding
6. T3.2 -- Remove ZMF references from model/ and zerfoo.go
7. T5.2 -- Map ONNX metadata to GGUF metadata
8. T5.3 -- Map ONNX tensor names to GGUF names

**Wave 3** (10 agents):
1. T1.6 -- Copy device/
2. T1.7 -- Copy tensor/
3. T2.3 -- Copy tokenizer to ztoken
4. T2.4 -- Extract GGUF tokenizer loader
5. T3.4 -- Update ConstantOfShape zmf imports
6. T5.4 -- Update zonnx CLI to output GGUF

**Wave 4** (8 agents):
1. T1.8 -- Copy compute/
2. T2.5 -- Run ztoken tests
3. T2.6 -- Add ztoken CI and README
4. T3.5 -- Close PR #70
5. T3.6 -- Run zerfoo test suite (ZMF removal)
6. T5.5 -- Run zonnx test suite

**Wave 5** (6 agents):
1. T1.9 -- Copy graph/
2. T2.7 -- Tag ztoken v0.1.0
3. T3.7 -- Run zerfoo linter

**Wave 6** (4 agents):
1. T1.10 -- Run ztensor test suite
2. T1.11 -- Add ztensor CI
3. T1.12 -- Write ztensor README

**Wave 7** (1 agent):
1. T1.13 -- Tag ztensor v0.1.0

**Wave 8** (4 agents, Sync 1: needs T1.13, T2.7, T3.7):
1. T4.1 -- Update zerfoo go.mod
2. T4.2 -- Create type aliases
3. T4.3 -- Update internal imports to ztensor
4. T4.4 -- Update tokenizer imports to ztoken

**Wave 9** (3 agents):
1. T4.5 -- Remove migrated source
2. T4.6 -- Run full zerfoo test suite
3. T4.7 -- Run linter

**Wave 10** (6 agents, Sync 2: needs T4.6, T5.5):
1. T6.1 -- Archive zmf repo
2. T6.2 -- Update root CLAUDE.md
3. T6.3 -- Update zerfoo CLAUDE.md
4. T6.4 -- Update VISION.md
5. T6.5 -- Update design.md

---

## 5. Timeline and Milestones

| ID | Milestone | Exit Criteria | Depends On |
|----|-----------|--------------|------------|
| M1 | ztensor v0.1.0 | Tests pass, CI green, tagged | T1.13 |
| M2 | ztoken v0.1.0 | Tests pass, CI green, tagged | T2.7 |
| M3 | ZMF removed from zerfoo | ~7,500 lines deleted, zmf not in go.mod, PR #70 closed | T3.7 |
| M4 | zonnx outputs GGUF | Converter produces valid GGUF, zmf not in go.mod | T5.5 |
| M5 | Migration complete | zerfoo uses ztensor+ztoken, all tests pass, docs updated | T6.5 |

---

## 6. Risk Register

| ID | Risk | Impact | Likelihood | Mitigation |
|----|------|--------|------------|------------|
| R1 | Go generic type aliases have syntax limitations | High | Medium | Test alias syntax with Go 1.25 before T4.2. Fall back to wrapper types. |
| R2 | Assembly files (.s) break in new module | Medium | Low | Run `go vet` and `go test -race` after copy. |
| R3 | GGUF tokenizer extraction needs too much duplication | Medium | Medium | Define minimal metadata interface. Accept keeping loader in zerfoo if duplication exceeds 100 lines. |
| R4 | Circular dependency between ztensor and zerfoo | High | Low | ztensor must never import zerfoo. Verify with `go mod graph`. |
| R5 | GGUF writer produces files incompatible with llama.cpp | High | Medium | Test round-trip: zonnx writes GGUF, llama.cpp gguf-dump reads it. Use GGUF v3 spec exactly. |
| R6 | ConstantOfShape or other layers have deep zmf dependency | Medium | Low | Audit all zmf import sites (18 files). Replace with local constants. |
| R7 | Removing ZMF breaks users who import zerfoo for ZMF loading | Low | Low | No known external users. The zmf import in zerfoo.go is internal. |

---

## 7. Operating Procedure

### Definition of Done

- All tests pass (`go test ./...` with race detector).
- `go vet ./...` reports zero issues.
- No circular dependencies (`go mod graph` clean).
- Code compiles on macOS ARM64 and Linux ARM64 (DGX Spark).
- Each change is a small, focused commit.

### Review and QA

- Each epic gets a PR with CI checks.
- Verify `go get` works from clean environment after tagging.
- Run zerfoo integration tests (TestProductionSmokeTest).
- Test GGUF files from zonnx with llama.cpp gguf-dump for validation.

### Commit Discipline

- Never commit files from different directories in the same commit.
- One logical change per commit.
- Rebase and merge, never squash or merge commits.

---

## 8. Progress Log

### Change Summary -- 2026-03-16

Phase 17 execution complete. All 52 tasks across 6 epics marked done.

Completed:
- E1 (T1.1-T1.13): ztensor repo created, populated, CI added, tagged v0.1.0
- E2 (T2.1-T2.7): ztoken repo created, populated, CI added, tagged v0.1.0
- E3 (T3.1-T3.7): ZMF removed from zerfoo (~7,500 lines deleted, zmf removed from go.mod)
- E4 (T4.1-T4.7): zerfoo imports migrated to ztensor/ztoken, ~48K lines of source removed
- E5 (T5.1-T5.5): zonnx GGUF writer + metadata/tensor mapping + CLI pivot
- E6 (T6.1-T6.6): zmf archived, all docs updated for 7-repo ecosystem

Deviations:
- T4.2 (type aliases) skipped -- no external consumers, direct import rewrite sufficient
- internal/ packages stayed in zerfoo (Go visibility rules prevent cross-module internal/ imports)
- zmf remains in zonnx go.mod as intermediate for quantization (full removal deferred)

### Change Summary -- 2026-03-15 (revision 2)

Updated Phase 17 plan to incorporate GGUF-only model format decision.

Changes from previous plan revision:
- Added E3 (ZMF removal): T3.1-T3.7 to delete ~7,500 lines of ZMF code.
- Added E5 (zonnx pivot): T5.1-T5.5 to implement GGUF writer and update CLI.
- Added E6 tasks for archiving zmf and updating docs.
- Added checkpoint storage design (GGUF-based) to Context section.
- Created ADR-037 (docs/adr/037-gguf-only-drop-zmf-model-format.md).
- Updated ADR-036 reference in dependency graph (zmf removed from zerfoo).
- Renumbered E3 (was migration) to E4, E4 (was docs) to E6.
- Updated parallel work section: ZMF removal and zonnx pivot run in Wave 1
  alongside ztensor/ztoken creation, increasing parallelism.
- Added R5, R6, R7 to risk register.
- Updated milestones to include M3 (ZMF removed) and M4 (zonnx GGUF).

### Change Summary -- 2026-03-15

New plan created for Phase 17: ztensor and ztoken repository extraction.
Created ADR-036 (docs/adr/036-ztensor-ztoken-repo-extraction.md).

---

## 9. Hand-off Notes

- **Current version:** v1.2.0.
- **Key decisions:**
  - ADR-036: Extract ztensor and ztoken repositories.
  - ADR-037: GGUF as sole model format, drop ZMF.
- **ZMF removal scope:** 18 files import zmf. Core deletion is ~7,500 lines
  (2,357 source + 5,124 tests). ConstantOfShape has a zmf dtype import that
  needs local replacement.
- **Checkpoint design:** Training checkpoints will be GGUF files with
  optimizer tensors prefixed `optimizer.` and training metadata as KV pairs.
  Not implemented yet (SaveModel is a stub).
- **GGUF writer for zonnx:** Must implement GGUF v3 binary format. Reference:
  llama.cpp gguf.py and the GGUF spec. Key: 64-byte alignment for tensor data.
- **DGX Spark:** ssh ndungu@192.168.86.250. Project at ~/zerfoo.
- **Pre-commit hook:** Rejects multi-directory commits.
- **PR #70:** Close as won't-fix after T3.3 (fusion pass deleted).

---

## 10. Appendix

### Files to Delete from zerfoo (E3)

| File | Lines | Purpose |
|------|-------|---------|
| model/builder.go | 824 | Generic graph-from-ZMF construction |
| model/zmf_loader.go | 95 | ZMF file loading |
| model/zmf_exporter.go | 217 | ZMF file export |
| model/zmf_mmap.go | 35 | ZMF mmap loading |
| model/tensor_encoder.go | 136 | Tensor to ZMF protobuf |
| model/tensor_decoder.go | 350 | ZMF protobuf to tensor |
| model/adapters.go | 487 | ZMF adapter types (partial, keep non-ZMF parts) |
| graph/fusion.go | 213 | RMSNorm fusion pass |
| Tests for above | 5,124 | 10 test files |
| **Total** | **~7,481** | |

### Ecosystem After Phase 17

```
float16 --+
float8  --+--> ztensor (tensor, compute, graph, GPU kernels, SIMD)
          |
          +--> zerfoo (GGUF inference, training, serving)
ztoken -----+
          |
          +--> zonnx (ONNX to GGUF converter)

zmf: ARCHIVED
```

6 active repositories. 1 archived.

### GGUF Training Checkpoint Schema

```
# Metadata KV pairs
general.architecture = "llama"
general.name = "my-finetuned-model"
training.epoch = 5
training.global_step = 15000
training.learning_rate = 0.0001
training.optimizer = "adamw"
training.loss = 0.342
training.beta1 = 0.9
training.beta2 = 0.999

# Model weight tensors (same as inference GGUF)
token_embd.weight
blk.0.attn_q.weight
blk.0.attn_k.weight
...

# Optimizer state tensors (additional, stripped for inference)
optimizer.m.token_embd.weight
optimizer.v.token_embd.weight
optimizer.m.blk.0.attn_q.weight
optimizer.v.blk.0.attn_q.weight
...
```

To produce an inference-ready GGUF from a checkpoint: strip all tensors
with `optimizer.` prefix and remove `training.*` metadata keys.
